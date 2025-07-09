from typing import Callable

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float, Bool
from torch import Tensor

import src.sets as sets
from tasks.envs.interfaces.torch_vector_env import TorchVectorEnv
from tasks.interfaces.safe_action_task import SafeActionTask
from tasks.interfaces.safe_state_task import SafeStateTask
from tasks.wrapper.safety import SafetyWrapper

# Actually its TorchVectorEnv & (SafeStateTask | SafeActionTask) but intersections are not supported
SafeEnv = TorchVectorEnv | SafeStateTask | SafeActionTask


class BoundaryProjectionWrapper(SafetyWrapper):
    """
    Projecting an unsafe action to the closest safe action.

    Attributes:
        action_constrained (bool): Whether the action is directly constrained.
        state_constrained (bool): Whether the state is constrained.
        safe_action_gens (int): Number of generators for the safe action set.
        safe_state_gens (int): Number of generators for the safe state set.
    """
    solver_args = {"solve_method": "Clarabel"}
    env: SafeEnv
    action_constrained: bool
    state_constrained: bool
    safe_action_gens: int
    safe_state_gens: int

    def __init__(self,
                 env: SafeEnv,
                 lin_state: list[float],
                 lin_action: list[float],
                 lin_noise: list[float]):
        self.action_constrained = False
        self.state_constrained = False
        if "action-constrained" in env.constrains():
            self.action_constrained = True
            self.safe_action_gens = env.num_action_gens
        if "state-constrained" in env.constrains():
            self.state_constrained = True
            self.safe_state_gens = env.num_state_gens

        SafetyWrapper.__init__(self, env, lin_state, lin_action, lin_noise)

    def actions(
            self, actions: Float[Tensor, "num_envs action_dim"]
    ) -> Float[Tensor, "num_envs action_dim"]:
        reachable_set = self.env.reachable_set()
        # This is an overapproximation so it may not intersect
        projectable = self.env.feasible_set.intersects(reachable_set)

        safe_actions = actions.clone()
        if projectable.any():
            safe_actions[projectable] = self.safety_layer(actions[projectable],
                                                            projectable)
        if safe_actions.isnan().any() or safe_actions.isinf().any():
            raise ValueError(f"""
            Safe actions are NaN. 
            {self.env.state[projectable][safe_actions.isnan().any(dim=1)]}
            {actions[projectable][safe_actions.isnan().any(dim=1)]}
            {self.env.state[projectable][safe_actions.isinf().any(dim=1)]}
            {actions[projectable][safe_actions.isinf().any(dim=1)]}
            """)

        safe_actions = torch.clamp(safe_actions, self.lower_clip, self.upper_clip)

        self.safe_actions = safe_actions

        self.interventions += ((~torch.isclose(safe_actions, actions)).count_nonzero(
            dim=1) == self.action_dim).sum().item()
        return safe_actions

    def construct_safety_func(self) -> Callable[
        [
            Float[Tensor, "batch_dim action_dim"],
            Bool[Tensor, "batch_dim"]
        ],
        Float[Tensor, "batch_dim action_dim"]]:
        """
        Construct the function which ensures safety by projecting the action to the
        closest safe action.

        Returns:
            Callable[
            [
                Float[Tensor, "batch_dim action_dim"],
                Bool[Tensor, "batch_dim"]
            ],
            Float[Tensor, "batch_dim action_dim"]]: The safety function.
        """
        orthogonal_safe_boundary_fn = self.construct_orthogonal_safe_boundary()

        def safety_func(actions: Float[Tensor, "batch_dim action_dim"],
                        mask: Bool[Tensor, "batch_dim"]
                        ) -> Float[Tensor, "batch_dim action_dim"]:
            parameters = [actions]
            if self.action_constrained:
                parameters += [*self.env.safe_action_set()[mask]]
            if self.state_constrained:
                parameters += [self.env.state[mask], *self.env.safe_state_set()[mask]]

            safe_actions = orthogonal_safe_boundary_fn(*parameters,
                                                       solver_args=self.solver_args)[0]

            return safe_actions

        return safety_func

    def construct_orthogonal_safe_boundary(self) -> CvxpyLayer:
        """
        Construct the CvxpyLayer that determines the closest safe action.

        Returns:
            CvxpyLayer: Can be treated like a function.
        """
        action = cp.Parameter(self.action_dim)
        safe_boundary = cp.Variable(self.action_dim)
        objective = cp.Minimize(cp.sum_squares(action - safe_boundary))

        parameters = [action]
        constraints = self.feasibility_constraints(safe_boundary)

        if self.action_constrained:
            constraint, params = self.action_safety_constraints(safe_boundary)
            constraints += constraint
            parameters += params

        if self.state_constrained:
            constraint, params = self.state_safety_constraints(safe_boundary)
            constraints += constraint
            parameters += params

        problem = cp.Problem(objective, constraints)

        projection_layer = CvxpyLayer(problem,
                                      parameters=parameters,
                                      variables=[safe_boundary])
        return projection_layer

    def feasibility_constraints(self, action: cp.Expression | np.ndarray) \
            -> list[bool | cp.Constraint]:
        """
        Construct feasibility constraints by ensuring containment in the control set.

        Args:
            action (cp.Expression | np.ndarray): The action to constrain.

        Returns:
            list[bool | cp.Constraint]: The constraints.
        """
        return [
            self.env.single_action_space.low <= action,
            self.env.single_action_space.high >= action,
        ]

    def action_safety_constraints(self,
                                  center: cp.Expression | np.ndarray,
                                  generator: cp.Expression | np.ndarray = None) \
            -> tuple[list[cp.Constraint], list[cp.Parameter]]:
        """
        Construct safety constraints by ensuring containment in the CTRL set.

        Args:
            center (cp.Expression | np.ndarray): Center of the safe action.
            generator (cp.Expression | np.ndarray): Generator of the safe action.

        Returns:
            tuple[list[cp.Constraint], list[cp.Parameter]]: The constraints and parameters.

        """
        safe_action_center = cp.Parameter(self.action_dim)
        safe_action_generator = cp.Parameter((self.action_dim, self.safe_action_gens))

        if generator is None:
            constraints = sets.Zonotope.point_containment_constraints(
                center,
                safe_action_center,
                safe_action_generator
            )
        else:
            constraints = sets.Zonotope.zonotope_containment_constraints(
                center,
                generator,
                safe_action_center,
                safe_action_generator
            )
        return constraints, [safe_action_center, safe_action_generator]

    def state_safety_constraints(self,
                                 action: cp.Expression | np.ndarray) \
            -> tuple[list[cp.Constraint], list[cp.Parameter]]:
        """
        Construct safety constraints by ensuring containment in the safe state set.

        Args:
            action (cp.Expression | np.ndarray): The action to take.

        Returns:
            tuple[list[cp.Constraint], list[cp.Parameter]]: The constraints and parameters.
        """
        state = cp.Parameter(self.state_dim)
        safe_state_center = cp.Parameter(self.state_dim)
        safe_state_generator = cp.Parameter((self.state_dim, self.safe_state_gens))

        next_state_center, next_state_generator = self.linear_step(action, state)
        if self.env.stochastic:
            constraints = sets.Zonotope.zonotope_containment_constraints(
                next_state_center,
                next_state_generator,
                safe_state_center,
                safe_state_generator
            )
        else:
            constraints = sets.Zonotope.point_containment_constraints(
                next_state_center,
                safe_state_center,
                safe_state_generator
            )
        return constraints, [state, safe_state_center, safe_state_generator]
