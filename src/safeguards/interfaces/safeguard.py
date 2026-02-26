from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import cvxpy as cp
import numpy as np
from torch import Tensor
from beartype import beartype
from jaxtyping import  Float, jaxtyped
from gymnasium.vector import VectorActionWrapper  # used to parallize the enviroments 

import src.sets as sets
from envs.simulators.interfaces.simulator import Simulator
from envs.interfaces.safe_state_env import SafeStateEnv
from envs.interfaces.safe_action_env import SafeActionEnv


# Actually its Simulator & (SafeStateEnv | SafeActionEnv) but this is not supported
SafeEnv = Simulator | SafeStateEnv | SafeActionEnv


class Safeguard(VectorActionWrapper, ABC):
    """
    Ensuring safety of actions according to some safe set.
    """
    env: SafeEnv
    solver_args = {"solve_method": "Clarabel"}

    @jaxtyped(typechecker=beartype)
    def __init__(self, env: SafeEnv, regularisation_coefficient, **kwargs):
        """
        Args:
            env: A custom secured, pytorch-based environment.
        """
        super().__init__(env)
        self.regularisation_coefficient = regularisation_coefficient
        self.batch_dim = self.env.num_envs
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim

        self.action_constrained = False
        self.state_constrained = False
        if isinstance(env, SafeActionEnv):
            self.action_constrained = True
        if isinstance(env, SafeStateEnv):
            self.state_constrained = True
            self.safe_state_gens = env.num_state_gens

        if self.action_constrained:
            self.safe_action_gens = self.env.num_action_gens
        else:
            self.safe_action_gens = 2 * self.action_dim


        self.constant_mat, self.state_mat, self.action_mat, self.noise_mat = self.env.linear_dynamics()

        self.safe_action = None
        self.interventions = 0



    @jaxtyped(typechecker=beartype)
    def actions(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        reachable_set = self.env.reachable_set()
        # This is an overapproximation so it may not intersect
        projectable = self.env.state_set.intersects(reachable_set)

        safe_action = torch.where(projectable.unsqueeze(1), self.safeguard(action), action)
        nan_mask = safe_action.isnan().any(dim=1) | safe_action.isinf().any(dim=1)
        bad_mask = projectable & nan_mask

        if bad_mask.any():
            raise ValueError(f"""
            Safe action are NaN or Inf.
            States: {self.env.state[bad_mask]}
            Actions: {action[bad_mask]}
            Safe actions: {safe_action[bad_mask]}
            """)

        self.safe_action = safe_action
        self.initial_action = action
        
        self.interventions += ((~torch.isclose(safe_action, action)).count_nonzero(dim=1) == self.action_dim).sum().item()
        return safe_action

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def safeguard(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        """
        Safeguard the action to ensure safety.

        Args:
            action: The action to safeguard.

        Returns:
            The safeguarded action.
        """
        pass

    @jaxtyped(typechecker=beartype)
    def regularisation(self,
                        action: Float[Tensor, "buffer_size {self.batch_dim} {self.action_dim}"],
                        safe_action: Float[Tensor, "buffer_size {self.batch_dim} {self.action_dim}"],
                        **kwargs
                        ) -> Float[Tensor, "..."]:
        """
        Compute the safeguard regularisation loss for the given action.

        Args:
            action: The action to compute the loss for.
        Returns:
            The safeguard regularisation loss.
        """
        return self.regularisation_coefficient * torch.nn.functional.mse_loss(safe_action, action)
    
    @jaxtyped(typechecker=beartype)
    def safeguard_metrics(self, 
                          safeguard_metrics_dict: Optional[dict[str, Any]] = None,
                          safeguard_metrics_dict_training_only:  Optional[dict[str, Any]] = None,
                          **kwargs
                          ) -> dict[str, Any]:
        """
        Get metrics related to the safeguard.

        Args:
            safeguard_metrics_dict: A dictionary to store metrics.
            safeguard_metrics_dict_training_only: A dictionary to store metrics only in training mode. 
                                                    Useful for incorporating metrics into the loss function.
            keyword arguments for compatibility
        Returns:
            A dictionary of metrics.
        """

        if safeguard_metrics_dict is None:
            safeguard_metrics_dict = {}

        if safeguard_metrics_dict_training_only is None:
            safeguard_metrics_dict_training_only = {}

        if torch.is_grad_enabled(): # in training mode we do not store metrics
            return safeguard_metrics_dict_training_only

        def compute_generic_constraint_violation(phase, action,metrics_dict: dict[str, Any] = {}):
            data = self.safe_action_set()
            data.setup_constraints()
            processed_action = data.pre_process_action(action)
            metrics_dict[f"{phase}_eq_violation"] = data.equality_constraint_violation(None, processed_action).square().sum(dim=1)
            metrics_dict[f"{phase}_ineq_violation"] = data.inequality_constraint_violation(None, processed_action).square().sum(dim=1)
            metrics_dict[f"{phase}_constraint_violation"] = metrics_dict[f"{phase}_eq_violation"] + metrics_dict[f"{phase}_ineq_violation"]
            return metrics_dict
        
        
        if not "pre_eq_violation" in safeguard_metrics_dict and not "pre_ineq_violation" in safeguard_metrics_dict:
            safeguard_metrics_dict = compute_generic_constraint_violation("pre", self.initial_action, safeguard_metrics_dict)
        if not "post_eq_violation" in safeguard_metrics_dict and not "post_ineq_violation" in safeguard_metrics_dict:
            safeguard_metrics_dict = compute_generic_constraint_violation("post", self.safe_action, safeguard_metrics_dict)
        safeguard_metrics_dict["projection_distance"] = torch.nn.functional.mse_loss(self.safe_action, self.initial_action)

        return safeguard_metrics_dict
    

    @jaxtyped(typechecker=beartype)
    def linear_step(self,action: cp.Expression | np.ndarray) \
            -> tuple[cp.Expression | np.ndarray, np.ndarray, list[cp.Parameter]]:
        """
        Propagate the system through its linearised dynamics. It is newly linearised at the current state,
        the noise matrix is assumed to be constant, and the linearisation point of the noise matrix is assumed
        to be the noise centre.

        Args:
            action: The action to take.

        Returns:
            The next state center, generator and the parameters.
        """

        constant_mat = cp.Parameter(self.state_dim)
        action_mat = cp.Parameter((self.state_dim, self.action_dim))
        noise_mat = self.noise_mat[0].cpu().numpy()

        lin_action = self.env.action_set.center[0].cpu().numpy()

        noise_generator = self.env.noise_set.generator[0].cpu().numpy()

        next_state_center = constant_mat + action_mat @ (action - lin_action)
        next_state_generator = noise_mat @ noise_generator

        return next_state_center, next_state_generator, [constant_mat, action_mat]

    @jaxtyped(typechecker=beartype)
    def feasibility_constraints(self, action: cp.Expression | np.ndarray) \
            -> list[bool | cp.Constraint]:
        """
        Construct feasibility constraints by ensuring containment in the feasible action set.

        Args:
            action: The action to constrain.

        Returns:
            list: The constraints.
        """
        
        return [
            self.env.action_set.min[0, :].cpu().numpy() <= action,
            self.env.action_set.max[0, :].cpu().numpy() >= action,
        ]

    @jaxtyped(typechecker=beartype)
    def action_safety_constraints(self,
                                  center: cp.Expression | np.ndarray,
                                  generator: cp.Expression | np.ndarray = None) \
            -> tuple[list[cp.Constraint], list[cp.Parameter]]:
        """
        Construct safety constraints by ensuring containment in the safe action set.

        Args:
            center: Center of the safe action.
            generator: Generator of the safe action.

        Returns:
            The constraints and parameters.
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

    @jaxtyped(typechecker=beartype)
    def state_safety_constraints(self, action: cp.Expression | np.ndarray) \
            -> tuple[list[cp.Constraint], list[cp.Parameter]]:
        """
        Construct safety constraints by ensuring containment in the safe state set.

        Args:
            action: The action to take.

        Returns:
            The constraints and parameters.
        """

        safe_state_center = cp.Parameter(self.state_dim)
        safe_state_generator = cp.Parameter((self.state_dim, self.safe_state_gens))

        next_state_center, next_state_generator, parameters = self.linear_step(action)
        constraints = sets.Zonotope.zonotope_containment_constraints(
            next_state_center,
            next_state_generator,
            safe_state_center,
            safe_state_generator
        )

        return constraints, [safe_state_center, safe_state_generator, *parameters]

    @jaxtyped(typechecker=beartype)
    def constraint_parameters(self) -> list[Tensor]:
        parameters = []
        if self.action_constrained:
            parameters += [*self.env.safe_action_set()]
        if self.state_constrained:
            parameters += [*self.env.safe_state_set()]
            constant_mat, _, action_mat, _ = self.env.linear_dynamics()
            parameters += [constant_mat, action_mat]
        return parameters

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def __dir__(self) -> list[str]:
        base = super().__dir__()  # Iterable[str]
        env_attrs = dir(self.env)  # list[str]
        attrs: set[str] = set()
        attrs.update(base)
        attrs.update(env_attrs)
        return sorted(attrs)

    @jaxtyped(typechecker=beartype)
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Tensor, dict[str, Any]]:
        """
        Compatibility shim for Gymnasium's VectorEnv.reset(seed=..., options=...).
        """
        return self.env.reset(seed=seed)


Simulator.register(Safeguard)
