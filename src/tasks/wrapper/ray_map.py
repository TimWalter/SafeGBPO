from abc import ABC
from typing import Callable

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float
from torch import Tensor

from tasks.wrapper.boundary_projection import BoundaryProjectionWrapper, SafeEnv


class RayMapWrapper(BoundaryProjectionWrapper, ABC):
    """
    Projecting actions radially towards the center.
    """

    def __init__(self,
                 env: SafeEnv,
                 lin_state: list[float],
                 lin_action: list[float],
                 lin_noise: list[float],
                 linear_projection: bool):
        """
        Args:
            env(SafeEnv): A custom secured, pytorch-based environment.
            lin_state(list[float]): State point to linearise around.
            lin_action(list[float]): Action point to linearise around.
            lin_noise(list[float]): Noise point to linearise around.
            linear_projection(bool): Whether to use linear projection.
        """
        super().__init__(env, lin_state, lin_action, lin_noise)
        self.high = torch.from_numpy(self.env.single_action_space.high).to(
            self.env.device, dtype=torch.float64)
        self.low = torch.from_numpy(self.env.single_action_space.low).to(
            self.env.device, dtype=torch.float64)
        self.linear_projection = linear_projection

    def project_towards_safe_center(
            self,
            actions: Float[Tensor, "batch_dim {self.action_dim}"],
            safe_centers: Float[Tensor, "batch_dim {self.action_dim}"],
            safe_boundaries: Float[Tensor, "batch_dim {self.action_dim}"],
            boundaries: Float[Tensor, "batch_dim {self.action_dim}"]) \
            -> Float[Tensor, "batch_dim {self.action_dim}"]:
        """
        Project the action towards the safe center.

        Args:
            actions (Float[Tensor, "batch_dim {self.action_dim}"]): The action to project.
            safe_centers (Float[Tensor, "batch_dim {self.action_dim}"]): The center of the safe set.
            safe_boundaries (Float[Tensor, "batch_dim {self.action_dim}"]): The boundary of the safe set.
            boundaries (Float[Tensor, "batch_dim {self.action_dim}"]): The boundary of the feasible set.

        Returns:
            Float[Tensor, "batch_dim {self.action_dim}"]: The projected action
        """
        full_dist = torch.linalg.vector_norm(safe_centers - boundaries, dim=1,
                                             keepdim=True)
        safe_dist = torch.linalg.vector_norm(safe_centers - safe_boundaries, dim=1,
                                             keepdim=True)
        action_dist = torch.linalg.vector_norm(safe_centers - actions, dim=1,
                                                  keepdim=True)
        ray = (actions - safe_centers) / action_dist

        if self.linear_projection:
            safe_actions = safe_centers + ray * safe_dist / full_dist * action_dist
        else:
            safe_actions = safe_centers + ray * safe_dist * torch.tanh(action_dist/safe_dist) / torch.tanh(full_dist/safe_dist)
        return safe_actions

    def get_boundary(self,
                     actions: Float[Tensor, "batch_dim {self.action_dim}"],
                     directions: Float[Tensor, "batch_dim {self.action_dim}"]) \
            -> Float[Tensor, "batch_dim {self.action_dim}"]:
        """
        Compute the boundary of the feasible action set along the given direction.

        Args:
            actions (Float[Tensor, "batch_dim {self.action_dim}"]): The current action.
            directions (Float[Tensor, "batch_dim {self.action_dim}"]): The direction to move.

        Returns:g
            Float[Tensor, "batch_dim {self.action_dim}"]: The boundary of the feasible set.
        """

        shift_high = torch.full_like(actions, torch.inf)
        shift_low = torch.full_like(actions, torch.inf)

        mask = directions != 0
        shift_high[mask] = (self.high.expand_as(actions)[mask] - actions[mask]) / \
                           directions[mask]
        shift_low[mask] = (self.low.expand_as(actions)[mask] - actions[mask]) / \
                          directions[mask]

        shifts = torch.cat([shift_high, shift_low], dim=-1)
        shift = torch.min(torch.where(shifts < 0, torch.inf, shifts), dim=-1,
                          keepdim=True).values

        boundary = actions + shift * directions

        return boundary

    def construct_safe_boundary(self) -> Callable[
        [
            Float[Tensor, "batch_dim {self.action_dim}"],
            Float[Tensor, "batch_dim {self.action_dim}"],
            ...
        ],
        Float[Tensor, "batch_dim {self.action_dim}"]
    ]:
        """
        Construct the function to calculate the safe action set boundary in a given
        direction.

        Returns:
            Callable[
            [
                Float[Tensor, "batch_dim {self.action_dim}"],
                Float[Tensor, "batch_dim {self.action_dim}"],
                ...
            ],
            Float[Tensor, "batch_dim {self.action_dim}"]
            ]: The safe boundary function.
        """
        action = cp.Parameter(self.action_dim)
        direction = cp.Parameter(self.action_dim)

        shift = cp.Variable(1, nonneg=True)
        objective = cp.Maximize(shift)

        safe_boundary = action + cp.multiply(shift, direction)

        parameters = [action, direction]
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

        shift_layer = CvxpyLayer(problem, parameters=parameters, variables=[shift])

        def safe_boundary_fn(actions, directions, *args) \
                -> Float[Tensor, "batch_dim {self.action_dim}"]:
            params = [actions, directions] + list(args)
            shifts = shift_layer(*params, solver_args=self.solver_args)[0]
            safe_boundaries = actions + shifts * directions
            return safe_boundaries

        return safe_boundary_fn
