from typing import Callable

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float, Bool
from torch import Tensor

from tasks.wrapper.ray_map import RayMapWrapper


class OrthogonalRayMapWrapper(RayMapWrapper):
    """
    Projecting unsafe actions orthogonally towards the center of the safe action set.
    """

    def construct_safety_func(self) -> Callable[
        [
            Float[Tensor, "batch_dim action_dim"],
            Bool[Tensor, "batch_dim"]
        ],
        Float[Tensor, "batch_dim action_dim"]]:
        """
        Construct the safety function.

        Returns:
            Callable[
            [
                Float[Tensor, "batch_dim action_dim"],
                Bool[Tensor, "batch_dim"]
            ],
            Float[Tensor, "batch_dim action_dim"]]: The safety function.
        """
        if self.action_constrained:
            action_eval_fn = self.construct_action_safety_evaluation()
        if self.state_constrained:
            state_eval_fn = self.construct_state_safety_evaluation()
        safe_boundary_fn = self.construct_orthogonal_safe_boundary()
        safe_centers_fn = self.construct_safe_centers()

        def safety_func(actions: Float[Tensor, "batch_dim {self.action_dim}"],
                        mask: Bool[Tensor, "batch_dim"]
                        ) -> Float[Tensor, "batch_dim {self.action_dim}"]:
            safe = torch.ones(actions.shape[0], dtype=torch.bool, device=self.env.device)
            if self.action_constrained:
                safe &= action_eval_fn(actions, mask)
            if self.state_constrained:
                safe &= state_eval_fn(actions, mask)

            safe_actions = actions.clone()
            if not safe.all():
                parameters = [actions[~safe]]

                if self.action_constrained:
                    parameters += [*self.env.safe_action_set()[mask][~safe]]

                if self.state_constrained:
                    parameters += [self.env.state[mask][~safe]]
                    parameters += [*self.env.safe_state_set()[mask][~safe]]

                safe_boundary = safe_boundary_fn(*parameters,
                                                 solver_args=self.solver_args)[0]

                safe_centers = safe_centers_fn(parameters[0], safe_boundary,
                                               *parameters[1:])

                boundary = self.get_boundary(parameters[0], parameters[0] - safe_centers)

                safe_actions[~safe] = self.project_towards_safe_center(
                    parameters[0], safe_centers, safe_boundary, boundary
                )
            return safe_actions

        return safety_func


    def construct_action_safety_evaluation(self) -> Callable[
        [
            Bool[Tensor, "batch_dim"],
            Float[Tensor, "batch_dim {self.action_dim}"]
        ],
        Bool[Tensor, "batch_dim"]]:
        """
        Construct the function to compute whether the actions are safe.

        Returns:
            Callable[
            [
                Float[Tensor, "batch_dim {self.action_dim}"],
                Bool[Tensor, "batch_dim"]
            ],
            Bool[Tensor, "batch_dim"]]: The safe function.
        """
        action = cp.Parameter(self.action_dim)

        constraints, parameters = self.action_safety_constraints(action)

        objective = cp.Minimize(constraints[1].args[0])
        problem = cp.Problem(objective, constraints[0:1])

        weight = constraints[1].args[0].args[0]
        safety_layer = CvxpyLayer(problem,
                                  parameters=[action, *parameters],
                                  variables=[weight])

        def action_safety_evaluation_fn(
                actions: Float[Tensor, "batch_dim {self.action_dim}"],
                mask: Bool[Tensor, "batch_dim"]) \
                -> Bool[Tensor, "batch_dim"]:
            weights = safety_layer(actions, *self.env.safe_action_set()[mask],
                                   solver_args=self.solver_args)[0]
            norm = torch.linalg.vector_norm(weights, dim=1, ord=torch.inf)
            return norm <= 1.001  # TODO why does cvxpylayers give different results?

        return action_safety_evaluation_fn


    def construct_state_safety_evaluation(self) -> Callable[
        [
            Bool[Tensor, "batch_dim"],
            Float[Tensor, "batch_dim action_dim"]
        ],
        Bool[Tensor, "batch_dim"]]:
        """
        Construct the function to compute whether the produced next state zonotope is
        contained in the safe state set.

        Returns:
            Callable[
            [
                Bool[Tensor, "batch_dim"],
                Float[Tensor, "batch_dim action_dim"]
            ],
            Bool[Tensor, "batch_dim"]]: The safe function.
        """
        action = cp.Parameter(self.action_dim)

        constraints, parameters = self.state_safety_constraints(action)

        objective = cp.Minimize(constraints[-1].args[0])
        problem = cp.Problem(objective, constraints[:-1])

        weight = constraints[-1].args[0].args[0].args[0].args[1].args[0]
        mapping = constraints[-1].args[0].args[0].args[0].args[0]
        safety_layer = CvxpyLayer(problem,
                                  parameters=[action, *parameters],
                                  variables=[weight, mapping])

        def state_safety_evaluation_fn(
                actions: Float[Tensor, "batch_dim {self.action_dim}"],
                mask: Bool[Tensor, "batch_dim"]) \
                -> Bool[Tensor, "batch_dim"]:
            weights, mappings = safety_layer(actions,
                                             self.env.state[mask],
                                             *self.env.safe_state_set()[mask],
                                             solver_args=self.solver_args)
            norm = torch.linalg.matrix_norm(torch.cat([mappings,
                                                       torch.reshape(weights, (
                                                           mask.sum(), -1, 1))],
                                                      dim=2), ord=torch.inf)
            return norm <= 1

        return state_safety_evaluation_fn


    def construct_safe_centers(self) -> Callable[
        [
            Float[Tensor, "unsafe_dim {self.action_dim}"],
            Float[Tensor, "unsafe_dim {self.action_dim}"],
            ...
        ],
        Float[Tensor, "unsafe_dim {self.action_dim}"]]:
        """
        Construct the function to approximate the center of the safe action set by
        taking the middle point between the safe boundary point and a second boundary
        point in opposite direction.

        Returns:
            Callable[
            [
                Float[Tensor, "unsafe_dim {self.action_dim}"],
                Float[Tensor, "unsafe_dim {self.action_dim}"],
                ...
            ],
            Float[Tensor, "unsafe_dim {self.action_dim}"]
            ]: The safe center function.
        """
        safe_boundary2_fn = self.construct_safe_boundary()

        def safe_center_fn(actions: Float[Tensor, "unsafe_dim {self.action_dim}"],
                           safe_boundary: Float[Tensor, "unsafe_dim {self.action_dim}"],
                           *args
                           ) -> Float[Tensor, "unsafe_dim  {self.action_dim}"]:
            direction = safe_boundary - actions
            safe_boundary2 = safe_boundary2_fn(safe_boundary, direction, *args)
            safe_centers = (safe_boundary + safe_boundary2) / 2
            return safe_centers

        return safe_center_fn
