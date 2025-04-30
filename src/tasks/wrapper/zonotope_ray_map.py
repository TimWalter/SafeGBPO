from typing import Callable

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float, Bool
from torch import Tensor

from tasks.wrapper.ray_map import RayMapWrapper, SafeEnv
import src.sets as sets


class ZonotopeRayMapWrapper(RayMapWrapper):
    """
    Projecting actions radially towards the approximated center of the safe action set.
    """

    def __init__(self,
                 env: SafeEnv,
                 lin_state: list[float],
                 lin_action: list[float],
                 lin_noise: list[float],
                 linear_projection: bool,
                 num_generators: int,
                 reuse_safe_set: bool,
                 passthrough: bool):
        """
        Args:
            env(SafeEnv): A custom kinematic, pytorch-based environment.
            lin_state(list[float]): State point to linearise around.
            lin_action(list[float]): Action point to linearise around.
            lin_noise(list[float]): Noise point to linearise around.
            linear_projection(bool): Whether to use linear projection.
            calculate the safe action set center.
            num_generators(int): The number of generators for the zonotope expansion.
            reuse_safe_set(bool): Whether to reuse the safe set for the boundary computation.
            passthrough(bool): Whether to use a passthrough gradient.
        """
        self.num_generators = num_generators
        self.reuse_safe_set = reuse_safe_set
        super().__init__(env, lin_state, lin_action, lin_noise, linear_projection)

        if passthrough:
            self.prev_actions = self.actions
            self.actions = lambda actions: self.PassthroughActions.apply(actions, self.prev_actions)

    class PassthroughActions(torch.autograd.Function):
        @staticmethod
        def forward(ctx, actions, func):
            return func(actions)

        @staticmethod
        def backward(ctx, grad):
            return grad, None

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
        safe_zonotope_fn = self.construct_safe_zonotope()
        if self.reuse_safe_set:
            safe_boundary_fn = self.construct_safe_action_boundary()
        else:
            safe_boundary_fn = self.construct_safe_rci_boundary()

        def projection_func(actions: Float[Tensor, "batch_dim {self.action_dim}"],
                            mask: Bool[Tensor, "batch_dim"]
                            ) -> Float[Tensor, "batch_dim {self.action_dim}"]:
            safe_centers, safe_generators = safe_zonotope_fn(mask)

            if self.reuse_safe_set or (self.action_constrained and not self.state_constrained):
                safe_boundary = safe_boundary_fn(actions, safe_centers, safe_generators)
            else:
                safe_boundary = safe_boundary_fn(actions, safe_centers)

            boundary = self.get_boundary(actions, actions - safe_centers)

            safe_actions = self.project_towards_safe_center(actions, safe_centers,
                                                            safe_boundary, boundary)
            return safe_actions

        return projection_func

    def construct_safe_zonotope(self) -> Callable[
        [
            Bool[Tensor, "batch_dim"]
        ],
        tuple[
            Float[Tensor, "batch_dim {self.action_dim}"],
            Float[Tensor, "batch_dim {self.action_dim} {self.num_generators}"]
        ]
    ]:
        """
        Construct the function to approximate the safe action set by a zonotope.

        Returns:
            Callable[
            [
            Bool[Tensor, "batch_dim"]
            ],
            tuple[
                Float[Tensor, "{self.num_envs} {self.action_dim}"],
                Float[Tensor, "{self.num_envs} {self.action_dim} {self.num_generators}"]
            ]
            ]: The approximation function.
        """
        direction = cp.Parameter((self.action_dim, self.num_generators))

        length = cp.Variable(self.num_generators, nonneg=True)
        center = cp.Variable(self.action_dim)

        generator = direction @ cp.diag(length)

        parameters = [direction]
        constraints = [
            self.env.single_action_space.low <= center - cp.abs(generator).sum(axis=1),
            self.env.single_action_space.high >= center + cp.abs(generator).sum(axis=1),
        ]

        if self.action_constrained:
            constraint, params = self.action_safety_constraints(center, generator)
            constraints += constraint
            parameters += params

        if self.state_constrained:
            state = cp.Parameter(self.state_dim)
            safe_state_center = cp.Parameter(self.state_dim)
            safe_state_generator = cp.Parameter((self.state_dim, self.safe_state_gens))
            parameters += [state, safe_state_center, safe_state_generator]

            next_state_center, next_state_noise_generator = self.linear_step(center, state)
            next_state_action_generator = self.action_mat.cpu().numpy() @ generator
            if self.env.stochastic:
                next_state_generator = cp.hstack([next_state_action_generator,
                                              next_state_noise_generator])
            else:
                next_state_generator = next_state_action_generator

            constraints += sets.Zonotope.zonotope_containment_constraints(
                next_state_center,
                next_state_generator,
                safe_state_center,
                safe_state_generator
            )

        objective = cp.Maximize(cp.geo_mean(length))

        problem = cp.Problem(objective, constraints)

        safe_center_layer = CvxpyLayer(problem,
                                       parameters=parameters,
                                       variables=[center, length])

        def safe_center_fn(mask: Bool[Tensor, "batch_dim"]) -> tuple[
            Float[Tensor, "batch_dim {self.action_dim}"],
            Float[Tensor, "batch_dim {self.action_dim} {self.num_generators}"]
        ]:
            if self.state_constrained:
                directions = torch.rand(mask.sum().item(), self.action_dim, self.num_generators,
                                        device=self.env.device, dtype=torch.float64) * 2 - 1
                directions = directions / torch.linalg.vector_norm(directions, dim=1,
                                                                   keepdim=True)

                parameters = [directions]
                if self.action_constrained:
                    parameters += [*self.env.safe_action_set()[mask]]
                parameters += [self.env.state[mask], *self.env.safe_state_set()[mask]]

                safe_center, lengths = safe_center_layer(*parameters,
                                                         solver_args=self.solver_args)
                safe_generator = directions * lengths.unsqueeze(1)
            elif self.action_constrained:
                safe_center, safe_generator = self.env.safe_action_set()[mask]
            else:
                safe_center = self.env.feasible_set.center
                safe_generator = self.env.feasible_set.generator
            return safe_center, safe_generator

        return safe_center_fn

    def construct_safe_rci_boundary(self) -> Callable[
        [
            Float[Tensor, "batch_dim {self.action_dim}"],
            Float[Tensor, "batch_dim {self.action_dim}"]
        ],
        Float[Tensor, "batch_dim {self.action_dim}"]
    ]:
        """
        Construct the function to compute the safe boundary point along a ray from the
        safe center to the action verified by the rci containment of the next state.

        Returns:
            Callable[
            [
                Float[Tensor, "batch_dim {self.action_dim}"],
                Float[Tensor, "batch_dim {self.action_dim}"]
            ],
            Float[Tensor, "batch_dim {self.action_dim}"]
            ]: The safe boundary function.
        """
        safe_boundary_fn = self.construct_safe_boundary()

        def safe_rci_boundary_fn(
                actions: Float[Tensor, "{self.num_envs} {self.action_dim}"],
                safe_centers: Float[Tensor, "{self.num_envs} {self.action_dim}"]) \
                -> Float[Tensor, "{self.num_envs} {self.action_dim}"]:
            directions = actions - safe_centers
            parameters = [safe_centers, directions]

            if self.action_constrained:
                parameters += [*self.env.safe_action_set()]
            if self.state_constrained:
                parameters += [self.env.state, *self.env.safe_state_set()]

            safe_boundary = safe_boundary_fn(*parameters)
            return safe_boundary

        return safe_rci_boundary_fn

    def construct_safe_action_boundary(self) -> Callable[
        [
            Float[Tensor, "batch_dim {self.action_dim}"],
            Float[Tensor, "batch_dim {self.action_dim}"],
            Float[Tensor, "batch_dim {self.action_dim} {self.num_generators}"]
        ],
        Float[Tensor, "batch_dim {self.action_dim}"]
    ]:
        """
        Construct the function to compute the safe boundary point along a ray from the
        safe center to the action verified by the containment of the action in the
        approximated zonotope.

        Returns:
            Callable[
            [
                Float[Tensor, "batch_dim {self.action_dim}"],
                Float[Tensor, "batch_dim {self.action_dim}"],
                Float[Tensor, "batch_dim {self.action_dim} {self.num_generators}"]
            ],
            Float[Tensor, "batch_dim {self.action_dim}"]
            ]: The safe boundary function.
        """
        action = cp.Parameter(self.action_dim)
        safe_center = cp.Parameter(self.action_dim)
        safe_generator = cp.Parameter((self.action_dim, self.num_generators))

        shift = cp.Variable(1, nonneg=True)

        safe_action_boundary = safe_center + cp.multiply(shift, action - safe_center)

        objective = cp.Maximize(shift)

        safety_constraints = sets.Zonotope.point_containment_constraints(
            safe_action_boundary,
            safe_center,
            safe_generator)

        problem = cp.Problem(objective, safety_constraints)

        shift_layer = CvxpyLayer(problem,
                                 parameters=[action, safe_center, safe_generator],
                                 variables=[shift])

        def safe_boundary_fn(
                actions: Float[Tensor, "{self.num_envs} {self.action_dim}"],
                safe_centers: Float[Tensor, "{self.num_envs} {self.action_dim}"],
                safe_generators: Float[
                    Tensor, "{self.num_envs} {self.action_dim} {self.num_generators}"]) \
                -> Float[Tensor, "{self.num_envs} {self.action_dim}"]:
            shifts = shift_layer(actions,
                                 safe_centers,
                                 safe_generators,
                                 solver_args=self.solver_args)[0]
            return safe_centers + shifts * (actions - safe_centers)

        return safe_boundary_fn
