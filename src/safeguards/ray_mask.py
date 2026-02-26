from typing import Any, Union

import torch
import cvxpy as cp
from torch import Tensor
from beartype import beartype
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float, jaxtyped

from safeguards.interfaces.safeguard import Safeguard, SafeEnv
from safeguards.boundary_projection import BoundaryProjectionSafeguard
import src.sets as sets


@jaxtyped(typechecker=beartype)
class RayMaskSafeguard(Safeguard):
    """
    Projecting actions radially towards the center.
    """

    class PassthroughAction(torch.autograd.Function):
        """ A custom autograd function to pass through action gradients without modification."""

        @staticmethod
        def forward(ctx, action, func):
            return func(action)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            return *grad_outputs, None

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 env: SafeEnv,
                 regularisation_coefficient: float,
                 linear_projection: bool,
                 zonotopic_approximation: bool,
                 passthrough: bool,
                 **kwargs):
        """
        Args:
            env: A custom secured, pytorch-based environment.
            linear_projection: Whether to use linear projection.
            zonotopic_approximation: Whether to use zonotopic approximation.
            passthrough: Whether to use passthrough gradients
        """
        super().__init__(env, regularisation_coefficient)
        self.linear_projection = linear_projection
        if passthrough:
            self.prev_action = self.actions
            self.actions = lambda action: self.PassthroughAction.apply(action, self.prev_action)

        if zonotopic_approximation:
            self.distance_approximations = self.zonotopic_approximation
        else:
            self.distance_approximations = self.orthogonal_approximation

        self.zonotope_expansion_layer = None
        self.zonotope_distance_layer = None
        self.boundary_projection_safeguard = None
        self.implicit_zonotope_distance_layer = None
    
    @jaxtyped(typechecker=beartype)
    def safeguard(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        """
        Safeguard the action to ensure safety.

        Args:
            action: The action to safeguard.

        Returns:
            The safeguarded action.
        """

        if self.state_constrained:
            safe_center, safe_dist, feasible_dist = self.distance_approximations(action)
        else:
            safe_center = self.env.safe_action_set().center
            safe_dist, feasible_dist = self.compute_distances(action, safe_center, self.env.safe_action_set().generator)

        action_dist = torch.linalg.vector_norm(action - safe_center, dim=1, ord=2, keepdim=True)
        directions = (action - safe_center) / (action_dist + 1e-8)

        central = action_dist < 1e-8  

        safe_action = torch.where(
            central,
            safe_center,
            safe_center + directions * self.radial_mapping(action_dist, safe_dist, feasible_dist)
        )

        return safe_action

    @jaxtyped(typechecker=beartype)
    def radial_mapping(self,
                       action_dist: Float[Tensor, "{self.batch_dim} 1"],
                       safe_dist: Float[Tensor, "{self.batch_dim} 1"],
                       feasible_dist: Float[Tensor, "{self.batch_dim} 1"]) \
            -> Float[Tensor, "{self.batch_dim} 1"]:
        """
        Project the action towards the safe center.

        Args:
            action_dist: The distance of the action from the safe center.
            safe_dist: The distance to the safe boundary from the safe center.
            feasible_dist: The distance to the feasible boundary from the safe center.

        Returns:
            The radial mapping
        """
        if self.linear_projection:
            mapping = action_dist / feasible_dist
        else:
            mapping = torch.tanh(action_dist / safe_dist) / torch.tanh(feasible_dist / safe_dist)

        return mapping

    @jaxtyped(typechecker=beartype)
    def zonotopic_approximation(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) -> tuple[
        Float[Tensor, "{self.batch_dim} {self.action_dim}"],
        Float[Tensor, "{self.batch_dim} 1"],
        Float[Tensor, "{self.batch_dim} 1"],
    ]:
        """
        Approximate the safe action set by a zonotope and compute the distances to the safe center and feasible boundary.

        Args:
            action: The action to safeguard.

        Returns:
            Safe centers, distances to the safe boundary, and distances to the feasible boundary.
        """
        safe_center, safe_generator = self.zonotope_expansion()
        safe_dist, feasible_dist = self.compute_distances(action, safe_center, safe_generator)
        return safe_center, safe_dist, feasible_dist

    @jaxtyped(typechecker=beartype)
    def zonotope_expansion(self) -> tuple[
        Float[Tensor, "{self.batch_dim} {self.action_dim}"],
        Float[Tensor, "{self.batch_dim} {self.action_dim} {self.action_dim*2}"]
    ]:
        """
        Approximate the safe action set by an expanding zonotope.

        Returns:
            A tuple containing the safe center and the safe generator of the zonotope.
        """
        if self.zonotope_expansion_layer is None:
            direction = cp.Parameter((self.action_dim, self.action_dim * 2))
            parameters = [direction]

            length = cp.Variable(self.action_dim * 2, nonneg=True)
            center = cp.Variable(self.action_dim)

            objective = cp.Maximize(cp.geo_mean(length))

            generator = direction @ cp.diag(length)
            constraints = [
                # Feasibility constraints
                self.env.action_set.min[0, :].cpu().numpy() <= center - cp.abs(generator).sum(axis=1),
                self.env.action_set.max[0, :].cpu().numpy() >= center + cp.abs(generator).sum(axis=1),
            ]
            if self.action_constrained:
                constraint, params = self.action_safety_constraints(center, generator)
                constraints += constraint
                parameters += params
            if self.state_constrained:
                safe_state_center = cp.Parameter(self.state_dim)
                safe_state_generator = cp.Parameter((self.state_dim, self.safe_state_gens))
                parameters += [safe_state_center, safe_state_generator]

                next_state_center, next_state_noise_generator, linear_parameters = self.linear_step(center)
                parameters += linear_parameters
                next_state_action_generator = self.action_mat[0].cpu().numpy() @ generator
                next_state_generator = cp.hstack([next_state_action_generator,
                                                  next_state_noise_generator])
                constraints += sets.Zonotope.zonotope_containment_constraints(
                    next_state_center,
                    next_state_generator,
                    safe_state_center,
                    safe_state_generator
                )

            problem = cp.Problem(objective, constraints)
            self.zonotope_expansion_layer = CvxpyLayer(problem, parameters=parameters, variables=[center, length])

        directions = torch.rand(self.batch_dim, self.action_dim, self.action_dim * 2) * 2 - 1
        directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
        parameters = [directions] + self.constraint_parameters()

        safe_center, lengths = self.zonotope_expansion_layer(*parameters, solver_args=self.solver_args)
        safe_generator = directions * lengths.unsqueeze(1)

        return safe_center, safe_generator

    @jaxtyped(typechecker=beartype)
    def compute_distances(self,
                          action: Float[Tensor, "{self.batch_dim} {self.action_dim}"],
                          center: Float[Tensor, "{self.batch_dim} {self.action_dim}"],
                          generator: Union[Float[Tensor, "{self.batch_dim} {self.action_dim} {self.safe_action_gens}"],
                          Float[Tensor, "{self.batch_dim} {self.action_dim} {self.action_dim*2}"]]
                          ) -> tuple[
        Float[Tensor, "{self.batch_dim} 1"],
        Float[Tensor, "{self.batch_dim} 1"]
    ]:
        """
        Compute the distance from the safe center to the zonotope boundary along the direction towards the action, as
        well as the distance to the feasible boundary along the same direction.

        Args:
            action: The action to compute the direction.
            center: The center of the safe action set.
            generator:  The generator of the safe action set.

        Returns:
            A tuple containing the distance to the safe boundary and the distance to the feasible boundary.
        """
        if self.zonotope_distance_layer is None:
            directions = cp.Parameter(self.action_dim)
            cp_center = cp.Parameter(self.action_dim)
            cp_generator = cp.Parameter(
                (self.action_dim, self.action_dim * 2 if self.state_constrained else self.safe_action_gens))
            parameters = [directions, cp_center, cp_generator]

            dist = cp.Variable(nonneg=True)

            objective = cp.Maximize(dist)

            zonotope_boundary = cp_center + cp.multiply(dist, directions)
            constraints = sets.Zonotope.point_containment_constraints(
                zonotope_boundary,
                cp_center,
                cp_generator
            )

            problem = cp.Problem(objective, constraints)
            self.zonotope_distance_layer = CvxpyLayer(problem, parameters=parameters, variables=[dist])

        directions = (action - center) / (torch.linalg.vector_norm(action - center, dim=1, keepdim=True) + 1e-8)
        safe_dist = self.zonotope_distance_layer(directions, center, generator, solver_args=self.solver_args)[
            0].unsqueeze(1)
        feasible_dist = self.axis_aligned_unit_box_dist(center, directions)
        return safe_dist, feasible_dist

    @jaxtyped(typechecker=beartype)
    def axis_aligned_unit_box_dist(self, safe_center: Float[Tensor, "{self.batch_dim} {self.action_dim}"],
                                   direction: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} 1"]:
        """
        Compute the boundary of the feasible action set along the given direction.

        Args:
            safe_center: The center of the safe action set.
            direction: The direction to move.

        Returns:
            The boundary of the feasible set.
        """

        distance_high = torch.full_like(safe_center, torch.inf)
        distance_low = torch.full_like(safe_center, torch.inf)

        mask = direction != 0
        distance_high[mask] = (1 - safe_center[mask]) / direction[mask]
        distance_low[mask] = (-1 - safe_center[mask]) / direction[mask]

        distances = torch.cat([distance_high, distance_low], dim=-1)
        distance = torch.min(torch.where(distances < 0, torch.inf, distances), dim=-1, keepdim=True).values
        return distance

    def orthogonal_approximation(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) -> tuple[
        Float[Tensor, "{self.batch_dim} {self.action_dim}"],
        Float[Tensor, "{self.batch_dim} 1"],
        Float[Tensor, "{self.batch_dim} 1"],
    ]:
        if self.boundary_projection_safeguard is None:
            self.boundary_projection_safeguard = BoundaryProjectionSafeguard(self.env, self.regularisation_coefficient)
        if self.implicit_zonotope_distance_layer is None:
            direction = cp.Parameter(self.action_dim)
            starting_point = cp.Parameter(self.action_dim)
            parameters = [starting_point, direction]

            dist = cp.Variable(nonneg=True)

            objective = cp.Maximize(dist)

            zonotope_boundary = starting_point + cp.multiply(dist, direction)
            constraints = self.feasibility_constraints(zonotope_boundary)
            if self.action_constrained:
                constraint, params = self.action_safety_constraints(zonotope_boundary)
                constraints += constraint
                parameters += params
            if self.state_constrained:
                # Cannot multiply parameters for dpp compliance
                safe_state_center = cp.Parameter(self.state_dim)
                safe_state_generator = cp.Parameter((self.state_dim, self.safe_state_gens))
                constant_mat = cp.Parameter(self.state_dim)
                action_mat = cp.Parameter((self.state_dim, self.action_dim))
                action_mat_times_starting_point = cp.Parameter(self.state_dim)
                action_mat_times_direction = cp.Parameter(self.state_dim)

                noise_mat = self.noise_mat[0].cpu().numpy()
                lin_action = self.env.action_set.center[0].cpu().numpy()
                noise_generator = self.env.noise_set.generator[0].cpu().numpy()

                next_state_center = (constant_mat
                                     + action_mat_times_starting_point
                                     + dist * action_mat_times_direction
                                     - action_mat @ lin_action)

                next_state_generator = noise_mat @ noise_generator

                constraint = sets.Zonotope.zonotope_containment_constraints(
                    next_state_center,
                    next_state_generator,
                    safe_state_center,
                    safe_state_generator
                )

                constraints += constraint
                parameters += [safe_state_center, safe_state_generator, constant_mat, action_mat,
                               action_mat_times_starting_point, action_mat_times_direction]

            problem = cp.Problem(objective, constraints)
            self.implicit_zonotope_distance_layer = CvxpyLayer(problem, parameters=parameters, variables=[dist])

        starting_point = self.boundary_projection_safeguard.safeguard(action)

        action_dist = torch.linalg.vector_norm(starting_point - action, dim=1, ord=2, keepdim=True)
        safe = action_dist < 1e-8

        directions = (starting_point - action) / action_dist
        parameters = [starting_point, directions] + self.constraint_parameters()
        parameters += [torch.einsum('bij,bj->bi', parameters[-1], starting_point)]
        parameters += [torch.einsum('bij,bj->bi', parameters[-2], directions)]

        dist = self.implicit_zonotope_distance_layer(*parameters, solver_args=self.solver_args)[0].unsqueeze(1)

        safe_dist = torch.where(safe, torch.ones_like(dist), dist / 2)
        safe_center = torch.where(safe, starting_point, starting_point + directions * safe_dist)
        feasible_dist = torch.where(safe, torch.ones_like(safe_dist),
                                    self.axis_aligned_unit_box_dist(safe_center, -directions))

        return safe_center, safe_dist, feasible_dist
