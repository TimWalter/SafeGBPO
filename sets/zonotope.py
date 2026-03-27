import warnings

import torch
import cvxpy as cp
import matplotlib.pyplot as plt

from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float, Bool
from torch import Tensor

from sets.interface.set import Set
from sets.interface.compact_convex_set import CompactConvexSet
from sets.polytope import Polytope
from sets.utils import SLACK_EPS

class Zonotope(CompactConvexSet):
    """
    A zonotope is a centrally symmetric polytope.
    A zonotope is a convex set defined as the sum of a point and a linear
    combination of vectors. It is defined by a centre and a set of generators as
    Z = {x | x = centre + sum_i lambda_i * generator_i, lambda_i in [-1, 1]}.

    Attributes:
        center: The centre of the zonotope.
        generator: The generators of the zonotope.
    """
    center: Float[Tensor, "batch_dim dim"]
    generator: Float[Tensor, "batch_dim dim num_generators"]

    cache: dict

    def __init__(self,
                 center: Float[Tensor, "batch_dim dim"],
                 generator: Float[Tensor, "batch_dim dim num_generators"]):
        """
        Initialise the zonotope with the given centre and generators.

        Args:
            center: The centre of the zonotope.
            generator: The generators of the zonotope.
        """
        super().__init__(*center.shape, device=center.device)
        self.center = center
        self.generator = generator

        self.cache = {}

    @property
    def as_polytope(self) -> Polytope:
        """
        Convert generator representation to halfspace representation.

        Returns:
            Normal of the halfspace and the anchor of the halfspace.
        """
        if "polytope" not in self.cache:
            combos = torch.combinations(torch.arange(self.generator.shape[2], device=self.device), self.dim - 1)

            subsets = self.generator[:, :, combos].permute(0, 2, 3, 1)
            normals = torch.linalg.svd(subsets)[-1][:, :, -1, :]
            normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)

            projections = torch.bmm(normals, self.generator).abs()
            b = projections.sum(dim=-1)

            center_offset = torch.einsum('bni,bi->bn', normals, self.center)

            normal = torch.cat([normals, -normals], dim=1)
            anchor = torch.cat([b + center_offset, b - center_offset], dim=1)

            self.cache["polytope"] = Polytope(normal, anchor)
        return self.cache["polytope"]

    def __getitem__(self, idx: int | Float[Tensor, "*"]):
        """
        Access a specific zonotope in the batch.

        Args:
            idx: The index of the zonotope.

        Returns:
            The zonotope at the given index.
        """
        if isinstance(idx, int):
            return Zonotope(self.center[idx:idx + 1], self.generator[idx:idx + 1])
        elif isinstance(idx, Tensor):
            return Zonotope(self.center[idx], self.generator[idx])
        else:
            raise TypeError(f"Invalid argument type {type(idx)}")

    @classmethod
    def random(cls, batch_dim: int, dim: int):
        """
        Return a random zonotope.

        Args:
            batch_dim: The batch dimension of the zonotope.
            dim: The dimension of the zonotope.

        Returns:
            The random zonotope.
        """
        center = torch.rand(batch_dim, dim) * 2 -1

        num_generators = torch.randint(dim * 2, 3 * dim, (1,)).item()
        lengths = torch.rand(batch_dim, 1, num_generators)
        directions = torch.randn(batch_dim, dim, num_generators)
        directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True)
        generator = directions * lengths

        return cls(center, generator)

    def contains(self, other: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"] | Set) \
            -> Bool[Tensor, "num_samples {self.batch_dim}"] | Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set or point is contained in the zonotope.

        Args:
            other: The set to check for containment.

        Returns:
            True if the point or set is contained in the zonotope, False otherwise.
        """
        import sets

        if isinstance(other, Tensor):
            num_samples = other.shape[0]
            weights = self.point_containment_layer(self.center.repeat_interleave(num_samples, dim=0),
                                                   self.generator.repeat_interleave(num_samples, dim=0),
                                                   other.reshape(-1, self.dim),
                                                   solver_args={"solve_method": "Clarabel"})[0]
            return (weights.reshape(num_samples, self.batch_dim, -1).abs() <= 1).all(dim=2)
        elif not isinstance(other, sets.CompactSet):
            return torch.zeros(self.batch_dim, dtype=torch.bool, device=self.device)
        elif isinstance(other, sets.AxisAlignedBox):
            return self.contains(other.as_zonotope)
        elif isinstance(other, sets.Ball) or isinstance(other, sets.Capsule) or isinstance(other, sets.Polytope):
            return self.as_polytope.contains(other)
        elif isinstance(other, sets.Box):
            return self.contains(other.as_zonotope)
        elif isinstance(other, Zonotope):
            weights, mapping = self.get_zonotope_containment_layer(other)(self.center, self.generator,
                                                                          other.center, other.generator)
            return 1 >= torch.cat([weights.unsqueeze(-1), mapping], dim=-1).norm(torch.inf, dim=(1, 2))
        else:
            raise NotImplementedError(f"Containment check not implemented for {type(other)}")

    def intersects(self, other: Set) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set intersects with the zonotope.

        Args:
            other: The set to check for intersection.

        Returns:
            True if other intersects with the zonotope, False otherwise.
        """
        import sets

        if isinstance(other, sets.AxisAlignedBox) or isinstance(other, sets.Box):
            weights, other_weights = self.intersection_box_layer(self.center, self.generator, other.center, other.generator)
            point = self.center + (self.generator * weights.unsqueeze(1)).sum(dim=2)
            other_point = other.center + (other.generator * other_weights.unsqueeze(1)).sum(dim=2)
            return (point - other_point).norm(dim=-1) <= SLACK_EPS
        elif isinstance(other, sets.Zonotope):
            weights, other_weights = self.get_intersection_zonotope_layer(other)(self.center, self.generator, other.center, other.generator)
            point = self.center + (self.generator * weights.unsqueeze(1)).sum(dim=2)
            other_point = other.center + (other.generator * other_weights.unsqueeze(1)).sum(dim=2)
            return (point - other_point).norm(dim=-1) <= SLACK_EPS
        elif isinstance(other, Polytope):
            return other.intersects(self)
        elif isinstance(other, sets.Ball):
            weights = self.intersection_ball_layer(self.center, self.generator, other.center)[0]
            closest_point = self.center + torch.sum(self.generator * weights.unsqueeze(1), dim=2)
            return torch.linalg.norm(closest_point - other.center, dim=1) <= other.radius
        elif isinstance(other, sets.Capsule):
            weights, t = self.intersection_capsule_layer(self.center, self.generator, other.start, other.end)
            zonotope_point = self.center + torch.sum(self.generator * weights.unsqueeze(1), dim=2)
            capsule_point = other.start + t * (other.end - other.start)
            return torch.linalg.norm(capsule_point - zonotope_point, dim=1) <= other.radius
        elif isinstance(other, sets.Hyperplane):
            return ((-self.support(-other.normal.unsqueeze(0)).squeeze(0) <= other.anchor) &
                    (self.support(other.normal.unsqueeze(0)).squeeze(0) >= other.anchor))
        else:
            raise NotImplementedError(f"Intersection check not implemented for {type(other)}")

    def draw(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """
        Draw the zonotope.

        Args:
            ax: The matplotlib axis to draw the zonotope on.
            kwargs: Additional keyword arguments for drawing.
        """
        return self.as_polytope.draw(ax, **kwargs)

    def sample(self, num_samples: int) -> Float[Tensor, "{num_samples} {self.batch_dim} {self.dim}"]:
        """
        Sample from the zonotope, however, not uniform but much cheaper.

        Args:
            num_samples: The number of samples to draw from the zonotope.

        Returns:
            A tensor of sampled points from the zonotope.
        """
        weights = 2 * torch.rand(num_samples, self.batch_dim, self.generator.shape[2], device=self.device) - 1
        exp_center = self.center.unsqueeze(0).expand(num_samples, -1, -1)

        return exp_center + torch.einsum('sbg,bdg->sbd', weights, self.generator)

    def support(self, direction: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"]) \
            -> Float[Tensor, "num_samples {self.batch_dim}"]:
        """
        Compute the support of the zonotope in the given direction.

        Args:
            direction: The direction in which to compute the support, expected to be of unit length.

        Returns:
            Support of the zonotope in the given direction.
        """
        lengths = torch.linalg.norm(direction, dim=-1, keepdim=True)
        if not torch.allclose(lengths, torch.ones_like(lengths)):
            warnings.warn("Expect directions to be of unit length. Normalising them.")
            direction = direction / lengths

        return torch.sum(direction * self.center.unsqueeze(0), dim=-1) + \
            torch.einsum("sbd,bdg->sbg", direction, self.generator).norm(p=1, dim=-1)

    @property
    def point_containment_layer(self) -> CvxpyLayer:
        if "point_containment_layer" not in self.cache:
            center = cp.Parameter(self.dim)
            generator = cp.Parameter(self.generator.shape[1:])
            other = cp.Parameter(self.dim)
            parameters = [center, generator, other]

            weights = cp.Variable(self.generator.shape[2])

            constraints = [
                other == center + generator @ weights
            ]

            objective = cp.Minimize(cp.max(cp.abs(weights)))
            problem = cp.Problem(objective, constraints)

            self.cache["point_containment_layer"] = CvxpyLayer(problem, parameters=parameters, variables=[weights])
        return self.cache["point_containment_layer"]

    def get_zonotope_containment_layer(self, other) -> CvxpyLayer:
        if ("get_zonotope_containment_layer" not in self.cache or
                self.cache["get_zonotope_containment_layer"].param_order[3].shape[1] != other.generator.shape[2]):
            center = cp.Parameter(self.dim)
            generator = cp.Parameter(self.generator.shape[1:])
            other_center = cp.Parameter(self.dim)
            other_generator = cp.Parameter((self.dim, other.generator.shape[2]))
            parameters = [center, generator, other_center, other_generator]

            weights = cp.Variable(self.generator.shape[2])
            mapping = cp.Variable((self.generator.shape[2], other.generator.shape[2]))
            variables = [weights, mapping]

            constraints = [
                other_generator == generator @ mapping,
                center - other_center == generator @ weights
            ]

            objective = cp.Minimize(cp.norm(cp.hstack([mapping, cp.reshape(weights, (-1, 1), "C")]), "inf"))
            problem = cp.Problem(objective, constraints)

            self.cache["zonotope_containment_layer"] = CvxpyLayer(problem, parameters=parameters, variables=variables)
        return self.cache["zonotope_containment_layer"]


    @property
    def intersection_box_layer(self) -> CvxpyLayer:
        if "intersection_box_layer" not in self.cache:
            center = cp.Parameter(self.dim)
            generator = cp.Parameter(self.generator.shape[1:])
            other_center = cp.Parameter(self.dim)
            other_generator = cp.Parameter((self.dim, self.dim))
            parameters = [center, generator, other_center, other_generator]

            weights = cp.Variable(self.generator.shape[2])
            other_weights = cp.Variable(self.dim)

            constraints = [
                cp.norm(weights, "inf") <= 1,
                cp.norm(other_weights, "inf") <= 1
            ]

            point = center + generator @ weights
            other_point = other_center + other_generator @ other_weights

            objective = cp.Minimize(cp.norm(point - other_point, 2))

            problem = cp.Problem(objective, constraints)
            self.cache["intersection_box_layer"] = CvxpyLayer(problem, parameters=parameters, variables=[weights, other_weights])
        return self.cache["intersection_box_layer"]


    def get_intersection_zonotope_layer(self, other) -> CvxpyLayer:
        if ("intersection_zonotope_layer" not in self.cache or
            self.cache["intersection_zonotope_layer"].param_order[3].shape[1] != other.generator.shape[2]):
            center = cp.Parameter(self.dim)
            generator = cp.Parameter(self.generator.shape[1:])
            other_center = cp.Parameter(self.dim)
            other_generator = cp.Parameter(other.generator.shape[1:])
            parameters = [center, generator, other_center, other_generator]

            weights = cp.Variable(self.generator.shape[2])
            other_weights = cp.Variable(other.generator.shape[2])

            constraints = [
                cp.norm(weights, "inf") <= 1,
                cp.norm(other_weights, "inf") <= 1
            ]

            point = center + generator @ weights
            other_point = other_center + other_generator @ other_weights
            objective = cp.Minimize(cp.norm(point-other_point, 2))

            problem = cp.Problem(objective, constraints)
            self.cache["intersection_zonotope_layer"] = CvxpyLayer(problem, parameters=parameters, variables=[weights, other_weights])
        return self.cache["intersection_zonotope_layer"]

    @property
    def intersection_ball_layer(self) -> CvxpyLayer:
        if "intersection_ball_layer" not in self.cache:
            center = cp.Parameter(self.dim)
            generator = cp.Parameter(self.generator.shape[1:])
            ball_center = cp.Parameter(self.dim)
            parameters = [center, generator, ball_center]

            weights = cp.Variable(self.generator.shape[2])

            constraints = [
                cp.norm(weights, "inf") <= 1
            ]

            objective = cp.Minimize(cp.norm(center + generator @ weights - ball_center, 2))

            problem = cp.Problem(objective, constraints)
            self.cache["intersection_ball_layer"] = CvxpyLayer(problem, parameters=parameters, variables=[weights])
        return self.cache["intersection_ball_layer"]

    @property
    def intersection_capsule_layer(self) -> CvxpyLayer:
        if "intersection_capsule_layer" not in self.cache:
            center = cp.Parameter(self.dim)
            generator = cp.Parameter(self.generator.shape[1:])
            start = cp.Parameter(self.dim)
            end = cp.Parameter(self.dim)
            parameters = [center, generator, start, end]

            weights = cp.Variable(self.generator.shape[2])
            t = cp.Variable(1)

            constraints = [
                cp.norm(weights, "inf") <= 1,
                t >= 0,
                t <= 1
            ]

            capsule_point = start + t * (end - start)
            objective = cp.Minimize(cp.norm(center + generator @ weights - capsule_point, 2))

            problem = cp.Problem(objective, constraints)
            self.cache["intersection_capsule_layer"] = CvxpyLayer(problem, parameters=parameters,
                                                                  variables=[weights, t])
        return self.cache["intersection_capsule_layer"]

    def to_vertices(self) -> Float[Tensor, "num_vertices {self.batch_dim} {self.dim}"]:
        """
        Compute the vertices of the zonotope

        Returns:
            Vertices.
        """
        return self.as_polytope.to_vertices()