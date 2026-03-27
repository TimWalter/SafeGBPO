import warnings

import torch
import cvxpy as cp
import matplotlib.pyplot as plt

from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float, Bool
from matplotlib.patches import Polygon
from torch import Tensor, combinations

from sets.interface.set import Set
from sets.interface.compact_convex_set import CompactConvexSet
from sets.utils import SLACK_EPS

class Polytope(CompactConvexSet):
    """
    Polytope is the intersection of half-spaces.
    P_i = { x | A_i x <= b_i }

    Attributes:
        normal: The normals of the hyperplanes.
        anchor: The distance of the normals to the origin
    """

    normal: Float[Tensor, "batch_dim num_planes dim"]
    anchor: Float[Tensor, "batch_dim num_planes"]

    cache: dict

    def __init__(self,
                 normal: Float[Tensor, "batch_dim num_planes dim"],
                 anchor: Float[Tensor, "batch_dim num_planes"]):
        super().__init__(batch_dim=normal.shape[0], dim=normal.shape[2], device=normal.device)
        normal_lengths = torch.linalg.norm(normal, dim=2, keepdim=True)
        if not torch.allclose(normal_lengths, torch.ones_like(normal_lengths)):
            warnings.warn("Expect normals to be of unit length. Normalising them.")
            self.normal = normal / normal_lengths
            self.anchor = anchor / normal_lengths.squeeze(-1)
        else:
            self.normal = normal
            self.anchor = anchor
        self.cache = {}

    def __getitem__(self, idx: int | Float[Tensor, "*"]):
        """
        Access a specific polytope in the batch.

        Args:
            idx: The index of the polytope.

        Returns:
            The polytope at the given index.
        """
        if isinstance(idx, int):
            return Polytope(self.normal[idx:idx + 1], self.anchor[idx:idx + 1])
        elif isinstance(idx, Tensor):
            return Polytope(self.normal[idx], self.anchor[idx])
        else:
            raise TypeError(f"Invalid argument type {type(idx)}")

    @classmethod
    def random(cls, batch_dim: int, dim: int):
        """
        Return a random polytope. Constructs a simplex, adds random constraints, then rotates, and translates.

        Args:
            batch_dim: The batch dimension of the polytope.
            dim: The dimension of the polytope.

        Returns:
            The random polytope.
        """
        num_extra = torch.randint(dim // 2, 2 * dim, (1,)).item()
        num_planes = (dim + 1) + num_extra

        simplex_normal = torch.cat([torch.eye(dim), -torch.ones(1, dim) / torch.sqrt(torch.tensor(dim))], dim=0)
        extra_normal = torch.randn(num_extra, dim)
        extra_normal = extra_normal / torch.linalg.norm(extra_normal, dim=1, keepdim=True)

        normal = torch.cat([simplex_normal, extra_normal], dim=0).unsqueeze(0).repeat(batch_dim, 1, 1)
        anchor = torch.ones(batch_dim, num_planes)

        normal = torch.bmm(normal, torch.linalg.svd(torch.randn(batch_dim, dim, dim))[0])
        anchor = anchor + torch.bmm(normal, torch.randn(batch_dim, dim, 1)).squeeze(-1)

        return cls(normal, anchor)

    def contains(self, other: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"] | Set) \
            -> Bool[Tensor, "num_samples {self.batch_dim}"] | Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set or point is contained in the polytope.

        Args:
            other: The set to check for containment.

        Returns:
            True if the point or set is contained in the polytope, False otherwise.
        """
        import sets

        if isinstance(other, Tensor):
            return (torch.einsum("bcd,sbd->sbc", self.normal, other) <= self.anchor.unsqueeze(1)).all(dim=2)
        elif not isinstance(other, sets.CompactSet):
            return torch.zeros(self.batch_dim, dtype=torch.bool, device=self.device)
        elif isinstance(other, sets.Ball):
            return (torch.sum(self.normal * other.center.unsqueeze(1), dim=2) <= self.anchor - other.radius).all(dim=1)
        elif isinstance(other, sets.Capsule):
            return (self.contains(sets.Ball(other.start, other.radius)) &
                    self.contains(sets.Ball(other.end, other.radius)))
        elif isinstance(other, sets.CompactConvexSet):
            directions = self.normal.transpose(0, 1)
            return torch.all(other.support(directions) <= self.anchor.transpose(0, 1), dim=0)
        else:
            raise NotImplementedError(f"Containment check not implemented for {type(other)}")

    def intersects(self, other: Set) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set intersects with the polytope.

        Args:
            other: The set to check for intersection.

        Returns:
            True if other intersects with the polytope, False otherwise.
        """
        import sets

        if isinstance(other, sets.AxisAlignedBox):
            slack = self.intersection_axis_aligned_box_layer(self.normal, self.anchor, *other.bounds())[0]
            return slack[:, 0] <= SLACK_EPS
        if isinstance(other, sets.Ball):
            return (torch.sum(self.normal * other.center.unsqueeze(1), dim=2) <= self.anchor + other.radius).all(dim=1)
        if isinstance(other, sets.Box):
            return self.intersects(other.as_zonotope)
        elif isinstance(other, sets.Capsule):
            polytope_point, t = self.intersection_capsule_layer(self.normal, self.anchor, other.start, other.end)
            capsule_point = other.start + t * (other.end - other.start)
            return torch.linalg.norm(polytope_point - capsule_point, dim=1) <= other.radius
        elif isinstance(other, sets.Hyperplane):
            return ((-self.support(-other.normal.unsqueeze(0)).squeeze(0) <= other.anchor) &
                    (self.support(other.normal.unsqueeze(0)).squeeze(0) >= other.anchor))
        elif isinstance(other, Polytope):
            slack = self.get_intersection_polytope_layer(other)(self.normal, self.anchor, other.normal, other.anchor)[0]
            return slack[:, 0] <= SLACK_EPS
        elif isinstance(other, sets.Zonotope):
            return self.intersects(other.as_polytope)
        else:
            raise NotImplementedError(f"Intersection check not implemented for {type(other)}")

    def draw(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """
        Draw the polytope.

        Args:
            ax: The matplotlib axis to draw the compact set on.
            kwargs: Additional keyword arguments for drawing.
        """
        ax = super().draw()

        vertices = self.to_vertices()[:, 0].cpu()
        center = vertices.mean(dim=0)
        angles = torch.atan2(
            vertices[:, 1] - center[1],
            vertices[:, 0] - center[0]
        )
        vertices = vertices[torch.argsort(angles, dim=0)]

        polygon = Polygon(vertices, fill=False, **kwargs)
        ax.add_patch(polygon)
        ax.relim()
        ax.autoscale_view()
        return ax

    def support(self, direction: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"]) \
            -> Float[Tensor, "num_samples {self.batch_dim}"]:
        """
        Compute the support of the polytope in the given direction.

        Args:
            direction: The direction in which to compute the support, expected to be of unit length.

        Returns:
            Support of the polytope in the given direction.
        """
        lengths = torch.linalg.norm(direction, dim=-1, keepdim=True)
        if not torch.allclose(lengths, torch.ones_like(lengths)):
            warnings.warn("Expect directions to be of unit length. Normalising them.")
            direction = direction / lengths

        num_samples = direction.shape[0]
        expanded_normal = self.normal.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(-1, *self.normal.shape[1:])
        expanded_anchor = self.anchor.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1, *self.anchor.shape[1:])
        flat_direction = direction.reshape(-1, self.dim)

        x_opt = self.support_layer(expanded_normal, expanded_anchor, flat_direction, solver_args={"solve_method": "Clarabel"})[0]
        return torch.sum(flat_direction * x_opt, dim=-1).reshape(num_samples, self.batch_dim)

    @property
    def intersection_axis_aligned_box_layer(self) -> CvxpyLayer:
        if "intersection_axis_aligned_box_layer" not in self.cache:
            normal = cp.Parameter(self.normal.shape[1:])
            anchor = cp.Parameter(self.anchor.shape[1:])
            lower = cp.Parameter(self.dim)
            upper = cp.Parameter(self.dim)

            x = cp.Variable(self.dim)
            slack = cp.Variable(1, nonneg=True)

            constraints = [
                normal @ x <= anchor + slack,
                x >= lower - slack,
                x <= upper + slack
            ]

            objective = cp.Minimize(slack)
            problem = cp.Problem(objective, constraints)
            self.cache["intersection_axis_aligned_box_layer"] = CvxpyLayer(
                problem,
                parameters=[normal, anchor, lower, upper],
                variables=[slack]
            )
        return self.cache["intersection_axis_aligned_box_layer"]

    @property
    def intersection_capsule_layer(self) -> CvxpyLayer:
        if "intersection_capsule_layer" not in self.cache:
            normal = cp.Parameter(self.normal.shape[1:])
            anchor = cp.Parameter(self.anchor.shape[1:])
            start = cp.Parameter(self.dim)
            end = cp.Parameter(self.dim)
            parameters = [normal, anchor, start, end]

            polytope_point = cp.Variable(self.dim)
            t = cp.Variable(1)

            constraints = [
                normal @ polytope_point <= anchor,
                t >= 0,
                t <= 1
            ]

            capsule_point = start + t * (end - start)
            objective = cp.Minimize(cp.sum_squares(polytope_point - capsule_point))

            problem = cp.Problem(objective, constraints)
            self.cache["intersection_capsule_layer"] = CvxpyLayer(problem, parameters=parameters,
                                                                  variables=[polytope_point, t])

        return self.cache["intersection_capsule_layer"]

    def get_intersection_polytope_layer(self, other) -> CvxpyLayer:
        if ("intersection_polytope_layer" not in self.cache or
                self.cache["intersection_polytope_layer"].param_order[2].shape[0] != other.normal.shape[1]):
            normal_1 = cp.Parameter(self.normal.shape[1:])
            anchor_1 = cp.Parameter(self.anchor.shape[1:])
            normal_2 = cp.Parameter(other.normal.shape[1:])
            anchor_2 = cp.Parameter(other.anchor.shape[1:])
            parameters = [normal_1, anchor_1, normal_2, anchor_2]

            x = cp.Variable(self.dim)
            slack = cp.Variable(1, nonneg=True)

            constraints = [
                normal_1 @ x <= anchor_1 + slack,
                normal_2 @ x <= anchor_2 + slack
            ]

            objective = cp.Minimize(slack)

            problem = cp.Problem(objective, constraints)
            self.cache["intersection_polytope_layer"] = CvxpyLayer(problem, parameters=parameters, variables=[slack])
        return self.cache["intersection_polytope_layer"]


    @property
    def support_layer(self):
        if "support_layer" not in self.cache:
            normal = cp.Parameter(self.normal.shape[1:])
            anchor = cp.Parameter(self.anchor.shape[1:])
            direction = cp.Parameter(self.dim)
            parameters = [normal, anchor, direction]

            x = cp.Variable(self.dim)
            constraints = [normal @ x <= anchor]

            objective = cp.Maximize(direction @ x)
            problem = cp.Problem(objective, constraints)

            self.cache["support_layer"] = CvxpyLayer(problem, parameters=parameters, variables=[x])

        return self.cache["support_layer"]

    def to_vertices(self) -> Float[Tensor, "num_vertices {self.batch_dim} {self.dim}"]:
        """
        Convert the polytope in halfspace-representation into vertex representation.

        Returns:
            Vertices.
        """
        indices = combinations(torch.arange(0, self.anchor.shape[1], device=self.device), self.dim)

        normal_comb = self.normal[:, indices, :]
        anchor_comb = self.anchor[:, indices].unsqueeze(-1)

        solution = torch.linalg.lstsq(normal_comb, anchor_comb)
        candidates = solution.solution.squeeze(-1)

        is_invertible = torch.linalg.svdvals(normal_comb)[..., -1] > 1e-6
        is_finite = torch.isfinite(candidates).all(dim=-1)
        is_inside = (torch.einsum('bcd,bnd->bcn', candidates, self.normal) <= self.anchor.unsqueeze(1) + 1e-6).all(
            dim=-1)
        is_vertex = is_invertible & is_finite & is_inside

        max_v = int(is_vertex.sum(dim=1).max().item())
        if max_v == 0:
            return torch.full((self.batch_dim, 1, self.dim), float('nan'), device=self.device)

        vertices = torch.zeros((max_v, self.batch_dim, self.dim), device=self.device)
        for batch_idx in range(self.batch_dim):
            valid_candidates = torch.unique(candidates[batch_idx][is_vertex[batch_idx]].round(decimals=6), dim=0)

            num_found = valid_candidates.shape[0]
            vertices[:num_found, batch_idx] = valid_candidates
            if num_found < max_v:
                vertices[num_found:, batch_idx] = valid_candidates[-1]

        return vertices
