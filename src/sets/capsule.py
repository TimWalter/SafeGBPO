from typing import Self

import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool
from matplotlib import pyplot as plt
from matplotlib.patches import Arc
from torch import Tensor

from src.sets.interface.convex_set import ConvexSet


class Capsule(ConvexSet):
    """
    A capsule is defined as the Minkowski sum of the convex hull of two points and a ball.
    It is described by two endpoints and a radius.

    Attributes:
        start: The first endpoint of the capsule.
        end: The second endpoint of the capsule.
        radius: The radius of the capsule.
    """
    start: Tensor
    end: Tensor
    radius: Tensor

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 start: Float[Tensor, "batch_dim dim"],
                 end: Float[Tensor, "batch_dim dim"],
                 radius: Float[Tensor, "batch_dim"]):
        """
        Initialize the capsule with the given endpoints and radius.

        Args:
            start: The first endpoint of the capsule.
            end: The second endpoint of the capsule.
            radius: The radius of the capsule.
        """
        super().__init__(*start.shape)
        self.start = start
        self.end = end
        self.radius = radius

    def __iter__(self):
        return iter((self.start, self.end, self.radius))

    def __getitem__(self, item) -> Self:
        return Capsule(self.start[item].unsqueeze(0), self.end[item].unsqueeze(0),
                       self.radius[item].unsqueeze(0))

    def draw(self, ax=None, **kwargs):
        """
        Draw the capsule. (Assumes 2D and batch_dim=1)

        Args:
            ax: The matplotlib axis to draw on.
            kwargs: Additional keyword arguments to pass to the plotting function.
        """
        if ax is None:
            ax = plt.gca()

        start = self.start[0].cpu()
        end = self.end[0].cpu()
        radius = self.radius.item()

        direction = (end - start) / torch.norm(end - start)
        orthogonal_direction = torch.tensor([-direction[1], direction[0]])

        s1 = start + radius * orthogonal_direction
        s2 = start - radius * orthogonal_direction
        e1 = end + radius * orthogonal_direction
        e2 = end - radius * orthogonal_direction

        x_line1 = torch.stack([s1[0], e1[0]])
        y_line1 = torch.stack([s1[1], e1[1]])
        x_line2 = torch.stack([s2[0], e2[0]])
        y_line2 = torch.stack([s2[1], e2[1]])

        ax.plot(x_line1.numpy(), y_line1.numpy(), **kwargs)
        ax.plot(x_line2.numpy(), y_line2.numpy(), **kwargs)

        angle = torch.atan2(direction[1], direction[0]).item() * 180 / torch.pi

        arc1 = Arc((start[0].item(), start[1].item()), 2 * radius, 2 * radius,
                   angle=angle, theta1=90, theta2=270, **kwargs)
        arc2 = Arc((end[0].item(), end[1].item()), 2 * radius, 2 * radius,
                   angle=angle, theta1=-90, theta2=90, **kwargs)
        ax.add_patch(arc1)
        ax.add_patch(arc2)

    @jaxtyped(typechecker=beartype)
    def sample(self) -> Float[Tensor, "{self.batch_dim} {self.dim}"]:
        """
        Sample points uniformly from the capsule.

        Returns:
            A tensor of sampled points from the capsule.
        """
        from src.sets.ball import Ball

        t = torch.rand(self.start.shape[0], 1)
        line_samples = t * self.start + (1 - t) * self.end
        return Ball(line_samples, self.radius).sample()

    @jaxtyped(typechecker=beartype)
    def contains(self, other: Float[Tensor, "{self.batch_dim} {self.dim}"] | ConvexSet) \
            -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set is contained in the capsule.

        Args:
            other: The convex set to check for containment.

        Returns:
            True if the point is contained in the capsule, False otherwise.
        """
        import src.sets as sets

        if isinstance(other, Tensor):
            closest_point = self.closest_point_on_line_segment(other)
            return sets.Ball(closest_point, self.radius).contains(other)
        elif isinstance(other, sets.Ball):
            closest_point = self.closest_point_on_line_segment(other.center)
            return sets.Ball(closest_point, self.radius).contains(other)
        elif isinstance(other, sets.Box):
            closest_center = self.closest_point_on_line_segment(other.center)
            farthest_corner = other.farthest_box_corner(closest_center)
            closest_point = self.closest_point_on_line_segment(farthest_corner)
            return torch.norm(farthest_corner - closest_point, dim=1) <= self.radius
        elif isinstance(other, sets.Capsule):
            return self.contains(sets.Ball(other.start, other.radius)) & \
                self.contains(sets.Ball(other.end, other.radius))
        elif isinstance(other, sets.Zonotope):
            # Over approximation of the zonotope by a box to lower computational complexity
            return self.contains(other.box())
        elif isinstance(other, sets.Polytope):
            raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")
        else:
            raise NotImplementedError(
                f"Containment check not implemented for {type(other)}")

    @jaxtyped(typechecker=beartype)
    def intersects(self, other: ConvexSet) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set intersects with the capsule.

        Args:
            other: The convex set to check for intersection.

        Returns:
            True if other intersects with the capsule, False otherwise.
        """
        import sets as sets

        if isinstance(other, sets.Ball):
            return other.intersects(self)
        elif isinstance(other, sets.Box):
            # Non-vectorized version
            closest_box_point = other.center
            for i in range(other.dim):
                edge_addition = \
                self.shortest_line_segment(other.center + other.generator[:, :, i],
                                           other.center - other.generator[:, :, i])[1]
                edge_addition -= other.center
                closest_box_point += edge_addition
            closest_point = self.closest_point_on_line_segment(closest_box_point)
            return torch.norm(closest_box_point - closest_point, dim=1) <= self.radius
        elif isinstance(other, sets.Capsule):
            pt1, pt2 = self.shortest_line_segment(other.start, other.end)
            min_distance = torch.norm(pt1 - pt2, dim=1)
            return min_distance <= self.radius + other.radius
        elif isinstance(other, sets.Zonotope):
            # Over approximation of the zonotope by a box to lower computational complexity
            return self.intersects(other.box())
        else:
            raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")

    @jaxtyped(typechecker=beartype)
    def closest_point_on_line_segment(self, point: Float[Tensor, "batch_dim dim"]) \
            -> Float[Tensor, "batch_dim dim"]:
        """
        Find the closest point on a line segment to a given point.

        Args:
            point: The point to find the closest point to.

        Returns:
            The closest point on the line segment to the given point.
        """
        v = self.end - self.start
        u = point - self.start

        return self.start + torch.clamp(
            torch.sum(u * v, dim=1, keepdim=True) / torch.sum(v * v, dim=1,
                                                              keepdim=True),
            0, 1
        ) * v

    @jaxtyped(typechecker=beartype)
    def shortest_line_segment(self,
                              start: Float[Tensor, "batch_dim dim"],
                              end: Float[Tensor, "batch_dim dim"]) \
            -> tuple[
                Float[Tensor, "batch_dim dim"],
                Float[Tensor, "batch_dim dim"],
            ]:
        """
        Calculate the shortest line segments to connect two line segments.
        Calculation is obtained from the analytical solution to the minimisation
        of the squared l2 norm.

        Args:
            start: The start of the second line segment
            end: The end of the second line segment

        Returns:
            Tuple containing the start and end of the shortest connecting line segment.
        """
        l1 = self.end - self.start
        l2 = start - end
        ds = self.start - start

        # Compute coefficients of the quadratic equation
        alpha = torch.sum(l1 ** 2, dim=1)
        beta = torch.sum(l2 ** 2, dim=1)
        gamma = torch.sum(l1 * l2, dim=1)
        delta = torch.sum(l1 * ds, dim=1)
        epsilon = torch.sum(l2 * ds, dim=1)

        # Determinant of the 2x2 matrix
        det = alpha * beta - gamma ** 2

        s1 = torch.zeros_like(alpha)
        s2 = torch.zeros_like(alpha)
        # Check for a degenerate case where the determinant is zero (parallel lines)
        deg = det == 0

        s1[~deg] = (gamma[~deg] * epsilon[~deg] - beta[~deg] * delta[~deg]) / det[
            ~deg]
        s2[~deg] = (gamma[~deg] * delta[~deg] - alpha[~deg] * epsilon[~deg]) / det[
            ~deg]
        # Degenerate cases and clamping cases have boundary solutions. The boundary
        # solution cannot be calculated explicitly due to the uneven curvature in both
        # dimensions. Therefore, we enumerate. (as this problem is always 2D)
        mask = deg | (s1 < 0) | (s1 > 1) | (s2 < 0) | (s2 > 1)
        if mask.any():
            other = Capsule(start, end, self.radius)
            cs1 = other.closest_point_on_line_segment(self.start)
            cs2 = self.closest_point_on_line_segment(start)
            ce1 = other.closest_point_on_line_segment(self.end)
            ce2 = self.closest_point_on_line_segment(end)
            dists = torch.stack([
                torch.norm(start - cs2, dim=1),
                torch.norm(end - ce2, dim=1),
                torch.norm(self.start - cs1, dim=1),
                torch.norm(self.end - ce1, dim=1),
            ], dim=1)
            # Check for axes-aligned lines to avoid division by 0
            idx1 = torch.argmax((l1 != 0).type(torch.int), dim=1)
            idx2 = torch.argmax((l2 != 0).type(torch.int), dim=1)

            s1[mask & (dists.argmin(dim=1) == 0)] = ((cs2 - self.start) / l1)[
                mask & (dists.argmin(dim=1) == 0), idx1]
            s2[mask & (dists.argmin(dim=1) == 0)] = 0
            s1[mask & (dists.argmin(dim=1) == 1)] = ((ce2 - self.start) / l1)[
                mask & (dists.argmin(dim=1) == 1), idx1]
            s2[mask & (dists.argmin(dim=1) == 1)] = 1
            s1[mask & (dists.argmin(dim=1) == 2)] = 0
            s2[mask & (dists.argmin(dim=1) == 2)] = ((cs1 - start) / (-l2))[
                mask & (dists.argmin(dim=1) == 2), idx2]
            s1[mask & (dists.argmin(dim=1) == 3)] = 1
            s2[mask & (dists.argmin(dim=1) == 3)] = ((ce1 - start) / (-l2))[
                mask & (dists.argmin(dim=1) == 3), idx2]

        return self.start + s1.unsqueeze(1) * (self.end - self.start), \
               start + s2.unsqueeze(1) * (end - start)

    def setup_constraints(self):
        raise NotImplementedError("Linear constraint matrices not defined for Ball.")