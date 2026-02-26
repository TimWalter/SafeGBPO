from typing import Self

import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from torch import Tensor

from src.sets.interface.convex_set import ConvexSet


class Ball(ConvexSet):
    """
    A ball is a set of points that are all within a certain distance from a
    given point (the center). It is defined by a center and a radius.

    Attributes:
        center: The center of the ball.
        radius: The radius of the ball.
    """
    center: Tensor
    radius: Tensor

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 center: Float[Tensor, "batch_dim dim"],
                 radius: Float[Tensor, "batch_dim"]):
        """
        Initialize the ball with the given center and radius.

        Args:
            center: The center of the ball.
            radius: The radius of the ball.
        """
        super().__init__(*center.shape)
        self.center = center
        self.radius = radius

    def __iter__(self):
        return iter((self.center, self.radius))

    def __getitem__(self, item) -> Self:
        return Ball(self.center[item].unsqueeze(0), self.radius[item].unsqueeze(0))

    def draw(self, ax=None, **kwargs):
        """
        Draw the ball. (Assumes 2D and batch_dim=1)

        Args:
            ax: The matplotlib axis to draw on.
            kwargs: Additional keyword arguments to pass to the plotting function.
        """
        if ax is None:
            ax = plt.gca()

        x = self.center[0, 0].item()
        y = self.center[0, 1].item()
        circle = Circle((x, y), self.radius[0].item(), fill=False, **kwargs)

        ax.add_patch(circle)

    @jaxtyped(typechecker=beartype)
    def sample(self) -> Float[Tensor, "{self.batch_dim} {self.dim}"]:
        """
        Sample points uniformly from the ball.

        Returns:
            A tensor of sampled points from the ball.
        """
        samples = torch.rand(self.batch_dim, self.dim) * 2 - 1
        samples = samples / torch.norm(samples, dim=1, keepdim=True)
        # Sphere to ball
        samples *= torch.pow(torch.rand(self.batch_dim, 1),
                             1 / self.dim)
        samples = self.center + self.radius * samples
        return samples

    @jaxtyped(typechecker=beartype)
    def contains(self, other: Float[Tensor, "{self.batch_dim} {self.dim}"] | ConvexSet) \
            -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set is contained in the ball.

        Args:
            other: The convex set to check for containment.

        Returns:
            True if the point is contained in the ball, False otherwise.
        """
        import src.sets as sets

        if isinstance(other, Tensor):
            return torch.norm(other - self.center, dim=1) <= self.radius
        elif isinstance(other, Ball):
            return torch.norm(other.center - self.center,
                              dim=1) + other.radius <= self.radius
        elif isinstance(other, sets.Box):
            farthest_corner = other.farthest_box_corner(self.center)

            return torch.norm(farthest_corner - self.center, dim=1) <= self.radius
        elif isinstance(other, sets.Capsule):
            return self.contains(Ball(other.start, other.radius)) & self.contains(
                Ball(other.end, other.radius))
        elif isinstance(other, sets.Zonotope):
            # Over approximation of the zonotope by a box to lower computational complexity
            return self.contains(other.box())
        else:
            raise NotImplementedError(
                f"Containment check not implemented for {type(other)}")

    @jaxtyped(typechecker=beartype)
    def intersects(self, other: ConvexSet) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set intersects with the ball.

        Args:
            other: The convex set to check for intersection.

        Returns:
            True if other intersects with the ball, False otherwise.
        """
        import src.sets as sets

        if isinstance(other, sets.Ball):
            center_distance = torch.norm(other.center - self.center, dim=1)
            return center_distance <= self.radius + other.radius
        elif isinstance(other, sets.Box):
            # calculates the closest point on each edge of the box to the center of the
            # ball to find the overall closest point by combining these edge moves. see
            # closest_point_on_line_segment
            v = -2 * other.generator
            u = self.center.unsqueeze(2) - other.center.unsqueeze(2) - other.generator
            closest_point = other.center + torch.sum(
                other.generator + torch.clamp(
                    torch.sum(u * v, dim=1, keepdim=True) / torch.sum(v * v, dim=1,
                                                                      keepdim=True),
                    0, 1
                ) * v, dim=2)
            return torch.norm(closest_point - self.center, dim=1) <= self.radius
        elif isinstance(other, sets.Capsule):
            closest_point = other.closest_point_on_line_segment(self.center)
            return self.intersects(Ball(closest_point, other.radius))
        elif isinstance(other, sets.Zonotope):
            # Overapproximations by boxes, once axis aligned and once with an edge
            # orthogonal to the vector between the centers
            direction = self.center - other.center
            distance = torch.norm(direction, dim=1)
            direction /= distance.unsqueeze(1)
            support = (direction.unsqueeze(2) * other.generator).sum(dim=1).abs().sum(
                dim=1)
            return self.intersects(other.box()) & (distance <= support + self.radius)
        
        elif isinstance(other, sets.Polytope):
            raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")

    def setup_constraints(self):
        raise NotImplementedError("Linear constraint matrices not defined for Ball.")