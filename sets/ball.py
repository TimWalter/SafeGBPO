import warnings

import torch
import matplotlib.pyplot as plt

from jaxtyping import Float, Bool
from matplotlib.patches import Circle
from torch import Tensor

from sets.interface.set import Set
from sets.interface.compact_convex_set import CompactConvexSet
from sets.utils import closest_point_on_line_segment


class Ball(CompactConvexSet):
    """
    A ball is a set of points that are all within a certain distance from the centre.

    Attributes:
        center: The centre of the ball.
        radius: The radius of the ball.
    """
    center: Float[Tensor, "batch_dim dim"]
    radius: Float[Tensor, "batch_dim"]

    def __init__(self,
                 center: Float[Tensor, "batch_dim dim"],
                 radius: Float[Tensor, "batch_dim"]):
        super().__init__(*center.shape, device=center.device)
        self.center = center
        self.radius = radius.abs()

    def __getitem__(self, idx: int | Float[Tensor, "*"]):
        """
        Access a specific ball in the batch.

        Args:
            idx: The index of the ball.

        Returns:
            The ball at the given index.
        """
        if isinstance(idx, int):
            return Ball(self.center[idx:idx + 1], self.radius[idx:idx + 1])
        elif isinstance(idx, Tensor):
            return Ball(self.center[idx], self.radius[idx])
        else:
            raise TypeError(f"Invalid argument type {type(idx)}")

    @classmethod
    def random(cls, batch_dim: int, dim: int):
        """
        Return a random ball.

        Args:
            batch_dim: The batch dimension of the ball.
            dim: The dimension of the ball.

        Returns:
            The random ball.
        """
        center = torch.rand(batch_dim, dim) * 2 - 1
        radius = torch.rand(batch_dim)

        return cls(center, radius)

    def contains(self, other: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"] | Set) \
            -> Bool[Tensor, "num_samples {self.batch_dim}"] | Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set or point is contained in the ball.

        Args:
            other: The set to check for containment.

        Returns:
            True if the point or set is contained in the ball, False otherwise.
        """
        import sets

        if isinstance(other, Tensor):
            return torch.norm(other - self.center.unsqueeze(0), dim=-1) <= self.radius
        elif not isinstance(other, sets.CompactSet):
            return torch.zeros(self.batch_dim, dtype=torch.bool, device=self.device)
        elif isinstance(other, Ball):
            return torch.norm(other.center - self.center, dim=1) + other.radius <= self.radius
        elif isinstance(other, sets.Capsule):
            return self.contains(Ball(other.start, other.radius)) & self.contains(Ball(other.end, other.radius))
        elif isinstance(other, sets.AxisAlignedBox) or \
                isinstance(other, sets.Box) or \
                isinstance(other, sets.Zonotope) or \
                isinstance(other, sets.Polytope):
            return self.contains(other.to_vertices()).all(dim=0)
        else:
            raise NotImplementedError(f"Containment check not implemented for {type(other)}")

    def intersects(self, other: Set) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set intersects with the ball.

        Args:
            other: The set to check for intersection.

        Returns:
            True if other intersects with the ball, False otherwise.
        """
        import sets

        if isinstance(other, sets.Ball):
            center_distance = torch.norm(other.center - self.center, dim=1)
            return center_distance <= self.radius + other.radius
        elif isinstance(other, sets.Capsule):
            closest_point = closest_point_on_line_segment(other.start, other.end, self.center.unsqueeze(0)).squeeze(0)
            return self.intersects(Ball(closest_point, other.radius))
        elif isinstance(other, sets.Hyperplane):
            return other.normal @ self.center.T - other.anchor <= self.radius
        elif isinstance(other, sets.AxisAlignedBox) or \
                isinstance(other, sets.Box) or \
                isinstance(other, sets.Zonotope) or \
                isinstance(other, sets.Polytope):
            return other.intersects(self)
        else:
            raise NotImplementedError(f"Intersection check not implemented for {type(other)}")

    def draw(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """
        Draw the ball.

        Args:
            ax: The matplotlib axis to draw on.
            kwargs: Additional keyword arguments to pass to the plotting function.
        """
        ax = super().draw(ax, **kwargs)

        x = self.center[0, 0].item()
        y = self.center[0, 1].item()
        circle = Circle((x, y), self.radius[0].item(), fill=False, **kwargs)
        ax.add_patch(circle)

        ax.relim()
        ax.autoscale_view()

        return ax

    def sample(self, num_samples: int) -> Float[Tensor, "{num_samples} {self.batch_dim} {self.dim}"]:
        """
        Sample uniformly from the ball.

        Args:
            num_samples: The number of samples to draw from the ball.

        Returns:
            A tensor of sampled points from the ball.
        """
        direction = torch.randn(num_samples, self.batch_dim, self.dim, device=self.device)
        direction /= torch.norm(direction, dim=-1, keepdim=True)
        radius = self.radius.unsqueeze(0) * torch.rand(num_samples, self.batch_dim, device=self.device) ** (1 / self.dim)
        return self.center.unsqueeze(0) + radius.unsqueeze(-1) * direction

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
        return torch.sum(direction * self.center.unsqueeze(0), dim=-1) + self.radius
