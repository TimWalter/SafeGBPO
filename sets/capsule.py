import warnings

import torch
import matplotlib.pyplot as plt

from jaxtyping import Float, Bool
from matplotlib.patches import Circle
from torch import Tensor

from sets.interface.set import Set
from sets.interface.compact_convex_set import CompactConvexSet
from sets.utils import closest_point_on_line_segment, shortest_line_segment


class Capsule(CompactConvexSet):
    """
    A capsule is defined as the Minkowski sum of the convex hull, a line segment, and a ball.

    Attributes:
        start: The first endpoint of the line segment.
        end: The second endpoint of the line segment.
        radius: The radius of the ball.
    """
    start: Float[Tensor, "batch_dim dim"]
    end: Float[Tensor, "batch_dim dim"]
    radius: Float[Tensor, "batch_dim"]

    def __init__(self,
                 start: Float[Tensor, "batch_dim dim"],
                 end: Float[Tensor, "batch_dim dim"],
                 radius: Float[Tensor, "batch_dim"]):
        super().__init__(*start.shape, device=start.device)
        self.start = start
        self.end = end
        self.radius = torch.abs(radius)

    def __getitem__(self, idx: int | Float[Tensor, "*"]):
        """
        Access a specific capsule in the batch.

        Args:
            idx: The index of the capsule.

        Returns:
            The capsule at the given index.
        """
        if isinstance(idx, int):
            return Capsule(self.start[idx:idx + 1], self.end[idx:idx + 1], self.radius[idx:idx + 1])
        elif isinstance(idx, Tensor):
            return Capsule(self.start[idx], self.end[idx], self.radius[idx])
        else:
            raise TypeError(f"Invalid argument type {type(idx)}")

    @classmethod
    def random(cls, batch_dim: int, dim: int):
        """
        Return a random capsule.

        Args:
            batch_dim: The batch dimension.
            dim: The dimension.

        Returns:
            The random capsule.
        """
        start = torch.rand(batch_dim, dim) * 2 - 1
        end = torch.rand(batch_dim, dim) * 2 - 1
        radius = torch.rand(batch_dim)

        return cls(start, end, radius)

    def contains(self, other: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"] | Set) \
            -> Bool[Tensor, "num_samples {self.batch_dim}"] | Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set or point is contained in the capsule.

        Args:
            other: The set to check for containment.

        Returns:
            True if the point or set is contained in the capsule, False otherwise.
        """
        import sets

        if isinstance(other, Tensor):
            num_samples = other.shape[0]
            closest_point = closest_point_on_line_segment(self.start, self.end, other).view(-1, self.dim)
            radius = self.radius.repeat_interleave(num_samples)
            return sets.Ball(closest_point, radius).contains(other.view(1, -1, self.dim)).view(num_samples, self.batch_dim)
        elif not isinstance(other, sets.CompactSet):
            return torch.zeros(self.batch_dim, dtype=torch.bool, device=self.device)
        elif isinstance(other, sets.Ball):
            closest_point = closest_point_on_line_segment(self.start, self.end, other.center.unsqueeze(0)).squeeze(0)
            return sets.Ball(closest_point, self.radius).contains(other)
        elif isinstance(other, Capsule):
            return (self.contains(sets.Ball(other.start, other.radius)) &
                    self.contains(sets.Ball(other.end, other.radius)))
        elif isinstance(other, sets.AxisAlignedBox) or \
                isinstance(other, sets.Box) or \
                isinstance(other, sets.Zonotope) or \
                isinstance(other, sets.Polytope):
            return self.contains(other.to_vertices()).all(dim=0)
        else:
            raise NotImplementedError(
                f"Containment check not implemented for {type(other)}")

    def intersects(self, other: Set) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set intersects with the capsule.

        Args:
            other: The set to check for intersection.

        Returns:
            True if other intersects with the capsule, False otherwise.
        """
        import sets

        if isinstance(other, sets.Ball):
            return other.intersects(self)
        elif isinstance(other, Capsule):
            start, end = shortest_line_segment(self.start, self.end, other.start, other.end)
            return torch.norm(start - end, dim=1) <= self.radius + other.radius
        elif isinstance(other, sets.Hyperplane):
            d_start = other.normal.T @ self.start - other.anchor
            d_end = other.normal.T @ self.end - other.anchor
            same_side = d_start * d_end > 0
            min_dist = torch.where(same_side, torch.minimum(d_start.abs(), d_end.abs()),
                                   torch.zeros(self.batch_dim, device=self.device))
            return min_dist <= self.radius
        elif isinstance(other, sets.AxisAlignedBox) or \
                isinstance(other, sets.Box) or \
                isinstance(other, sets.Zonotope) or \
                isinstance(other, sets.Polytope):
            return other.intersects(self)
        else:
            raise NotImplementedError(f"Intersection check not implemented for {type(other)}")

    def draw(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """
        Draw the capsule.

        Args:
            ax: The matplotlib axis to draw on.
            kwargs: Additional keyword arguments to pass to the plotting function.
        """
        ax = super().draw(ax, **kwargs)

        start = self.start[0]
        end = self.end[0]
        radius = self.radius[0]

        line = (end - start) / torch.norm(end - start)
        angle_base = torch.atan2(line[1], line[0])
        theta_end = torch.linspace(angle_base - torch.pi / 2, angle_base + torch.pi / 2, 100)
        theta_start = torch.linspace(angle_base + torch.pi / 2, angle_base + 3 * torch.pi / 2, 100)
        x_end = end[0] + radius * torch.cos(theta_end)
        y_end = end[1] + radius * torch.sin(theta_end)
        x_start = start[0] + radius * torch.cos(theta_start)
        y_start = start[1] + radius * torch.sin(theta_start)

        x_hull = torch.cat([x_end, x_start, x_end[:1]])
        y_hull = torch.cat([y_end, y_start, y_end[:1]])
        ax.plot(x_hull.numpy(), y_hull.numpy(), **kwargs)

        ax.relim()
        ax.autoscale_view()

        return ax

    def sample(self, num_samples: int) -> Float[Tensor, "{num_samples} {self.batch_dim} {self.dim}"]:
        """
        Sample uniformly from the capsule.

        Args:
            num_samples: The number of samples to draw.

        Returns:
            A tensor of sampled points from the capsule.
        """
        volume_caps = self.radius ** self.dim * torch.pi ** (self.dim / 2) / torch.lgamma(
            torch.tensor([self.dim / 2 + 1])).exp()
        caps_samples = self._sample_caps(num_samples)

        cylinder_samples = self._sample_cylinder(num_samples)
        line_length = torch.norm(self.end - self.start, dim=-1)
        cylinder_face = self.radius ** (self.dim - 1) * torch.pi ** ((self.dim - 1) / 2) / torch.lgamma(
            torch.tensor([(self.dim - 1) / 2 + 1])).exp()
        volume_cylinder = line_length * cylinder_face

        coin = torch.rand(num_samples, self.batch_dim, 1, device=self.device)
        weighting = volume_cylinder / (volume_cylinder + volume_caps)
        return torch.where((coin < weighting.view(1, self.batch_dim, 1)),
                           cylinder_samples,
                           caps_samples)

    def support(self, direction: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"]) \
            -> Float[Tensor, "num_samples {self.batch_dim}"]:
        """
        Compute the support of the capsule in the given direction.

        Args:
            direction: The direction in which to compute the support, expected to be of unit length.

        Returns:
            Support of the capsule in the given direction.
        """
        lengths = torch.linalg.norm(direction, dim=-1, keepdim=True)
        if not torch.allclose(lengths, torch.ones_like(lengths)):
            warnings.warn("Expect directions to be of unit length. Normalising them.")
            direction = direction / lengths

        start_projection = torch.sum(self.start.unsqueeze(0) * direction, dim=-1)
        end_projection = torch.sum(self.end.unsqueeze(0) * direction, dim=-1)
        return torch.maximum(start_projection, end_projection) + self.radius.unsqueeze(0) * direction.norm(dim=2)

    def _sample_cylinder(self, num_samples: int) -> Float[Tensor, "{num_samples} {self.batch_dim} {self.dim}"]:
        """
        Sample uniformly from the inner cylinder of the capsule.

        Args:
            num_samples: The number of samples to draw.

        Returns:
            A tensor of sampled points from the cylinder.
        """
        from sets.ball import Ball

        center = torch.zeros(num_samples * self.batch_dim, self.dim - 1, device=self.device)
        radius = self.radius.repeat_interleave(num_samples, 0)
        lower_ball_samples = Ball(center, radius).sample(1).reshape(num_samples, self.batch_dim, self.dim - 1)

        line = (self.end - self.start) / torch.norm(self.end - self.start, dim=-1, keepdim=True)
        null_basis = torch.linalg.svd(line.unsqueeze(1))[-1][:, 1:, :]
        ball_samples = torch.einsum('sbi,bid->sbd', lower_ball_samples, null_basis)

        t = torch.rand(num_samples, self.batch_dim, 1, device=self.device)
        anchor = self.start.unsqueeze(0) + (self.end - self.start).unsqueeze(0) * t

        return anchor + ball_samples

    def _sample_caps(self, num_samples: int) -> Float[Tensor, "{num_samples} {self.batch_dim} {self.dim}"]:
        """
        Sample uniformly from the half-circles at the start and end of the capsule.

        Args:
            num_samples: The number of samples to draw.

        Returns:
            A tensor of sampled points from the caps.
        """
        from sets.ball import Ball

        center = torch.zeros(num_samples * self.batch_dim, self.dim, device=self.device)
        radius = self.radius.repeat_interleave(num_samples, 0)
        ball_samples = Ball(center, radius).sample(1).reshape(num_samples, self.batch_dim, self.dim)

        line = (self.end - self.start) / torch.norm(self.end - self.start, dim=-1, keepdim=True)
        proj = (ball_samples * line.unsqueeze(0)).sum(dim=-1, keepdim=True)
        to_start = proj < 0

        anchor = torch.where(to_start, self.start.unsqueeze(0), self.end.unsqueeze(0))
        return anchor + ball_samples
