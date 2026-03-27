import torch
import matplotlib.pyplot as plt

from jaxtyping import Float, Bool
from torch import Tensor

from sets.interface.set import Set
from sets.interface.compact_convex_set import CompactConvexSet
from sets.polytope import Polytope
from sets.zonotope import Zonotope
from sets.box import Box


class AxisAlignedBox(CompactConvexSet):
    """
    An axis-aligned box is a box with the additional requirement that the generator matrix is diagonal.
    """
    center: Float[Tensor, "batch_dim dim"]
    generator: Float[Tensor, "batch_dim dim dim"]

    cache: dict

    def __init__(self,
                 center: Float[Tensor, "batch_dim dim"],
                 generator: Float[Tensor, "batch_dim dim dim"]):
        diagonal = torch.diag_embed(torch.diagonal(generator, dim1=-2, dim2=-1))

        assert diagonal.shape == generator.shape and torch.equal(generator, diagonal), \
            "Axis-aligned box needs square, diagonal generators"

        super().__init__(*center.shape, device=center.device)
        self.center = center
        self.generator = generator

        self.cache = {}

    @property
    def as_box(self) -> Box:
        """
        Convert the box to a zonotope.

        Returns:
            Zonotope.
        """
        if "box" not in self.cache:
            self.cache["box"] = Box(self.center, self.generator)
        return self.cache["box"]

    @property
    def as_zonotope(self) -> Zonotope:
        """
        Convert the box to a zonotope.

        Returns:
            Zonotope.
        """
        return self.as_box.as_zonotope

    @property
    def as_polytope(self) -> Polytope:
        """
        Convert generator representation to halfspace representation.

        Returns:
            Normal of the halfspace and the anchor of the halfspace.
        """
        return self.as_box.as_zonotope.as_polytope

    def __getitem__(self, idx: int | Float[Tensor, "*"]):
        """
        Access a specific box in the batch.

        Args:
            idx: The index of the box.

        Returns:
            The box at the given index.
        """
        if isinstance(idx, int):
            return AxisAlignedBox(self.center[idx:idx + 1], self.generator[idx:idx + 1])
        elif isinstance(idx, Tensor):
            return AxisAlignedBox(self.center[idx], self.generator[idx])
        else:
            raise TypeError(f"Invalid argument type {type(idx)}")

    @classmethod
    def random(cls, batch_dim: int, dim: int):
        """
        Return a random axis-aligned box.

        Args:
            batch_dim: The batch dimension of the box.
            dim: The dimension of the box.

        Returns:
            The random axis-aligned box.
        """
        center = torch.rand(batch_dim, dim) * 2 - 1
        generator = torch.diag_embed(torch.rand(batch_dim, dim))

        return cls(center, generator)

    def contains(self, other: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"] | Set) \
            -> Bool[Tensor, "num_samples {self.batch_dim}"] | Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set or point is contained in the axis-aligned box.

        Args:
            other: The set to check for containment.

        Returns:
            True if the point or set is contained in the axis-aligned box, False otherwise.
        """
        import sets

        if isinstance(other, Tensor):
            return ((other - self.center).abs() <= torch.diagonal(self.generator, dim1=-2, dim2=-1)).all(dim=-1)
        elif not isinstance(other, sets.CompactSet):
            return torch.zeros(self.batch_dim, dtype=torch.bool, device=self.device)
        elif isinstance(other, sets.CompactConvexSet):
            return self.as_polytope.contains(other)
        else:
            raise NotImplementedError(f"Containment check not implemented for {type(other)}")

    def intersects(self, other: Set) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set intersects with the axis-aligned box.

        Args:
            other: The set to check for intersection.

        Returns:
            True if other intersects with the axis-aligned box, False otherwise.
        """
        import sets
        if isinstance(other, sets.AxisAlignedBox):
            r1 = torch.diagonal(self.generator, dim1=-2, dim2=-1)
            r2 = torch.diagonal(other.generator, dim1=-2, dim2=-1)
            center_dist = torch.abs(self.center - other.center)
            return torch.all(center_dist <= (r1 + r2), dim=-1)
        elif isinstance(other, sets.Box) or isinstance(other, sets.Hyperplane):
            return self.as_zonotope.intersects(other)
        elif isinstance(other, sets.Ball):
            lower, upper = self.bounds()
            closest_point = torch.clamp(other.center, min=lower, max=upper)
            return torch.linalg.norm(closest_point - other.center, dim=-1) <= other.radius
        elif isinstance(other, sets.Capsule):
            return self.as_zonotope.intersects(other)
        elif isinstance(other, sets.Zonotope) or isinstance(other, sets.Polytope):
            return other.intersects(self)
        else:
            raise NotImplementedError(f"Intersection check not implemented for {type(other)}")

    def draw(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """
        Draw the axis-aligned box.

        Args:
            ax: The matplotlib axis to draw the box on.
            kwargs: Additional keyword arguments for drawing.
        """
        return self.as_polytope.draw(ax, **kwargs)

    def sample(self, num_samples: int) -> Float[Tensor, "{num_samples} {self.batch_dim} {self.dim}"]:
        """
        Sample uniformly from the axis-aligned box.

        Args:
            num_samples: The number of samples to draw from the axis-aligned box.

        Returns:
            A tensor of sampled points from the axis-aligned box.
        """
        return self.center.unsqueeze(0) + ((torch.rand(num_samples, self.batch_dim, self.dim, device=self.device) * 2 - 1) *
                torch.diagonal(self.generator, dim1=-2, dim2=-1))

    def bounds(self) \
            -> tuple[Float[Tensor, "{self.batch_dim} {self.dim}"], Float[Tensor, "{self.batch_dim} {self.dim}"]]:
        """
        Return the bounds of the axis-aligned box.

        Returns:
            Lower and upper bounds of the axis-aligned box.
        """
        extend = torch.diagonal(self.generator, dim1=-2, dim2=-1)
        return self.center - extend, self.center + extend

    def support(self, direction: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"]) \
            -> Float[Tensor, "num_samples {self.batch_dim}"]:
        """
        Compute the support of the box in the given direction.

        Args:
            direction: The direction in which to compute the support, expected to be of unit length.

        Returns:
            Support of the box in the given direction.
        """
        return self.as_zonotope.support(direction)

    def to_vertices(self) -> Float[Tensor, "num_vertices {self.batch_dim} {self.dim}"]:
        """
        Compute the vertices of the axis-aligned box.

        Returns:
            Vertices.
        """
        return self.as_polytope.to_vertices()