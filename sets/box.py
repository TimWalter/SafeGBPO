import torch
import matplotlib.pyplot as plt

from jaxtyping import Float, Bool
from torch import Tensor

from sets.interface.set import Set
from sets.interface.compact_convex_set import CompactConvexSet
from sets.polytope import Polytope
from sets.zonotope import Zonotope


class Box(CompactConvexSet):
    """
    A box is a zonotope with the additional requirements that the generators are orthogonal and that there
    are exactly 'dim' generators.
    """
    center: Float[Tensor, "batch_dim dim"]
    generator: Float[Tensor, "batch_dim dim dim"]

    cache: dict

    def __init__(self,
                 center: Float[Tensor, "batch_dim dim"],
                 generator: Float[Tensor, "batch_dim dim dim"]):
        """
        Initialise the box.

        Args:
            center: The centre of the box.
            generator: The half edges of the box.
        """
        gram_matrix = torch.bmm(generator.transpose(-2, -1), generator)
        diag_mask = torch.eye(gram_matrix.size(2), dtype=torch.bool).unsqueeze(0).repeat(gram_matrix.size(0), 1, 1)

        assert torch.allclose(gram_matrix[~diag_mask], torch.zeros_like(gram_matrix[~diag_mask]), atol=1e-6), \
            "Box needs orthogonal generators"

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
        return self.as_zonotope.as_polytope

    @property
    def as_zonotope(self) -> Zonotope:
        """
        Convert the box to a zonotope.

        Returns:
            Zonotope.
        """
        if "zonotope" not in self.cache:
            self.cache["zonotope"] = Zonotope(self.center, self.generator)
        return self.cache["zonotope"]

    def __getitem__(self, idx: int | Float[Tensor, "*"]):
        """
        Access a specific box in the batch.

        Args:
            idx: The index of the box.

        Returns:
            The box at the given index.
        """
        if isinstance(idx, int):
            return Box(self.center[idx:idx + 1], self.generator[idx:idx + 1])
        elif isinstance(idx, Tensor):
            return Box(self.center[idx], self.generator[idx])
        else:
            raise TypeError(f"Invalid argument type {type(idx)}")

    @classmethod
    def random(cls, batch_dim: int, dim: int):
        """
        Return a random box.

        Args:
            batch_dim: The batch dimension of the box.
            dim: The dimension of the box.

        Returns:
            The random box.
        """
        center = torch.rand(batch_dim, dim) * 2 - 1

        lengths = torch.rand(batch_dim, 1, dim)
        directions = torch.linalg.qr(torch.randn(batch_dim, dim, dim))[0]
        generator = directions * lengths

        return cls(center, generator)

    def contains(self, other: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"] | Set) \
            -> Bool[Tensor, "num_samples {self.batch_dim}"] | Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set or point is contained in the box.

        Args:
            other: The set to check for containment.

        Returns:
            True if the point or set is contained in the box, False otherwise.
        """
        import sets

        if isinstance(other, Tensor):
            return (torch.abs(self.to_local_coordinate(other)) <= 1.0 + 1e-6).all(dim=-1)
        elif not isinstance(other, sets.CompactSet):
            return torch.zeros(self.batch_dim, dtype=torch.bool, device=self.device)
        elif isinstance(other, sets.CompactConvexSet):
            return self.as_polytope.contains(other)
        else:
            raise NotImplementedError(f"Containment check not implemented for {type(other)}")

    def intersects(self, other: Set) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set intersects with the box.

        Args:
            other: The convex set to check for intersection.

        Returns:
            True if other intersects with the box, False otherwise.
        """
        import sets

        if isinstance(other, sets.AxisAlignedBox) or \
                isinstance(other, sets.Hyperplane) or \
                isinstance(other, sets.Capsule):
            return self.as_zonotope.intersects(other)
        if isinstance(other, sets.Box):
            return self.as_zonotope.intersects(other)
        if isinstance(other, sets.Ball):
            closest_point = self.to_global_coordinates(
                torch.clamp(self.to_local_coordinate(other.center.unsqueeze(0)), min=-1.0, max=1.0)).squeeze(0)
            return torch.linalg.norm(closest_point - other.center, dim=-1) <= other.radius
        elif isinstance(other, sets.Zonotope) or isinstance(other, sets.Polytope):
            return other.intersects(self)
        else:
            raise NotImplementedError(f"Intersection check not implemented for {type(other)}")

    def draw(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """
        Draw the box.

        Args:
            ax: The matplotlib axis to draw the box on.
            kwargs: Additional keyword arguments for drawing.
        """
        return self.as_polytope.draw(ax, **kwargs)

    def sample(self, num_samples: int) -> Float[Tensor, "{num_samples} {self.batch_dim} {self.dim}"]:
        """
        Sample uniformly from the box.

        Args:
            num_samples: The number of samples to draw from the box.

        Returns:
            A tensor of sampled points from the box.
        """
        local_samples = torch.rand(num_samples, self.batch_dim, self.dim, device=self.device) * 2 - 1
        return self.to_global_coordinates(local_samples)

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

    def to_local_coordinate(self, points: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"]) \
            -> Float[Tensor, "num_samples {self.batch_dim} {self.dim}"]:
        """
        Convert a set of points from global to local coordinate. (Using the generators as basis)

        Args:
            points: Points in global coordinates.

        Returns:
            Points in local coordinates
        """
        return (torch.einsum('sbd,bdg->sbg', points - self.center.unsqueeze(0), self.generator)
                / torch.sum(self.generator ** 2, dim=1, keepdim=True))

    def to_global_coordinates(self, points: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"]) \
            -> Float[Tensor, "num_samples {self.batch_dim} {self.dim}"]:
        """
        Convert a set of points from local to global coordinates. Inverse to to_local_coordinate.

        Args:
            points: Points in local coordinates.

        Returns:
            Points in global coordinates
        """
        return self.center.unsqueeze(0) + torch.einsum('sbx,bdx->sbd', points, self.generator)

    def to_vertices(self) -> Float[Tensor, "num_vertices {self.batch_dim} {self.dim}"]:
        """
        Compute the vertices of the box

        Returns:
            Vertices.
        """
        return self.as_polytope.to_vertices()
