import warnings

import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool

from sets.interface.set import Set


@jaxtyped(typechecker=beartype)
class Hyperplane(Set):
    """
    Affine hyperplane defined by: {x | A x = b }

    Attributes:
        normal: The normal of the hyperplane.
        anchor: The distance of the normal to the origin.
    """
    normal: Float[Tensor, "batch_dim dim"]
    anchor: Float[Tensor, "batch_dim"]

    def __init__(self,
                 normal: Float[Tensor, "batch_dim dim"],
                 anchor: Float[Tensor, "batch_dim"]):
        super().__init__(*normal.shape, device=normal.device)
        normal_lengths = torch.linalg.norm(normal, dim=1, keepdim=True)
        if not torch.allclose(normal_lengths, torch.ones_like(normal_lengths)):
            warnings.warn("Expect normals to be of unit length. Normalising them.")
            self.normal = normal / normal_lengths
            self.anchor = anchor / normal_lengths.squeeze(-1)
        else:
            self.normal = normal
            self.anchor = anchor

    def __getitem__(self, idx: int | Float[Tensor, "*"]):
        """
        Access a specific hyperplane in the batch.

        Args:
            idx: The index of the hyperplane.

        Returns:
            The hyperplane at the given index.
        """
        if isinstance(idx, int):
            return Hyperplane(self.normal[idx:idx + 1], self.anchor[idx:idx + 1])
        elif isinstance(idx, Tensor):
            return Hyperplane(self.normal[idx], self.anchor[idx])
        else:
            raise TypeError(f"Invalid argument type {type(idx)}")

    @classmethod
    def random(cls, batch_dim: int, dim: int):
        """
        Return a random hyperplane.

        Args:
            batch_dim: The batch dimension of the hyperplane.
            dim: The dimension of the hyperplane.

        Returns:
            The random hyperplane.
        """
        normal = torch.randn(batch_dim, dim)
        normal /= torch.linalg.norm(normal, dim=1, keepdim=True)
        anchor = torch.rand(batch_dim)

        return cls(normal, anchor)

    def contains(self, other: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"] | Set) \
            -> Bool[Tensor, "num_samples {self.batch_dim}"] | Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another point or set is contained in the hyperplane.

        Args:
            other: The set to check for containment.

        Returns:
            True if other is contained in the hyperplane, False otherwise.
        """
        import sets

        if isinstance(other, Tensor):
            return torch.sum(self.normal.unsqueeze(0) * other, dim=-1) == self.anchor.unsqueeze(0)
        elif isinstance(other, sets.Ball) or \
                isinstance(other, sets.Capsule) or \
                isinstance(other, sets.AxisAlignedBox) or \
                isinstance(other, sets.Box) or \
                isinstance(other, sets.Zonotope) or \
                isinstance(other, sets.Polytope):
            return torch.zeros(self.batch_dim, dtype=torch.bool, device=self.device)
        elif isinstance(other, Hyperplane):
            return (self.normal == other.normal).all(dim=1) & (self.anchor == other.anchor)
        else:
            raise NotImplementedError(f"Containment check not implemented for {type(other)}")

    def intersects(self, other: Set) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set intersects with the hyperplane.

        Args:
            other: The set to check for intersection.

        Returns:
            True if other intersects with the ball, False otherwise.
        """
        import sets

        if isinstance(other, sets.Ball) or \
                isinstance(other, sets.Capsule) or \
                isinstance(other, sets.AxisAlignedBox) or \
                isinstance(other, sets.Box) or \
                isinstance(other, sets.Zonotope) or \
                isinstance(other, sets.Polytope):
            return other.intersects(self)
        elif isinstance(other, Hyperplane):
            return ~((self.normal == other.normal).all(dim=1) & (self.anchor != other.anchor))
        else:
            raise NotImplementedError(f"Containment check not implemented for {type(other)}")
