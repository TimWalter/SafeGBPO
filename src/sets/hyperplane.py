from typing import Self

import torch
from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool

from src.sets.interface.convex_set import ConvexSet

class Hyperplane(ConvexSet):
    """
    Hyperplane (affine subspace) defined by:

        H = { x | A x = b }

    Batched representation.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 A: Float[Tensor, "batch_dim m dim"],
                 b: Float[Tensor, "batch_dim m 1"]):
        """
        Args:
            A: Constraint matrix
            b: Constraint vector
        """
        batch_dim, m, dim = A.shape
        super().__init__(batch_dim, dim)

        self.A = A.detach()
        self.b = b.detach()

        # Precompute projection operator: P y + c
        with torch.no_grad():
            Apinv = torch.linalg.pinv(A)
            I = torch.eye(dim, device=A.device).expand(batch_dim, dim, dim)
            self.P = I - torch.bmm(Apinv, A)          
            self.c = torch.bmm(Apinv, b)

        self.P = self.P.detach()
        self.c = self.c.detach()

    def __iter__(self):
        return iter((self.A, self.b))

    def __getitem__(self, item) -> Self:
        if isinstance(item, int):
            return Hyperplane(self.A[item].unsqueeze(0),
                              self.b[item].unsqueeze(0))
        elif isinstance(item, Tensor):
            return Hyperplane(self.A[item], self.b[item])
        else:
            raise TypeError(f"Invalid argument type {type(item)}")

    def project(self, y: Tensor) -> Tensor:
        """
        Project onto the affine subspace {x | A x = b}.

        Args:
            y: (batch, dim)

        Returns:
            Projected tensor satisfying A x = b
        """
        return torch.bmm(self.P, y) + self.c

    def contains(self, other) -> Bool[Tensor, "{self.batch_dim}"]:
        raise NotImplementedError("Containment for Hyperplane not implemented.")

    def intersects(self, other: ConvexSet) -> Bool[Tensor, "{self.batch_dim}"]:
        raise NotImplementedError("Intersection for Hyperplane not implemented.")
    
    def draw(self, ax=None, **kwargs):
        """
        Draw the hyperplane set.

        Args:
            ax: The matplotlib axis to draw the convex set on.
            kwargs: Additional keyword arguments for drawing.
        """
        raise NotImplementedError("Draw for Hyperplane not implemented.")
    
    def sample(self) -> Float[Tensor, "{self.batch_dim} {self.dim}"]:
        """
        Sample points uniformly from the convex set.

        Returns:
            A tensor of sampled points from the convex set.
        """
        raise NotImplementedError("Sample for Hyperplane not implemented.")
    
    def setup_constraints(self):
        return super().setup_constraints()