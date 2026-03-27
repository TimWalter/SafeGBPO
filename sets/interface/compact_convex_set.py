from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import Tensor

from sets.interface.compact_set import CompactSet

class CompactConvexSet(CompactSet, ABC):
    """
    Interface for compact, convex sets.
    """

    def bounds(self) \
            -> tuple[Float[Tensor, "{self.batch_dim} {self.dim}"], Float[Tensor, "{self.batch_dim} {self.dim}"]]:
        """
        Return the bounds of an over-approximative axis-aligned box.

        Returns:
            Lower and upper bounds of the axis-aligned box.

        """
        lower = []
        upper = []

        for i in range(self.dim):
            e_i = torch.zeros(self.batch_dim, self.dim, device=self.device)
            e_i[:, i] = 1.0

            x_max = self.support(e_i.unsqueeze(0)).squeeze(0)
            upper.append(x_max)

            x_min = self.support(-e_i.unsqueeze(0)).squeeze(0)
            lower.append(-x_min)

        return torch.stack(lower, dim=1), torch.stack(upper, dim=1)

    @abstractmethod
    def support(self, direction: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"]) \
            -> Float[Tensor, "num_samples {self.batch_dim}"]:
        """
        Compute the support of the set in the given direction.

        Args:
            direction: The direction in which to compute the support, expected to be of unit length.

        Returns:
            Support of the set in the given direction.
        """
        pass

