from abc import ABC, abstractmethod

import torch
import matplotlib.pyplot as plt
from torch import Tensor
from jaxtyping import Float

from sets.interface.set import Set

class CompactSet(Set, ABC):
    """
    Interface for compact set classes. (Batched)
    """

    def draw(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """
        Draw the compact set, assumes a batch_size of 1 and dim = 2

        Args:
            ax: The matplotlib axis to draw the compact set on.
            kwargs: Additional keyword arguments for drawing.
        """
        assert self.batch_dim == 1, "Draw only supports batch_size=1."
        assert self.dim == 2, "Draw only supports dim=2."
        if ax is None:
            ax = plt.gca()
        return ax

    @abstractmethod
    def bounds(self) \
            -> tuple[Float[Tensor, "{self.batch_dim} {self.dim}"], Float[Tensor, "{self.batch_dim} {self.dim}"]]:
        """
        Return the bounds of an over-approximative axis-aligned box.
        
        Returns:
            Lower and upper bounds of the axis-aligned box.
        
        """
        pass

    def sample(self, num_samples: int) -> Float[Tensor, "{num_samples} {self.batch_dim} {self.dim}"]:
        """
        Sample uniformly from the compact set. Rejection sampling.

        Args:
            num_samples: The number of samples to draw from the compact set.

        Returns:
            A tensor of sampled points from the compact set.
        """
        lower, upper = self.bounds()

        sample = lower + (upper - lower) * torch.rand(num_samples, self.batch_dim, self.dim, device=self.device)
        while (mask := ~self.contains(sample)).any():
            sample[mask] = lower + (upper - lower) * torch.rand(mask.sum().item(), self.dim, device=self.device)

        return sample
