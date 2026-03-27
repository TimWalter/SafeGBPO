from abc import ABC, abstractmethod

import torch

from torch import Tensor
from jaxtyping import Float, Bool, Integer


class Set(ABC):
    """
    Interface for torch-native, batched set classes.
    """

    def __init__(self, batch_dim: int, dim: int, device: torch.device):
        """
        Initialize the set.

        Args:
            batch_dim: The batch dimension of the set.
            dim: The dimension of the set.
            device: The device of the set.
        """
        self.batch_dim = batch_dim
        self.dim = dim
        self.device = device

    def __getitem__(self, idx: int | Integer[Tensor, "*"]):
        """
        Access a specific set in the batch.

        Args:
            idx: The index of the set.

        Returns:
            The set at the given index.
        """
        pass

    def __iter__(self):
        """
        Iterate over the batch dimension.

        Yields:
            The set of the given batch.
        """
        for i in range(self.batch_dim):
            yield self[i]

    def to(self, device: torch.device):
        """
        Transfer set to the device.

        Args:
            device: The device of the set.

        Returns:
            Transferred set.
        """
        self.device = device
        for attr_name, value in self.__dict__.items():
            if torch.is_tensor(value) or hasattr(value, "to"):
                try:
                    setattr(self, attr_name, value.to(device))
                except AttributeError:
                    pass

        return self

    @classmethod
    def random(cls, batch_dim: int, dim: int):
        """
        Return a random set.

        Args:
            batch_dim: The batch dimension of the set.
            dim: The dimension of the set.

        Returns:
            The random set.
        """
        pass

    @abstractmethod
    def contains(self, other: Float[Tensor, "num_samples {self.batch_dim} {self.dim}"]) \
            -> Bool[Tensor, "num_samples {self.batch_dim}"] | Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set is contained in the set.

        Args:
            other: The set to check for containment.

        Returns:
            True if other is contained in the set, False otherwise.
        """
        pass

    @abstractmethod
    def intersects(self, other) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another set intersects with the set.

        Args:
            other: The set to check for intersection.

        Returns:
            True if other intersects with the set, False otherwise.
        """
        pass
