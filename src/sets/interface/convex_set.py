from abc import ABC, abstractmethod
from typing import Self

from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool
from torch import Tensor, device

import src.sets

class ConvexSet(ABC):
    """
    Interface for convex set classes. (Batched)
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self, _device: device, batch_dim: int, dim: int):
        """
        Initialize the convex set.

        Args:
            batch_dim: The batch dimension of the convex set.
            dim: The dimension of the convex set.
        """
        self.device = _device
        self.batch_dim = batch_dim
        self.dim = dim

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __getitem__(self, item) -> Self:
        pass

    @abstractmethod
    def draw(self, ax=None, **kwargs):
        """
        Draw the convex set.

        Args:
            ax: The matplotlib axis to draw the convex set on.
            kwargs: Additional keyword arguments for drawing.
        """
        pass

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def sample(self) -> Float[Tensor, "{self.batch_dim} {self.dim}"]:
        """
        Sample points uniformly from the convex set.

        Returns:
            A tensor of sampled points from the convex set.
        """

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def contains(self, other: Float[Tensor, "{self.batch_dim} {self.dim}"]) \
            -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set is contained in the convex set.

        Args:
            other: The convex set to check for containment.

        Returns:
            True if other is contained in the convex set, False otherwise.
        """
        pass

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def intersects(self, other) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set intersects with the convex set.

        Args:
            other: The convex set to check for intersection.

        Returns:
            True if other intersects with the convex set, False otherwise.
        """
        pass
