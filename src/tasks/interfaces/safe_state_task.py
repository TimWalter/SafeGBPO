from abc import ABC, abstractmethod
from typing import Literal

from beartype import beartype
from jaxtyping import jaxtyped

import src.sets as sets

class SafeStateTask(ABC):
    """
    Base class for tasks that can provided a (state-dependent) safe state set.
    """

    @staticmethod
    def constrains() -> tuple:
        return ("state-constrained",)

    def __init__(
            self,
            device: Literal["cpu", "cuda"],
            num_state_gens: int
    ):
        """Base class for kinematic environments.

        Args:
            device: Device to deliver the safe set on.
            num_state_gens: Number of generators in the state set.
        """
        self.device = device
        self.num_state_gens = num_state_gens

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def safe_state_set(self) -> sets.Zonotope:
        """
        Get the safe state set for the current state.

        Returns:
            A convex set representing the safe state set.

        Note:
            Cache the result if it is expensive to compute.
        """
        pass
