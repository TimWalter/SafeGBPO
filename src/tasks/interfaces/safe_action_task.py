from abc import ABC, abstractmethod
from typing import Literal

from beartype import beartype
from jaxtyping import jaxtyped

import src.sets as sets


class SafeActionTask(ABC):
    """
    Base class for tasks that can provided a (state-dependent) safe action set.
    """

    @staticmethod
    def constrains() -> tuple:
        return ("action-constrained",)

    def __init__(
            self,
            device: Literal["cpu", "cuda"],
            num_action_gens: int
    ):
        """Base class for kinematic environments.

        Args:
            device: Device to deliver the safe set on.
            num_action_gens: Number of generators in the action set.
        """
        self.device = device
        self.num_action_gens = num_action_gens

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def safe_action_set(self) -> sets.Zonotope:
        """
        Get the safe state set for the current state.

        Returns:
            A convex set representing the safe action set.

        Note:
            Cache the result if it is expensive to compute
        """
        pass
