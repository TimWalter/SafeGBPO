from abc import ABC, abstractmethod

from beartype import beartype
from jaxtyping import jaxtyped

import src.sets as sets


class SafeActionEnv(ABC):
    def __init__(self, num_action_gens: int):
        """Base class for envs that provide a safe action set.

        Args:
            num_action_gens: Number of generators in the action set.
        """
        self.num_action_gens = num_action_gens

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def safe_action_set(self) -> sets.Zonotope:
        """
        Get the safe action set for the current state.

        Returns:
            A convex set representing the safe action set.
        """
        pass
