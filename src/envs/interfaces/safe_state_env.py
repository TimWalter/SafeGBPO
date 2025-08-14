from abc import ABC, abstractmethod

from beartype import beartype
from jaxtyping import jaxtyped

import src.sets as sets

class SafeStateEnv(ABC):
    def __init__(self, num_state_gens: int):
        """
        Base class for envs that provide a safe state set.

        Args:
            num_state_gens: Number of generators in the state set.
        """
        self.num_state_gens = num_state_gens

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def safe_state_set(self) -> sets.Zonotope:
        """
        Get the safe state set for the current state.

        Returns:
            A convex set representing the safe state set.
        """
        pass
