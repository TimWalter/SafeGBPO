from abc import ABC
from pathlib import Path

import torch

import src.sets as sets
from envs.interfaces.safe_action_env import SafeActionEnv
from envs.interfaces.safe_state_env import SafeStateEnv


class RCIEnv(SafeActionEnv, SafeStateEnv, ABC):
    def __init__(self, num_envs: int, path: Path):
        """
        Base class for simulators that ensure safety via Robust Control Invariance (RCI) sets using zonotopes.

        Args:
            num_envs: Number of environments in the vector.
            path: Path to the zonotope representation of the sets.
        """
        self.ctrl = sets.Zonotope(
            torch.load(Path(f"{path}ctrl_center.pt"), weights_only=True).to(torch.get_default_device()).expand(
                num_envs, -1),
            torch.load(Path(f"{path}ctrl_generators.pt"), weights_only=True).to(torch.get_default_device()).expand(
                num_envs, -1, -1)
        )

        self.rci = sets.Zonotope(
            torch.load(Path(f"{path}rci_center.pt"), weights_only=True).to(torch.get_default_device()).expand(
                num_envs, -1),
            torch.load(Path(f"{path}rci_generators.pt"), weights_only=True).to(torch.get_default_device()).expand(
                num_envs, -1, -1)
        )

        SafeActionEnv.__init__(self, self.ctrl.generator.shape[2])
        SafeStateEnv.__init__(self, self.rci.generator.shape[2])


    def safe_action_set(self) -> sets.Zonotope:
        """
        Get the safe action set for the current state.

        Returns:
            The safe action set.
        """
        return self.ctrl


    def safe_state_set(self) -> sets.Zonotope:
        """
        Get the safe state set for the current state.

        Returns:
            The safe state set.
        """

        return self.rci
