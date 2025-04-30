from abc import ABC
from typing import Literal

import torch

import src.sets as sets
from tasks.interfaces.safe_action_task import SafeActionTask
from tasks.interfaces.safe_state_task import SafeStateTask


class RCITask(SafeActionTask, SafeStateTask, ABC):
    """
    Base class for tasks that ensure safety via Robust Control Invariance (RCI) sets
    using zonotopes.
    """

    @staticmethod
    def constrains() -> tuple:
        return "state-constrained", "action-constrained"

    def __init__(
            self,
            device: Literal["cpu", "cuda"],
            num_envs: int,
            rci_size: int,
            path: str
    ):
        """
        Args:
            device: Device to deliver the safe set on.
            num_envs: Number of environments in the vector.
            rci_size: Indicator for the number of generators in the RCI set.
            path: Path to the zonotope representation of the sets.
        """
        self.num_envs = num_envs
        self.ctrl = sets.Zonotope(
            torch.load(path + "ctrl_center.pt", weights_only=True).expand(num_envs, -1),
            torch.load(path + f"ctrl_generators_{rci_size}.pt",
                       weights_only=True).expand(num_envs, -1, -1)
        )

        self.rci = sets.Zonotope(
            torch.load(path + "rci_center.pt", weights_only=True).expand(num_envs, -1),
            torch.load(path + f"rci_generators_{rci_size}.pt",
                       weights_only=True).expand(num_envs, -1, -1)
        )

        SafeActionTask.__init__(self, device, self.ctrl.generator.shape[2])

        SafeStateTask.__init__(self, device, self.rci.generator.shape[2])

        self.ctrl.center = self.ctrl.center.to(self.device)
        self.ctrl.generator = self.ctrl.generator.to(self.device)
        self.rci.center = self.rci.center.to(self.device)
        self.rci.generator = self.rci.generator.to(self.device)
        self.ctrl.device = self.device
        self.rci.device = self.device

    def safe_action_set(self) -> sets.Zonotope:
        """
        Get the safe state set for the current state.

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
