from abc import ABC, abstractmethod

import torch

from src.algorithms.components.coupled_buffer import CoupledBuffer
from src.algorithms.components.policy import Policy
from tasks.envs.interfaces.torch_vector_env import TorchVectorEnv


class ActorCriticAlgorithm(ABC):
    """
    Abstract class for actor critic algorithms
    """
    env: TorchVectorEnv
    device: torch.device
    policy: Policy
    policy_optim: torch.optim.Optimizer
    policy_learning_rate_schedule: str
    value_function_optim: torch.optim.Optimizer
    vf_learning_rate_schedule: str
    buffer: CoupledBuffer
    samples_per_episode: int

    def learn(self, samples: int, callback):
        num_learn_episodes = samples // self.samples_per_episode

        policy_lr_update = 0
        if self.policy_learning_rate_schedule == "linear":
            policy_lr_update = (1e-5 - self.policy_optim.param_groups[0][
                "lr"]) / num_learn_episodes
        vf_lr_update = 0
        if self.vf_learning_rate_schedule == "linear":
            vf_lr_update = (1e-5 - self.value_function_optim.param_groups[0][
                "lr"]) / num_learn_episodes

        for eps in range(num_learn_episodes):
            policy_loss, value_loss = self._learn_episode(eps)

            for param_group in self.policy_optim.param_groups:
                param_group["lr"] += policy_lr_update
            for param_group in self.value_function_optim.param_groups:
                param_group["lr"] += vf_lr_update

            with torch.no_grad():
                callback.on_learning_episode(eps, policy_loss, value_loss, num_learn_episodes)

    @abstractmethod
    def _learn_episode(self, eps: int) -> tuple[float, float]:
        pass
