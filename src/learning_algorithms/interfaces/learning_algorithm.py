from __future__ import annotations
from abc import ABC, abstractmethod

import torch
from beartype import beartype
from jaxtyping import jaxtyped
from tqdm import tqdm
from torch import Tensor

from learning_algorithms.components.value_function import ValueFunction
from src.learning_algorithms.components.policy import Policy
from envs.simulators.interfaces.simulator import Simulator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.logger import Logger

from tqdm import tqdm

class LearningAlgorithm(ABC):
    """
    Abstract class for actor critic learning_algorithms
    """
    env: Simulator
    policy: Policy
    policy_optim: torch.optim.Optimizer
    value_function_optim: torch.optim.Optimizer
    POLICY_LEARNING_RATE_SCHEDULE = "constant"
    VF_LEARNING_RATE_SCHEDULE = "constant"
    interactions_per_episode: int

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 env: Simulator,
                 policy_kwargs: dict,
                 policy_optim_kwargs: dict,
                 vf_kwargs: dict,
                 vf_optim_kwargs: dict,
                 regularisation_coefficient: float,
                 q_function: bool):
        """
        Initialise the learning algorithm.

        Args:
            env: The environment to train on.
            policy_kwargs: The keyword arguments for the policy network.
            policy_optim_kwargs: The keyword arguments for the policy optimizer.
            vf_kwargs: The keyword arguments for the value function network.
            vf_optim_kwargs: The keyword arguments for the value function optimizer.
            regularisation_coefficient: Regularisation coefficient for the regularisation towards safe actions.
            q_function: Whether to use a Q-function.
        """
        self.env = env
        self.policy = Policy(env.obs_dim, env.action_dim, **policy_kwargs)
        self.value_function = ValueFunction(env.obs_dim + (env.action_dim if q_function else 0), **vf_kwargs)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), **policy_optim_kwargs)
        self.value_function_optim = torch.optim.Adam(self.value_function.parameters(), **vf_optim_kwargs)
        self.regularisation_coefficient = regularisation_coefficient

    @jaxtyped(typechecker=beartype)
    def learn(self, interactions: int, logger: Logger):
        """
        Learn the policy and value function using the given number of samples.

        Args:
            interactions: The number of environment interactions to use for learning.
            logger: The logger to use for logging learning progress.
        """

        num_learn_episodes = interactions // self.interactions_per_episode

        policy_lr_update = 0
        if self.POLICY_LEARNING_RATE_SCHEDULE == "linear":
            policy_lr_update = (1e-5 - self.policy_optim.param_groups[0]["lr"]) / num_learn_episodes
        vf_lr_update = 0
        if self.VF_LEARNING_RATE_SCHEDULE == "linear":
            vf_lr_update = (1e-5 - self.value_function_optim.param_groups[0]["lr"]) / num_learn_episodes

        for eps in tqdm(range(num_learn_episodes), desc="Learning", unit="episodes"):
            average_reward, policy_loss, value_loss = self._learn_episode(eps)

            for param_group in self.policy_optim.param_groups:
                param_group["lr"] += policy_lr_update
            for param_group in self.value_function_optim.param_groups:
                param_group["lr"] += vf_lr_update

            additional_metrics = self._learn_episode_additional_metrics()
            with torch.no_grad():
                logger.on_learning_episode(eps, average_reward, policy_loss, value_loss, num_learn_episodes, additional_metrics=additional_metrics)

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def _learn_episode(self, eps: int) -> tuple[float, float, float]:
        """
        Learn a single episode of the policy and value function.

        Args:
            eps: The index of the current learning episode.

        Returns:
            Average reward, policy loss, and value loss for the episode.
        """
        pass

    @jaxtyped(typechecker=beartype)
    def _learn_episode_additional_metrics(self) -> dict[str, Tensor]:
        """
        Get metrics for the learning algorithm. 
        If the learning algorithm has no additional metrics, return an empty dictionary.

        Returns:
            A dictionary of metrics.
        """
        return dict(getattr(self, '_last_episode_additional_metrics', {}))
    