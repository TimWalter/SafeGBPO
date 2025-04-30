from dataclasses import dataclass
from typing import Optional, Generator

import torch
from beartype import beartype
from jaxtyping import Float, Bool, jaxtyped
from torch import Tensor

from src.algorithms.components.coupled_tensor import CoupledTensor


@dataclass
class CoupledBufferBatch:
    observations: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    next_terminals: torch.Tensor
    values: torch.Tensor | None = None
    next_values: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None
    advantages: torch.Tensor | None = None


class CoupledBuffer:
    """
    Buffer that stores coupled tensors for RL algorithms and fills orderly.

    Attributes:

    """

    observations: CoupledTensor
    values: CoupledTensor
    actions: CoupledTensor
    safe_actions: CoupledTensor
    log_probs: CoupledTensor
    rewards: CoupledTensor
    advantages: CoupledTensor
    terminals: CoupledTensor


    def __init__(self,
                 len_trajectories: int,
                 num_envs: int,
                 obs_dim: int,
                 store_values: bool = False,
                 action_dim: int = None,
                 store_log_probs: bool = False,
                 store_safe_actions: bool = False,
                 batch_size: int = None,
                 gamma: float = None,
                 gae_lambda: float = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        Initialize the coupled buffer.
        Args:
            len_trajectories (int): The length of the trajectories
            num_envs (int): The number of environments
            obs_dim (int): The observation dimension
            store_values (bool): Whether to store the values
            action_dim (int): The action dimension
            store_log_probs (bool): Whether to store the log probabilities
            store_safe_actions (bool): Whether to store the safe actions
            batch_size (int): The batch size for sampling
            gamma (float): The discount factor for advantage calculation
            gae_lambda (float): The GAE lambda for advantage calculation
            device (torch.device): The device to store the tensors on
        """
        self.len_trajectories = len_trajectories
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.store_values = store_values
        self.action_dim = action_dim
        self.store_log_probs = store_log_probs
        self.store_safe_actions = store_safe_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.t = torch.zeros(self.num_envs, dtype=torch.int64).to(self.device)

        self.observations = CoupledTensor(
            (self.len_trajectories + 1, self.num_envs, self.obs_dim),
            dtype=torch.float64, device=self.device)
        if self.store_values:
            self.values = CoupledTensor(
                (self.len_trajectories + 1, self.num_envs),
                dtype=torch.float64, device=self.device)
        if self.action_dim is not None:
            self.actions = CoupledTensor(
                (self.len_trajectories + 1, self.num_envs, self.action_dim),
                dtype=torch.float64, device=self.device)
            if self.store_log_probs:
                self.log_probs = CoupledTensor(
                    (self.len_trajectories + 1, self.num_envs),
                    dtype=torch.float64, device=self.device)
            if self.store_safe_actions:
                self.safe_actions = CoupledTensor(
                    (self.len_trajectories + 1, self.num_envs, self.action_dim),
                    dtype=torch.float64, device=self.device)
        self.rewards = CoupledTensor(
            (self.len_trajectories + 1, self.num_envs),
            dtype=torch.float64, device=self.device)
        if self.gae_lambda is not None:
            self.advantages = CoupledTensor(
                (self.len_trajectories + 1, self.num_envs),
                dtype=torch.float64, device=self.device)
        self.terminals = CoupledTensor(
            (self.len_trajectories + 1, self.num_envs),
            dtype=torch.bool, device=self.device)

    def reset(self,
              reset_observation: Optional[
                  Float[Tensor, "{self.num_envs} {self.obs_dim}"]] = None,
              reset_value: Optional[
                  Float[Tensor, "{self.num_envs}"]] = None) -> None:
        """
        Clear the buffer and cut computation graph.

        Args:
            reset_observation: The reset observation to set the buffer to
            reset_value:  The respective target value
        """
        if reset_observation is None:
            reset_observation = self.observations[self.t].detach()
            if self.store_values:
                reset_value = self.values[self.t].detach()

        self.t.fill_(0)
        self.t = self.t.detach()

        self.observations.reset()
        if self.store_values:
            self.values.reset()
        if self.action_dim is not None:
            self.actions.reset()
            if self.store_log_probs:
                self.log_probs.reset()
            if self.store_safe_actions:
                self.safe_actions.reset()
        self.rewards.reset()
        if self.gae_lambda is not None:
            self.advantages.reset()
        self.terminals.reset()

        self.observations[0] = reset_observation
        if self.store_values:
            self.values[0] = reset_value

    @jaxtyped(typechecker=beartype)
    def add(
            self,
            observation: Float[Tensor, "{self.num_envs} {self.obs_dim}"],
            reward: Float[Tensor, "{self.num_envs}"],
            terminal: Bool[Tensor, "{self.num_envs}"],
            value: Optional[Float[Tensor, "{self.num_envs}"]] = None,
            action: Optional[Float[Tensor, "{self.num_envs} {self.action_dim}"]] = None,
            log_prob: Optional[Float[Tensor, "{self.num_envs}"]] = None,
            safe_action: Optional[Float[Tensor, "{self.num_envs} {self.action_dim}"]] = None
    ) -> None:
        """
        Add a new transition to the buffer

        Args:
            observation: Observation at time t + 1
            reward: Reward caused by the action/observation at time t
            terminal: Mask indicating if the episode ends at time t + 1
            value: State evaluation at time t + 1
            action: Action at time t
            log_prob: Log probability of the action at time t
            safe_action: Safe action at time t
        """
        if (self.t >= self.len_trajectories).any():
            print("[BUFFER OVERFLOW] Overwriting oldest transitions")
            self.t[self.t >= self.len_trajectories] = 0

        self.observations[self.t + 1] = observation
        if value is not None:
            self.values[self.t + 1] = value
        if action is not None:
            self.actions[self.t] = action
            if log_prob is not None:
                self.log_probs[self.t] = log_prob
            if safe_action is not None:
                self.safe_actions[self.t] = safe_action
        self.rewards[self.t] = reward
        self.terminals[self.t + 1] = terminal

        self.t = self.t + 1

    def calculate_advantages(self) -> None:
        t = self.t.detach() - 1
        while (t >= 0).any():
            reward = self.rewards[t[t >= 0], t >= 0]
            value = self.values[t[t >= 0], t >= 0]
            next_value = self.values[t[t >= 0] + 1, t >= 0]
            terminal = self.terminals[t[t >= 0] + 1, t >= 0]
            if not (t == self.t - 1).all():
                next_advantage = self.advantages[t[t >= 0] + 1, t >= 0]
            else:
                next_advantage = 0

            advantage = reward - value + self.gamma * ~terminal * (
                    next_value + self.gae_lambda * next_advantage)
            self.advantages[t[t >= 0], t >= 0] = advantage
            t = t - 1

    def batch(self) -> CoupledBufferBatch:
        env_indices = torch.randint(0, self.num_envs, (self.batch_size,),
                                    device=self.device)
        t_indices = torch.round(torch.rand(self.batch_size, device=self.device) * (
                self.t[env_indices] - 2)).type(torch.int64)

        batch = CoupledBufferBatch(
            observations=self.observations[t_indices, env_indices],
            next_observations=self.observations[t_indices + 1, env_indices],
            rewards=self.rewards[t_indices, env_indices],
            terminals=self.terminals[t_indices, env_indices],
            next_terminals=self.terminals[t_indices + 1, env_indices]
        )

        if self.store_values:
            batch.values = self.values[t_indices, env_indices]
            batch.next_values = self.values[t_indices + 1, env_indices]
        if self.action_dim is not None:
            batch.actions = self.actions[t_indices, env_indices]
            if self.store_log_probs:
                batch.log_probs = self.log_probs[t_indices, env_indices]
        if self.gae_lambda is not None:
            batch.advantages = self.advantages[t_indices, env_indices]

        return batch

    def unique_batches(self) -> Generator[CoupledBufferBatch, None, None]:
        observations = torch.cat(
            [self.observations[:self.t[env_idx], env_idx, :] for env_idx in
             range(self.num_envs)], dim=0
        )
        if self.store_values:
            values = torch.cat(
                [self.values[:self.t[env_idx], env_idx] for env_idx in
                 range(self.num_envs)], dim=0
            )
        if self.action_dim is not None:
            actions = torch.cat(
                [self.actions[:self.t[env_idx], env_idx, :] for env_idx in
                 range(self.num_envs)], dim=0
            )
            if self.store_log_probs:
                log_probs = torch.cat(
                    [self.log_probs[:self.t[env_idx], env_idx] for env_idx in
                     range(self.num_envs)], dim=0
                )
        rewards = torch.cat(
            [self.rewards[:self.t[env_idx], env_idx] for env_idx in
             range(self.num_envs)], dim=0
        )
        if self.advantages is not None:
            advantages = torch.cat(
                [self.advantages[:self.t[env_idx], env_idx] for env_idx in
                 range(self.num_envs)], dim=0
            )
        terminals = torch.cat(
            [self.terminals[:self.t[env_idx], env_idx] for env_idx in
             range(self.num_envs)], dim=0
        )

        indices = torch.randperm(observations.shape[0] - 1, device=self.device)
        for start in range(0, observations.shape[0], self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = CoupledBufferBatch(
                observations=observations[batch_indices],
                next_observations=observations[batch_indices + 1],
                rewards=rewards[batch_indices],
                terminals=terminals[batch_indices],
                next_terminals=terminals[batch_indices + 1]
            )
            if self.store_values:
                batch.values = values[batch_indices]
                batch.next_values = values[batch_indices + 1]
            if self.action_dim is not None:
                batch.actions = actions[batch_indices]
                if self.store_log_probs:
                    batch.log_probs = log_probs[batch_indices]
            if self.advantages is not None:
                batch.advantages = advantages[batch_indices]

            yield batch

    def episodic_rewards(self):
        episode_rewards = []

        for env_idx in range(self.num_envs):
            # Find the indices where an episode ends for the current environment
            episode_end_indices = torch.nonzero(self.terminals[:, env_idx]).squeeze(dim=1)+1

            # Get all reward values for the current environment
            rewards_env = self.rewards[:self.t[env_idx], env_idx]


            rewards_per_episode = torch.tensor_split(rewards_env, episode_end_indices)

            # Add the rewards for this environment
            episode_rewards.append(rewards_per_episode)

        return episode_rewards