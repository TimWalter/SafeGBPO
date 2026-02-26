from dataclasses import dataclass
from beartype.typing import Optional, Generator

import torch
from beartype import beartype
from jaxtyping import Float, Bool, jaxtyped
from torch import Tensor

from src.learning_algorithms.components.coupled_tensor import CoupledTensor


@dataclass
class CoupledBufferBatch:
    observations: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    next_terminals: torch.Tensor
    values: Optional[torch.Tensor] = None
    next_values: Optional[torch.Tensor] = None
    actions: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None


class CoupledBuffer:
    """
    Buffer that stores coupled tensors for RL learning_algorithms and fills orderly.
    """

    observations: CoupledTensor
    values: CoupledTensor
    actions: CoupledTensor
    safe_actions: CoupledTensor
    log_probs: CoupledTensor
    rewards: CoupledTensor
    advantages: CoupledTensor
    terminals: CoupledTensor

    @jaxtyped(typechecker=beartype)
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
                 ):
        """
        Initialize the coupled buffer.
        Args:
            len_trajectories: The length of the trajectories
            num_envs: The number of environments
            obs_dim: The observation dimension
            store_values: Whether to store the values
            action_dim: The action dimension
            store_log_probs: Whether to store the log probabilities
            store_safe_actions: Whether to store the safe actions
            batch_size: The batch size for sampling
            gamma: The discount factor for advantage calculation
            gae_lambda: The GAE lambda for advantage calculation
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
        self.additional_metrics = {}

        self.t = torch.zeros(self.num_envs, dtype=torch.int64)

        self.observations = CoupledTensor(self.len_trajectories + 1, self.num_envs, self.obs_dim)
        if self.store_values:
            self.values = CoupledTensor(self.len_trajectories + 1, self.num_envs)
        if self.action_dim is not None:
            self.actions = CoupledTensor(self.len_trajectories + 1, self.num_envs, self.action_dim)
            if self.store_log_probs:
                self.log_probs = CoupledTensor(self.len_trajectories + 1, self.num_envs)
            if self.store_safe_actions:
                self.safe_actions = CoupledTensor(self.len_trajectories + 1, self.num_envs, self.action_dim)
        self.rewards = CoupledTensor(self.len_trajectories + 1, self.num_envs)
        if self.gae_lambda is not None:
            self.advantages = CoupledTensor(self.len_trajectories + 1, self.num_envs)
        self.terminals = CoupledTensor(self.len_trajectories + 1, self.num_envs, dtype=torch.bool)

    @jaxtyped(typechecker=beartype)
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
        if self.additional_metrics:
            for key in self.additional_metrics.keys():
                self.additional_metrics[key].reset()

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
            safe_action: Optional[Float[Tensor, "{self.num_envs} {self.action_dim}"]] = None,
            additional_metrics: Optional[dict] = None
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
            additional_metrics: Additional metrics to store
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
            if additional_metrics:
                for key, metric in additional_metrics.items():
                    if key not in self.additional_metrics:
                        self.additional_metrics[key] = CoupledTensor(self.len_trajectories + 1, *metric.shape)    
                    self.additional_metrics[key][self.t] = metric         
        self.rewards[self.t] = reward
        self.terminals[self.t + 1] = terminal

        self.t = self.t + 1

    def aggregate_additional_metrics(self) -> dict:
        """
        Compute the mean of each safeguard metric over the episode.
        """
        aggregated_metrics = {}
        if self.additional_metrics is not None:
            aggregated_metrics = {}
            for key, metric in self.additional_metrics.items():
                if metric.tensor.numel() > 0:
                    aggregated_metrics[key] = metric[:self.t.max()].mean()
            return aggregated_metrics
        return {}

    @jaxtyped(typechecker=beartype)
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

    @jaxtyped(typechecker=beartype)
    def batch(self) -> CoupledBufferBatch:
        env_indices = torch.randint(0, self.num_envs, (self.batch_size,))
        t_indices = torch.round(torch.rand(self.batch_size) * (
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

    @jaxtyped(typechecker=beartype)
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

        indices = torch.randperm(observations.shape[0] - 1)
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

    @jaxtyped(typechecker=beartype)
    def episodic_rewards(self) -> list[Tensor]:
        episode_rewards = []

        for env_idx in range(self.num_envs):
            episode_end_indices = torch.nonzero(self.terminals[:, env_idx]).squeeze(dim=1) + 1
            rewards_env = self.rewards[:self.t[env_idx], env_idx]
            rewards_per_episode = torch.tensor_split(rewards_env, episode_end_indices)
            episode_rewards.extend(rewards_per_episode)

        return episode_rewards
