import torch
from beartype import beartype
from jaxtyping import jaxtyped

from envs.simulators.interfaces.simulator import Simulator
from learning_algorithms.interfaces.learning_algorithm import LearningAlgorithm
from src.learning_algorithms.components.coupled_buffer import CoupledBuffer, CoupledBufferBatch


class PPO(LearningAlgorithm):
    """
    Proximal Policy Optimization (PPO) Algorithm. https://arxiv.org/abs/1707.06347

    Constants:
        GAMMA: Discount factor for future rewards.
        GAE_LAMBDA: Lambda parameter for Generalized Advantage Estimation.
        MAX_GRAD_NORM: Maximum gradient norm for clipping.
    """

    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    MAX_GRAD_NORM = 0.5

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 env: Simulator,
                 policy_kwargs: dict,
                 policy_optim_kwargs: dict,
                 vf_kwargs: dict,
                 vf_optim_kwargs: dict,
                 regularisation_coefficient: float,
                 len_trajectories: int,
                 clip_coef: float,
                 ent_coef: float,
                 num_batches: int,
                 num_fits: int,
                 ):
        """
        Initialize the PPO algorithm.

        Args:
            env: The environment to train on.
            policy_kwargs: The keyword arguments for the policy network.
            policy_optim_kwargs: The keyword arguments for the policy optimizer.
            vf_kwargs: The keyword arguments for the value function network.
            vf_optim_kwargs: The keyword arguments for the value function optimizer.
            regularisation_coefficient: Regularisation coefficient for the regularisation towards safe actions.
            clip_coef: The surrogate clipping coefficient.
            ent_coef: The entropy coefficient.
            num_batches: The number of mini-batches to split the data into.
            num_fits: The number of optimization steps to take on each mini-batch.
        """
        super().__init__(env, policy_kwargs, policy_optim_kwargs, vf_kwargs, vf_optim_kwargs,
                         regularisation_coefficient, False)
        self.len_trajectories = len_trajectories
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.batch_size = self.env.num_envs * len_trajectories // num_batches
        self.num_fits = num_fits

        self.buffer = CoupledBuffer(len_trajectories, self.env.num_envs, self.env.obs_dim, True,
                                    self.env.action_dim, True, hasattr(self.env, "safe_action"),
                                    self.batch_size, self.GAMMA, self.GAE_LAMBDA)

        reset_observations, info = self.env.reset()
        with torch.no_grad():
            reset_values = self.value_function(reset_observations).squeeze(dim=1)
        self.buffer.reset(reset_observations, reset_values)

        self.interactions_per_episode = self.len_trajectories * self.env.num_envs

    @jaxtyped(typechecker=beartype)
    def _learn_episode(self, eps: int) -> tuple[float, float, float]:
        """
        Learn a single episode of the policy and value function.

        Args:
            eps: The index of the current learning episode.

        Returns:
            Average reward, policy loss, and value loss for the episode.
        """
        self.buffer.reset()
        average_reward = self.collect_trajectories()
        with torch.no_grad():
            self.buffer.calculate_advantages()

        policy_loss = torch.nan
        value_loss = torch.nan
        for _ in range(self.num_fits):
            for batch in self.buffer.unique_batches():
                if batch.observations.shape[0] != 0:
                    policy_loss = self.update_policy(batch)
                    value_loss = self.update_value_function(batch)

        self._last_episode_additional_metrics = self.buffer.aggregate_additional_metrics()
        return average_reward, policy_loss, value_loss

    @jaxtyped(typechecker=beartype)
    def collect_trajectories(self) -> float:
        """
        Collect trajectories from the environment using the current policy.

        Returns:
            The average reward from the collected trajectories.
        """
        average_reward = 0.0
        for t in range(self.len_trajectories):
            with torch.no_grad():
                action = self.policy(self.buffer.observations[self.buffer.t])
                log_prob = self.policy.log_prob(action)

            observation, reward, terminated, truncated, info = self.env.step(action)
            with torch.no_grad():
                value = self.value_function(observation).squeeze(dim=1)
            terminal = terminated | truncated

            safe_action = self.env.safe_action if hasattr(self.env, "safe_action") else None
            safeguard_metrics  = self.env.safeguard_metrics()  if hasattr(self.env, "safeguard_metrics") else None
            self.buffer.add(observation, reward, terminal, value, action, log_prob, safe_action=safe_action, additional_metrics=safeguard_metrics)
            average_reward += reward.sum().item()
        return average_reward / self.env.num_envs / self.len_trajectories

    @jaxtyped(typechecker=beartype)
    def update_policy(self, batch: CoupledBufferBatch) -> float:
        """
        Update the policy network using the PPO surrogate loss.

        Args:
            batch: The batch of data to use for the update.

        Returns:
            The policy loss.
        """
        curr_log_prob = self.policy.log_prob(batch.actions, batch.observations)  
        entropy = self.policy.entropy()

        log_prob_diff = torch.clamp(curr_log_prob - batch.log_probs, min=-20, max=20)
        prob_ratio = log_prob_diff.exp()

        advantages = batch.advantages
        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss = -advantages * prob_ratio
        loss_clamped = -advantages * torch.clamp(prob_ratio,
                                                 1 - self.clip_coef,
                                                 1 + self.clip_coef)
        policy_loss = torch.max(loss, loss_clamped).mean() - self.ent_coef * entropy.mean()

        if self.buffer.store_safe_actions:
            policy_loss += self.env.regularisation(self.buffer.actions.tensor, 
                                                   self.buffer.safe_actions.tensor,
                                                   safeguard_metrics= self.buffer.additional_metrics)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.MAX_GRAD_NORM)
        self.policy_optim.step()

        return policy_loss.item()

    @jaxtyped(typechecker=beartype)
    def update_value_function(self, batch: CoupledBufferBatch) -> float:
        """
        Update the value function network using a mse loss on the value predictions
        and the return.

        Args:
            batch: The batch of data to use for the update.

        Returns:
            The value loss.
        """

        curr_value = self.value_function(batch.observations).view(-1)

        value_loss = (curr_value - batch.advantages - batch.values) ** 2
        values_clipped = batch.values + torch.clamp(
            curr_value - batch.values,
            -self.clip_coef,
            self.clip_coef,
        )
        value_loss_clipped = (values_clipped - batch.advantages - batch.values) ** 2
        value_loss = torch.max(value_loss, value_loss_clipped)
        value_loss = 0.5 * value_loss.mean()

        self.value_function_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), self.MAX_GRAD_NORM)
        self.value_function_optim.step()

        return value_loss.item()
