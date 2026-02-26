import math

import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor

from learning_algorithms.interfaces.learning_algorithm import LearningAlgorithm
from learning_algorithms.components.coupled_buffer import CoupledBuffer
from learning_algorithms.components.value_function import ValueFunction
from envs.simulators.interfaces.simulator import Simulator


class SHAC(LearningAlgorithm):
    """
    Short Horizon Actor Critic (SHAC) algorithm. https://arxiv.org/pdf/2204.07137

    Constants:
        GAMMA: Discount factor for future rewards.
        MAX_GRAD_NORM: Maximum gradient norm for clipping.
    """

    GAMMA = 0.99
    MAX_GRAD_NORM = 1.0

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 env: Simulator,
                 policy_kwargs: dict,
                 policy_optim_kwargs: dict,
                 vf_kwargs: dict,
                 vf_optim_kwargs: dict,
                 regularisation_coefficient: float,
                 len_trajectories: int,
                 polyak_target: float = 0.995,
                 td_weight: float = 0.95,
                 vf_num_fits: int = 16,
                 vf_fit_num_batches: int = 4,
                 ):
        """
        Initialize the SHAC algorithm.

        Args:
            env: The environment to train on.
            policy_kwargs: The keyword arguments for the policy network.
            policy_optim_kwargs: The keyword arguments for the policy optimizer.
            vf_kwargs: The keyword arguments for the value function network.
            vf_optim_kwargs: The keyword arguments for the value function optimizer.
            regularisation_coefficient: Regularisation coefficient for the regularisation towards safe actions.
            len_trajectories: Length of trajectories to collect.
            polyak_target: The soft update coefficient for the target value function.
            td_weight: The soft update coefficient for the estimated state value.
            vf_num_fits: Number of gradient steps to take on the value function per learning episode.
            vf_fit_num_batches: Number of batches to split the trajectory for value function fitting into.
        """
        super().__init__(env, policy_kwargs, policy_optim_kwargs, vf_kwargs, vf_optim_kwargs,
                         regularisation_coefficient, False)

        self.len_trajectories = len_trajectories
        self.polyak_target = polyak_target
        self.td_weight = td_weight
        self.vf_fit_num_batches = vf_fit_num_batches
        self.vf_num_fits = vf_num_fits

        self.target_value_function = ValueFunction(self.env.obs_dim, **vf_kwargs)

        self.buffer = CoupledBuffer(len_trajectories + 1, self.env.num_envs, self.env.obs_dim, True,
                                    self.env.action_dim, store_safe_actions=hasattr(self.env, "safe_action"))

        reset_observation, info = self.env.reset()
        reset_value = self.target_value_function(reset_observation).squeeze(dim=1)
        self.buffer.reset(reset_observation, reset_value)

        self.interactions_per_episode = len_trajectories * self.env.num_envs

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
        self.policy_optim.zero_grad()
        average_reward = self.collect_trajectories()
        policy_loss = self.update_policy()
        value_loss = self.update_value_function()
        self.update_target_value_function()
        self._last_episode_additional_metrics = self.buffer.aggregate_additional_metrics()
        return average_reward, policy_loss, value_loss

    @jaxtyped(typechecker=beartype)
    def collect_trajectories(self) -> float:
        """
        Collect trajectories using the current policy.

        Returns:
            The average reward collected during the trajectories.
        """
        self.env.cut_computation_graph()
        average_reward = 0.0
        t = 0
        while t < self.len_trajectories:
            action = self.policy(self.buffer.observations[self.buffer.t])
            observation, reward, terminated, truncated, info = self.env.step(action)
            terminal = terminated | truncated

            value = self.target_value_function(observation).squeeze(dim=1)

            if t == self.len_trajectories - 1:
                terminal = torch.ones_like(terminal)

            safe_action = self.env.safe_action if hasattr(self.env, "safe_action") else None
            safeguard_metrics  = self.env.safeguard_metrics()  if hasattr(self.env, "safeguard_metrics") else None
            self.buffer.add(observation, reward, terminal, value, action, safe_action=safe_action, additional_metrics=safeguard_metrics)
            t += 1
            average_reward += reward.sum().item()
        return average_reward / self.env.num_envs / self.len_trajectories

    @jaxtyped(typechecker=beartype)
    def update_policy(self) -> float:
        """
        Update the policy using the trajectory loss with a terminal value estimation by
        the target value function.

        Returns:
            The policy loss
        """
        
        exponent = torch.arange(self.len_trajectories + 2).view(-1, 1).repeat(1, self.env.num_envs)

        for end, env in self.buffer.terminals.nonzero():
            exponent[end + 1:, env] -= end + 1

        discount = self.GAMMA ** exponent
        values = torch.where(self.buffer.terminals.tensor,
                             self.buffer.values.tensor,
                             self.buffer.rewards.tensor)
        normalisation = self.env.num_envs * self.len_trajectories + self.buffer.terminals.count_nonzero()

        policy_loss = -(discount * values).sum() / normalisation
        
        if self.buffer.store_safe_actions:
            policy_loss += self.env.regularisation(self.buffer.actions.tensor, 
                                                   self.buffer.safe_actions.tensor,
                                                   safeguard_metrics=self.buffer.additional_metrics)

        policy_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.MAX_GRAD_NORM)
        self.policy_optim.step()

        return policy_loss.item()

    @jaxtyped(typechecker=beartype)
    def update_value_function(self) -> float:
        """
        Polyak update the value function using an MSE loss with the estimated value
        of the collected states.

        Returns:
            The value loss
        """
        with torch.no_grad():
            estimated_values, observations = self.calculate_estimated_values()
        batch_size = math.ceil(len(estimated_values) / self.vf_fit_num_batches)
        value_loss = torch.empty(0)
        for vf_iter in range(self.vf_num_fits):
            for batch_iter in range(self.vf_fit_num_batches):
                start_idx = batch_iter * batch_size
                end_idx = min(start_idx + batch_size, len(estimated_values))
                if start_idx >= len(estimated_values):
                    break
                estimated_values_batch = estimated_values[start_idx: end_idx]
                observations_batch = observations[start_idx: end_idx]

                self.value_function_optim.zero_grad()
                pred_values_batch = self.value_function(observations_batch).squeeze(dim=1)
                value_loss = (pred_values_batch - estimated_values_batch).square().mean()
                value_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), self.MAX_GRAD_NORM)

                self.value_function_optim.step()

        return value_loss.item()

    @jaxtyped(typechecker=beartype)
    def calculate_estimated_values(self) -> tuple[
        Float[Tensor, "{self.len_trajectories}*{self.env.num_envs}"],
        Float[Tensor, "{self.len_trajectories}*{self.env.num_envs} {self.env.obs_dim}"]]:
        """
        Estimate the values of the observed states in td fashion.

        Returns:
            Estimated values and the observations
        """
        estimated_values = torch.zeros(self.len_trajectories * self.env.num_envs)
        observations = torch.zeros(
            (self.len_trajectories * self.env.num_envs, self.env.obs_dim))

        td_coefficients = torch.ones(self.env.num_envs)
        avg_returns = torch.zeros(self.env.num_envs)
        terminal_return = torch.zeros(self.env.num_envs)

        t = self.buffer.t.clone()
        rewards = self.buffer.rewards
        target_values = self.buffer.values
        terminals = self.buffer.terminals

        pos = 0
        t -= 1  # last value is always terminal
        while torch.any(t >= 0):
            to_estimate = (~ terminals[t]) & (t >= 0)
            to_init = terminals[t + 1] & to_estimate
            to_update = (~ terminals[t + 1]) & to_estimate

            te = t[to_estimate]
            ti = t[to_init]
            tu = t[to_update]

            td_coefficients[to_init] = 1
            avg_returns[to_init] = 0
            terminal_return[to_init] = rewards[ti, to_init] + self.GAMMA * \
                                       target_values[ti + 1, to_init]

            td_coefficients[to_update] *= self.td_weight
            avg_returns[to_update] *= self.td_weight * self.GAMMA
            avg_returns[to_update] += self.GAMMA * target_values[tu + 1, to_update]
            geometric_sum = (1 - td_coefficients[to_update]) / (1 - self.td_weight)
            avg_returns[to_update] += geometric_sum * rewards[tu, to_update]
            terminal_return[to_update] = rewards[tu, to_update] + self.GAMMA * \
                                         terminal_return[to_update]

            num_estimations = len(to_estimate.nonzero())
            estimations = (1 - self.td_weight) * avg_returns[to_estimate]
            estimations += td_coefficients[to_estimate] * terminal_return[to_estimate]
            estimated_values[pos:pos + num_estimations] = estimations
            observations[pos:pos + num_estimations] = self.buffer.observations[
                te, to_estimate]
            t -= 1
            pos += num_estimations
        return estimated_values, observations

    @jaxtyped(typechecker=beartype)
    def update_target_value_function(self):
        """
        Polyak update the target value function.
        """
        with torch.no_grad():
            for target_param, param in zip(self.target_value_function.parameters(), self.value_function.parameters()):
                target_param.data.mul_(self.polyak_target)
                target_param.data.add_((1 - self.polyak_target) * param.data)