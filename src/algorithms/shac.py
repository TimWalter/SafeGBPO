import math
from typing import Optional, Literal

import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor

from algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from algorithms.components.coupled_buffer import CoupledBuffer
from algorithms.components.policy import Policy
from algorithms.components.value_function import ValueFunction
from tasks.envs.interfaces.torch_vector_env import TorchVectorEnv


class SHAC(ActorCriticAlgorithm):
    """
    Short Horizon Actor Critic (SHAC) algorithm. https://arxiv.org/pdf/2204.07137

    Attributes:
        env (TorchVectorEnv): The environment to learn from.
        len_trajectories (int): Length of trajectories to collect.
        gamma (float): The discount factor.
        polyak_target (float): The soft update coefficient for the target function.
        td_weight (float): The soft update coefficient for the estimated state value.
        vf_num_fits (int): Number of gradient steps to take on the value function per learning episode.
        vf_fit_num_batches (int): Number of batches to split the trajectory for value function fitting into.
        clip_grad (bool): Whether to clip the gradients.
        max_grad_norm (float): The maximum gradient norm
        policy (Policy): The policy to be learned.
        value_function (ValueFunction): The value function to be learned.
        target_value_function (ValueFunction): The target value function.
        policy_optim (torch.optim): The optimizer for the policy.
        value_function_optim (torch.optim): The optimizer for the value function.
        buffer (RolloutBuffer): The buffer to store the trajectories.
        policy_learning_rate_schedule (str): The learning rate schedule for the policy.
        vf_learning_rate_schedule (str): The learning rate schedule for the value function.
    """

    def __init__(self,
                 env: TorchVectorEnv,
                 len_trajectories: int,
                 gamma: float = 0.99,
                 polyak_target: float = 0.995,
                 td_weight: float = 0.95,
                 policy_kwargs: dict = None,
                 policy_learning_rate_schedule: str = "constant",
                 policy_optim_kwargs: dict = None,
                 vf_kwargs: dict = None,
                 vf_learning_rate_schedule: str = "constant",
                 vf_optim_kwargs: dict = None,
                 vf_num_fits: int = 16,
                 vf_fit_num_batches: int = 4,
                 clip_grad: bool = True,
                 max_grad_norm: float = 1.0,
                 device: Optional[Literal["cpu", "cuda"]] = None,
                 regularisation_coefficient: float = 0.1,
                 adaptive_regularisation: bool = False
                 ):
        """
        Initialize the SHAC algorithm.

        Args:
            env (TorchVectorEnv): The environment to learn from.
            len_trajectories (int): Length of trajectories to collect.
            gamma (float): The discount factor.
            polyak_target (float): The soft update coefficient for the target value function.
            td_weight (float): The soft update coefficient for the estimated state value.
            policy_kwargs (dict): Additional arguments to be passed to the policy on creation.
            policy_learning_rate_schedule (str): The learning rate schedule for the policy.
            policy_optim_kwargs (dict): Additional arguments to be passed to the policy optimizer on creation.
            vf_kwargs (dict): Additional arguments to be passed to the value function on creation.
            vf_learning_rate_schedule (str): The learning rate schedule for the value function.
            vf_optim_kwargs (dict): Additional arguments to be passed to the value function optimizer on creation.
            vf_num_fits (int): Number of gradient steps to take on the value function per learning episode.
            vf_fit_num_batches (int): Number of batches to split the trajectory for value function fitting into.
            clip_grad (bool): Whether to clip the gradients.
            max_grad_norm (float): The maximum gradient norm
            device (str): The device to run the algorithm on.
            regularisation_coefficient (float): The coefficient for the regularisation term.
            adaptive_regularisation (bool): Whether to use adaptive regularisation.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.env = env
        self.obs_dim = env.observation_space.shape[1]
        self.action_dim = env.action_space.shape[1]
        self.num_envs = env.num_envs
        self.len_trajectories = len_trajectories
        self.gamma = gamma
        self.polyak_target = polyak_target
        self.td_weight = td_weight
        self.vf_fit_num_batches = vf_fit_num_batches
        self.vf_num_fits = vf_num_fits
        self.clip_grad = clip_grad
        self.max_grad_norm = max_grad_norm

        self.policy = Policy(self.obs_dim, self.action_dim, **policy_kwargs,
                             device=self.device)
        self.value_function = ValueFunction(self.obs_dim, **vf_kwargs,
                                            device=self.device)
        self.target_value_function = ValueFunction(self.obs_dim, **vf_kwargs,
                                                   device=self.device)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(),
                                             **policy_optim_kwargs)
        self.value_function_optim = torch.optim.Adam(self.value_function.parameters(),
                                                     **vf_optim_kwargs)

        self.safe = hasattr(env, "safe_actions")
        self.buffer = CoupledBuffer(len_trajectories + 1, self.num_envs, self.obs_dim,
                                    True, self.action_dim, store_safe_actions=self.safe,
                                    device=self.device)
        reset_observation, info = self.env.reset()
        reset_value = self.target_value_function(reset_observation).squeeze(dim=1)
        self.buffer.reset(reset_observation, reset_value)

        self.policy_learning_rate_schedule = policy_learning_rate_schedule
        self.vf_learning_rate_schedule = vf_learning_rate_schedule

        self.samples_per_episode = len_trajectories * self.num_envs
        self.regularisation_coefficient = regularisation_coefficient
        self.adaptive_regularisation = adaptive_regularisation

    def _learn_episode(self, eps: int) -> tuple[float, float]:
        self.buffer.reset()
        self.policy_optim.zero_grad()
        self.collect_trajectories()
        policy_loss = self.update_policy()
        value_loss = self.update_value_function()
        self.update_target_value_function()
        return policy_loss, value_loss

    def collect_trajectories(self):
        """
        Collect trajectories using the current policy.
        """
        self.env.clear_computation_graph()
        t = 0
        while t < self.len_trajectories:
            action = self.policy(self.buffer.observations[self.buffer.t])
            observation, reward, terminated, truncated, info = self.env.step(action)
            terminal = terminated | truncated

            value = self.target_value_function(observation).squeeze(dim=1)

            if t == self.len_trajectories - 1:
                terminal = torch.ones_like(terminal)

            safe_action = self.env.safe_actions if self.safe else None
            self.buffer.add(observation, reward, terminal, value, action,
                            safe_action=safe_action)
            t += 1

    def update_policy(self) -> float:
        """
        Update the policy using the trajectory loss with a terminal value estimation by
        the target value function.

        Returns:
            The policy loss
        """
        exponent = torch.arange(self.len_trajectories + 2,
                                device=self.device, dtype=torch.float64).view(-1, 1).repeat(1, self.num_envs)

        for end, env in self.buffer.terminals.nonzero():
            exponent[end + 1:, env] -= end + 1

        discount = self.gamma ** exponent
        values = torch.where(self.buffer.terminals.tensor,
                             self.buffer.values.tensor,
                             self.buffer.rewards.tensor)
        normalisation = (self.num_envs * self.len_trajectories
                         + self.buffer.terminals.count_nonzero())

        policy_loss = -(discount * values).sum() / normalisation
        if self.safe:
            regularisation = torch.nn.functional.mse_loss(
                self.buffer.safe_actions.tensor, self.buffer.actions.tensor)
            if self.adaptive_regularisation:
                policy_loss = policy_loss * (
                            1 + self.regularisation_coefficient * regularisation)
            else:
                policy_loss = policy_loss + self.regularisation_coefficient * regularisation
        policy_loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optim.step()

        return policy_loss

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
        value_loss = torch.empty(0, device=self.device, dtype=torch.float64)
        for vf_iter in range(self.vf_num_fits):
            for batch_iter in range(self.vf_fit_num_batches):
                start_idx = batch_iter * batch_size
                end_idx = min(start_idx+batch_size, len(estimated_values))
                if start_idx >= len(estimated_values):
                    break
                estimated_values_batch = estimated_values[start_idx: end_idx]
                observations_batch = observations[start_idx: end_idx]

                self.value_function_optim.zero_grad()
                pred_values_batch = self.value_function(observations_batch).squeeze(
                    dim=1)
                value_loss = (
                        pred_values_batch - estimated_values_batch).square().mean()
                value_loss.backward()

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.value_function.parameters(),
                                                   self.max_grad_norm)

                self.value_function_optim.step()


        return value_loss.item()

    @jaxtyped(typechecker=beartype)
    def calculate_estimated_values(self) -> tuple[
        Float[Tensor, "{self.len_trajectories}*{self.num_envs}"],
        Float[Tensor, "{self.len_trajectories}*{self.num_envs} {self.obs_dim}"]]:
        """
        Estimate the values of the observed states in td fashion.

        Returns:
            Estimated values and the observations
        """
        estimated_values = torch.zeros(self.len_trajectories * self.num_envs,
                                       device=self.device, dtype=torch.float64)
        observations = torch.zeros(
            (self.len_trajectories * self.num_envs, self.obs_dim),
            device=self.device, dtype=torch.float64)

        td_coefficients = torch.ones(self.num_envs, device=self.device, dtype=torch.float64)
        avg_returns = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        terminal_return = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)

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
            terminal_return[to_init] = rewards[ti, to_init] + self.gamma * \
                                       target_values[ti + 1, to_init]

            td_coefficients[to_update] *= self.td_weight
            avg_returns[to_update] *= self.td_weight * self.gamma
            avg_returns[to_update] += self.gamma * target_values[tu + 1, to_update]
            geometric_sum = (1 - td_coefficients[to_update]) / (1 - self.td_weight)
            avg_returns[to_update] += geometric_sum * rewards[tu, to_update]
            terminal_return[to_update] = rewards[tu, to_update] + self.gamma * \
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

    def update_target_value_function(self):
        """
        Polyak update the target value function.
        """
        with torch.no_grad():
            for target_param, param in zip(self.target_value_function.parameters(),
                                           self.value_function.parameters()):
                target_param.data.mul_(self.polyak_target)
                target_param.data.add_((1 - self.polyak_target) * param.data)
