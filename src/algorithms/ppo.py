from typing import Optional, Literal

import torch
from torch import Tensor

from algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from algorithms.components.policy import Policy
from algorithms.components.value_function import ValueFunction
from tasks.envs.interfaces.torch_vector_env import TorchVectorEnv
from src.algorithms.components.coupled_buffer import CoupledBuffer, CoupledBufferBatch


class PPO(ActorCriticAlgorithm):
    """
    Proximal Policy Optimization (PPO) Algorithm. https://arxiv.org/abs/1707.06347

    Attributes:
        device (Optional[Literal["cpu", "cuda"]]): The device to use for training
        env (TorchVectorEnv): The environment to train on.
        obs_dim (int): The dimension of the observation space.
        action_dim (int): The dimension of the action space.
        num_envs (int): The number of environments to run in parallel.
        len_trajectories (int): The number of steps in each trajectory.
        batch_size (int): The number of samples in each mini-batch.
        num_fits (int): The number of optimization steps to take on each mini-batch.
        norm_adv (bool): Whether to normalize the advantages.
        clip_coef (float): The surrogate clipping coefficient.
        clip_value_loss (bool): Whether to clip the value loss.
        ent_coef (float): The entropy coefficient.
        max_grad_norm (float): The maximum gradient norm.
        target_kl (float): The target KL divergence.
        policy (Policy): The policy network.
        value_function (ValueFunction): The value function network.
        policy_optim (torch.optim.Optimizer): The optimizer for the policy network.
        value_function_optim (torch.optim.Optimizer): The optimizer for the value function network.
        policy_learning_rate_schedule (str): The learning rate schedule for the policy network.
        vf_learning_rate_schedule (str): The learning rate schedule for the value function network.
        buffer (CoupledBuffer): The buffer to store trajectories.
    """

    def __init__(self,
                 env: TorchVectorEnv,
                 len_trajectories: int = 2048,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 num_batches: int = 32,
                 num_fits: int = 10,
                 norm_adv: bool = True,
                 clip_coef: float = 0.2,
                 clip_value_loss: bool = True,
                 ent_coef: float = 0.0,
                 target_kl: float = None,
                 policy_kwargs: dict = None,
                 policy_learning_rate_schedule: str = "constant",
                 policy_optim_kwargs: dict = None,
                 vf_kwargs: dict = None,
                 vf_learning_rate_schedule: str = "constant",
                 vf_optim_kwargs: dict = None,
                 max_grad_norm: float = 0.5,
                 device: Optional[Literal["cpu", "cuda"]] = None
                 ):
        """
        Initialize the PPO algorithm.

        Args:
            env (TorchVectorEnv): The environment to train on.
            len_trajectories (int): The number of steps in each trajectory.
            gamma (float): The discount factor.
            gae_lambda (float): The GAE lambda.
            num_batches (int): The number of mini-batches to split the data into.
            num_fits (int): The number of optimization steps to take on each mini-batch.
            norm_adv (bool): Whether to normalize the advantages.
            clip_coef (float): The surrogate clipping coefficient.
            clip_value_loss (bool): Whether to clip the value loss.
            ent_coef (float): The entropy coefficient.
            target_kl (float): The target KL divergence.
            policy_kwargs (dict): The keyword arguments for the policy network.
            policy_learning_rate_schedule (str): The learning rate schedule for the policy network.
            policy_optim_kwargs (dict): The keyword arguments for the policy optimizer.
            vf_kwargs (dict): The keyword arguments for the value function network.
            vf_learning_rate_schedule (str): The learning rate schedule for the value function network.
            vf_optim_kwargs (dict): The keyword arguments for the value function optimizer.
            max_grad_norm (float): The maximum gradient norm.
            device (Optional[Literal["cpu", "cuda"]]): The device to use for training
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
        self.batch_size = self.num_envs * len_trajectories // num_batches
        self.num_fits = num_fits
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_value_loss = clip_value_loss
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.policy = Policy(self.obs_dim, self.action_dim, **policy_kwargs,
                             device=self.device)
        self.value_function = ValueFunction(self.obs_dim, **vf_kwargs,
                                            device=self.device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(),
                                             **policy_optim_kwargs)
        self.value_function_optim = torch.optim.Adam(self.value_function.parameters(),
                                                     **vf_optim_kwargs)

        self.policy_learning_rate_schedule = policy_learning_rate_schedule
        self.vf_learning_rate_schedule = vf_learning_rate_schedule

        self.buffer = CoupledBuffer(len_trajectories, self.num_envs, self.obs_dim, True,
                                    self.action_dim, True, False, self.batch_size,
                                    gamma,
                                    gae_lambda,
                                    self.device)
        reset_observations, info = self.env.reset()
        with torch.no_grad():
            reset_values = self.value_function(reset_observations).squeeze(dim=1)
        self.buffer.reset(reset_observations, reset_values)

        self.samples_per_episode = self.len_trajectories * self.num_envs

    def _learn_episode(self, eps: int) -> tuple[float, float]:
        self.buffer.reset()
        self.collect_trajectories()
        with torch.no_grad():
            self.buffer.calculate_advantages()

        policy_loss = torch.nan
        value_loss = torch.nan
        for _ in range(self.num_fits):
            prob_ratio = torch.tensor([1])
            for batch in self.buffer.unique_batches():
                if batch.observations.shape[0] != 0:
                    prob_ratio, policy_loss = self.update_policy(batch)
                    value_loss = self.update_value_function(batch)
            with torch.no_grad():
                approx_kl = ((prob_ratio - 1) - torch.log(prob_ratio)).mean()
            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        return policy_loss, value_loss

    def collect_trajectories(self):
        """
        Collect trajectories from the environment using the current policy.
        """
        for t in range(self.len_trajectories):
            with torch.no_grad():
                action = self.policy(self.buffer.observations[self.buffer.t])
                log_prob = self.policy.log_prob(action)

            observation, reward, terminated, truncated, info = self.env.step(action)
            with torch.no_grad():
                value = self.value_function(observation).squeeze(dim=1)
            terminal = terminated | truncated

            self.buffer.add(observation, reward, terminal, value, action, log_prob)

    def update_policy(self, batch: CoupledBufferBatch) -> tuple[Tensor, float]:
        """
        Update the policy network using the PPO surrogate loss.

        Args:
            batch (CoupledBufferBatch): The batch of data to use for the update.

        Returns:
            tuple[float, float]: The probability ratio and the policy loss.
        """

        curr_log_prob = self.policy.log_prob(batch.actions, batch.observations)
        entropy = self.policy.entropy()

        log_prob_diff = torch.clamp(curr_log_prob - batch.log_probs, min=-20, max=20)
        prob_ratio = log_prob_diff.exp()

        advantages = batch.advantages
        if self.norm_adv and advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss = -advantages * prob_ratio
        loss_clamped = -advantages * torch.clamp(prob_ratio,
                                                 1 - self.clip_coef,
                                                 1 + self.clip_coef)
        policy_loss = torch.max(loss,
                                loss_clamped).mean() - self.ent_coef * entropy.mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                       self.max_grad_norm)
        self.policy_optim.step()

        return prob_ratio, policy_loss.item()

    def update_value_function(self, batch: CoupledBufferBatch) -> float:
        """
        Update the value function network using a mse loss on the value predictions
        and the return.

        Args:
            batch (CoupledBufferBatch): The batch of data to use for the update.

        Returns:
            float: The value loss.
        """

        curr_value = self.value_function(batch.observations).view(-1)

        value_loss = (curr_value - batch.advantages - batch.values) ** 2
        if self.clip_value_loss:
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
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(),
                                       self.max_grad_norm)
        self.value_function_optim.step()

        return value_loss.item()
