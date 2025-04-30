from typing import Optional, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from algorithms.components.coupled_buffer import CoupledBuffer, CoupledBufferBatch
from algorithms.components.policy import Policy
from algorithms.components.value_function import ValueFunction
from tasks.envs.interfaces.torch_vector_env import TorchVectorEnv


class SAC(ActorCriticAlgorithm):
    """
    Soft Actor-Critic (SAC) Algorithm. https://arxiv.org/pdf/1812.05905

    Attributes:
        device (torch.device): The device to run the algorithm on.
        env (TorchVectorEnv): The environment to train on.
        obs_dim (int): The dimension of the observation space.
        action_dim (int): The dimension of the action space.
        num_envs (int): The number of environments to train on.
        gamma (float): The discount factor.
        polyak_target (float): The soft update coefficient for the target function.
        learning_starts (int): The number of episodes to collect before training.
        autotune (bool): Whether to automatically tune the entropy coefficient.
        alpha (torch.Tensor): The entropy regularization coefficient.
        policy_frequency (int): The frequency and number of policy updates.
        target_frequency (int): The frequency to update the target value function.
        policy (Policy): The policy network.
        value_function1 (ValueFunction): The first value function network.
        value_function2 (ValueFunction): The second value function network.
        target_value_function1 (ValueFunction): The first target value function network.
        target_value_function2 (ValueFunction): The second target value function network.
        policy_optim (torch.optim.Optimizer): The optimizer for the policy network.
        value_function_optim (torch.optim.Optimizer): The optimizer for the value function networks.
        policy_learning_rate_schedule (str): The learning rate schedule for the policy network.
        vf_learning_rate_schedule (str): The learning rate schedule for the value function networks.
        buffer (CoupledBuffer): The replay buffer to store experiences.
    """

    def __init__(self,
                 env: TorchVectorEnv,
                 buffer_size: int = int(1e6),
                 gamma: float = 0.99,
                 polyak_target: float = 0.995,
                 batch_size: int = 256,
                 learning_starts: int = int(5e3),
                 alpha: float = 0.2,
                 autotune: bool = True,
                 policy_kwargs: dict = None,
                 policy_frequency: int = 2,
                 policy_learning_rate_schedule: str = "constant",
                 policy_optim_kwargs: dict = None,
                 vf_kwargs: dict = None,
                 vf_learning_rate_schedule: str = "constant",
                 vf_optim_kwargs: dict = None,
                 target_frequency: int = 1,
                 device: Optional[Literal["cpu", "cuda"]] = None
                 ):
        """
        Initialize the SAC algorithm.

        Args:
            env (TorchVectorEnv): The environment to train on.
            buffer_size (int): The maximal size of the buffer.
            gamma (float): The discount factor.
            polyak_target (float): The soft update coefficient for the target function.
            batch_size (int): The batch size to sample from the buffer.
            learning_starts (int): The number of episodes to collect before training.
            alpha (float): The entropy regularization coefficient.
            autotune (bool): Whether to automatically tune the entropy coefficient.
            policy_kwargs (dict): The keyword arguments for the policy network.
            policy_frequency (int): The frequency and number of policy updates.
            policy_learning_rate_schedule (str): The learning rate schedule for the policy network.
            policy_optim_kwargs (dict): The keyword arguments for the policy optimizer.
            vf_kwargs (dict): The keyword arguments for the value function networks.
            vf_learning_rate_schedule (str): The learning rate schedule for the value function networks.
            vf_optim_kwargs (dict): The keyword arguments for the value function optimizer.
            target_frequency (int): The frequency to update the target value function.
            device (Optional[Literal["cpu", "cuda"]]): The device to run the algorithm on.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.env = env
        self.obs_dim = env.observation_space.shape[1]
        self.action_dim = env.action_space.shape[1]
        self.num_envs = env.num_envs
        self.gamma = gamma
        self.polyak_target = polyak_target
        self.learning_starts = learning_starts
        self.autotune = autotune
        if self.autotune:
            self.alpha = torch.tensor([1.], requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.alpha], **vf_optim_kwargs)
        else:
            self.alpha = torch.tensor([alpha], device=self.device)

        self.policy_frequency = policy_frequency
        self.target_frequency = target_frequency

        self.policy = Policy(self.obs_dim, self.action_dim, **policy_kwargs,
                             device=self.device)
        self.value_function1 = ValueFunction(self.obs_dim + self.action_dim,
                                             **vf_kwargs,
                                             device=self.device)
        self.value_function2 = ValueFunction(self.obs_dim + self.action_dim,
                                             **vf_kwargs,
                                             device=self.device)
        self.target_value_function1 = ValueFunction(self.obs_dim + self.action_dim,
                                                    **vf_kwargs,
                                                    device=self.device)
        self.target_value_function2 = ValueFunction(self.obs_dim + self.action_dim,
                                                    **vf_kwargs,
                                                    device=self.device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(),
                                             **policy_optim_kwargs)
        self.value_function_optim = torch.optim.Adam(
            list(self.value_function1.parameters()) +
            list(self.value_function2.parameters()),
            **vf_optim_kwargs)

        self.policy_learning_rate_schedule = policy_learning_rate_schedule
        self.vf_learning_rate_schedule = vf_learning_rate_schedule

        self.buffer = CoupledBuffer(buffer_size, self.num_envs, self.obs_dim, False,
                                    self.action_dim, batch_size=batch_size,
                                    device=self.device)
        reset_observation, info = self.env.reset()
        self.buffer.reset(reset_observation)

        self.samples_per_episode = self.num_envs

    def _learn_episode(self, eps: int) -> tuple[float, float]:
        self.collect_trajectory(eps)

        policy_loss = None
        value_loss = None
        if eps > self.learning_starts:
            sample = self.buffer.batch()
            value_loss = self.update_value_function(sample)
            if eps % self.policy_frequency == 0:
                for _ in range(self.policy_frequency):
                    policy_loss = self.update_policy(sample.observations)
                    if self.autotune:
                        self.update_alpha(sample.observations)
            if eps % self.target_frequency == 0:
                self.update_target()

        return policy_loss, value_loss

    def collect_trajectory(self, eps: int):
        """
        Collect trajectories using random actions until learning starts, once the policy
        is sampled.

        Args:
            eps (int): The current episode number.
        """
        if eps < self.learning_starts:
            action = torch.from_numpy(self.env.action_space.sample()).to(self.device)
        else:
            with torch.no_grad():
                action = self.policy(self.buffer.observations[self.buffer.t])

        observation, reward, terminated, truncated, info = self.env.step(action)
        terminal = terminated | truncated

        self.buffer.add(observation, reward, terminal, action=action)

    def update_value_function(self, sample: CoupledBufferBatch) -> float:
        """
        Update the value function by the mse of the estimated next value and the
        predicted next value.

        Args:
            sample (CoupledBufferBatch): The batch of samples to train on.
        Returns:
            float: The value loss.
        """
        with torch.no_grad():
            next_actions = self.policy(sample.next_observations)
            next_log_probs = self.policy.log_prob(next_actions)
            next_target1 = self.target_value_function1(sample.next_observations,
                                                       next_actions).view(-1)
            next_target2 = self.target_value_function2(sample.next_observations,
                                                       next_actions).view(-1)
            min_next_target = torch.min(next_target1,
                                        next_target2) - self.alpha * next_log_probs

            next_value = sample.rewards + self.gamma * ~sample.next_terminals * min_next_target

        values_1 = self.value_function1(sample.observations, sample.actions).view(-1)
        values_2 = self.value_function2(sample.observations, sample.actions).view(-1)

        mse1 = torch.nn.functional.mse_loss(values_1, next_value)
        mse2 = torch.nn.functional.mse_loss(values_2, next_value)
        value_loss = mse1 + mse2

        self.value_function_optim.zero_grad()
        value_loss.backward()
        self.value_function_optim.step()

        return value_loss.item()

    def update_policy(self, observation: Float[
        Tensor, "{self.buffer.batch_size} self.obs_dim"]):
        """
        Update the policy using the entropy-regularized policy gradient.
        Args:
            observation (Float[Tensor, "{self.buffer.batch_size} self.obs_dim"]): The observations.

        Returns:
            float: The policy loss.
        """
        actions = self.policy(observation)
        log_probs = self.policy.log_prob(actions)

        value1 = self.value_function1(observation, actions).view(-1)
        value2 = self.value_function2(observation, actions).view(-1)
        min_value = torch.min(value1, value2)

        policy_loss = (self.alpha * log_probs - min_value).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return policy_loss

    def update_alpha(self, observation: Float[
        Tensor, "{self.buffer.batch_size} self.obs_dim"]):
        """
        Update the entropy regularization coefficient.

        Args:
            observation (Float[Tensor, "{self.buffer.batch_size} self.obs_dim"]): The observations.
        """
        with torch.no_grad():
            actions = self.policy(observation)
            log_probs = self.policy.log_prob(actions)
        alpha_loss = (- torch.log(self.alpha) * (log_probs + self.action_dim)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def update_target(self):
        """
        Polyak update the target value function.
        """
        for target_param, param in zip(self.target_value_function1.parameters(),
                                       self.value_function1.parameters()):
            target_param.data.mul_(self.polyak_target)
            target_param.data.add_((1 - self.polyak_target) * param.data)
        for target_param, param in zip(self.target_value_function2.parameters(),
                                       self.value_function2.parameters()):
            target_param.data.mul_(self.polyak_target)
            target_param.data.add_((1 - self.polyak_target) * param.data)
