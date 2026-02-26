import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from learning_algorithms.interfaces.learning_algorithm import LearningAlgorithm
from learning_algorithms.components.coupled_buffer import CoupledBuffer, CoupledBufferBatch
from learning_algorithms.components.value_function import ValueFunction
from envs.simulators.interfaces.simulator import Simulator


class SAC(LearningAlgorithm):
    """
    Soft Actor-Critic (SAC) Algorithm. https://arxiv.org/pdf/1812.05905

    Constants:
        GAMMA: Discount factor for future rewards.
        BATCH_SIZE: The batch size to sample from the buffer.
        ALPHA: The entropy regularization coefficient.
    """

    GAMMA = 0.99
    BATCH_SIZE = 256
    ALPHA = 0.2

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 env: Simulator,
                 policy_kwargs: dict,
                 policy_optim_kwargs: dict,
                 vf_kwargs: dict,
                 vf_optim_kwargs: dict,
                 regularisation_coefficient: float,
                 buffer_size: int,
                 polyak_target: float,
                 learning_starts: int,
                 policy_frequency: int,
                 target_frequency: int
                 ):
        """
        Initialize the SAC algorithm.

        Args:
            env: The environment to train on.
            policy_kwargs: The keyword arguments for the policy network.
            policy_optim_kwargs: The keyword arguments for the policy optimizer.
            vf_kwargs: The keyword arguments for the value function network.
            vf_optim_kwargs: The keyword arguments for the value function optimizer.
            regularisation_coefficient: Regularisation coefficient for the regularisation towards safe actions.
            buffer_size: The maximal size of the buffer.
            polyak_target: The soft update coefficient for the target function.
            learning_starts: The number of episodes to collect before training.
            policy_frequency: The frequency and number of policy updates.
            target_frequency: The frequency to update the target value function.
        """
        super().__init__(env, policy_kwargs, policy_optim_kwargs, vf_kwargs, vf_optim_kwargs,
                         regularisation_coefficient, True)
        self.buffer = CoupledBuffer(buffer_size, self.env.num_envs, self.env.obs_dim, False,
                                    self.env.action_dim, batch_size=self.BATCH_SIZE,
                                    store_safe_actions=hasattr(self.env, "safe_action"))
        self.polyak_target = polyak_target
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.target_frequency = target_frequency

        self.value_function2 = ValueFunction(self.env.obs_dim + self.env.action_dim, **vf_kwargs)
        self.value_function_optim = torch.optim.Adam(list(self.value_function.parameters()) +
                                                     list(self.value_function2.parameters()),
                                                     **vf_optim_kwargs)

        self.target_value_function1 = ValueFunction(self.env.obs_dim + self.env.action_dim, **vf_kwargs)
        self.target_value_function2 = ValueFunction(self.env.obs_dim + self.env.action_dim, **vf_kwargs)

        self.alpha = torch.tensor([1.], requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.alpha], **vf_optim_kwargs)

        reset_observation, info = self.env.reset()
        self.buffer.reset(reset_observation)

        self.interactions_per_episode = self.env.num_envs

    @jaxtyped(typechecker=beartype)
    def _learn_episode(self, eps: int) -> tuple[float, float, float]:
        """
        Learn a single episode of the policy and value function.

        Args:
            eps: The index of the current learning episode.

        Returns:
            Average reward, policy loss, and value loss for the episode.
        """
        average_reward = self.interact(eps)

        policy_loss = 0.0
        value_loss = 0.0
        if eps > self.learning_starts:
            sample = self.buffer.batch()
            value_loss = self.update_value_function(sample)
            if eps % self.policy_frequency == 0:
                for _ in range(self.policy_frequency):
                    policy_loss = self.update_policy(sample.observations)
                    self.update_alpha(sample.observations)
            if eps % self.target_frequency == 0:
                self.update_target()

        self._last_episode_additional_metrics = self.buffer.aggregate_additional_metrics()
        return average_reward, policy_loss, value_loss

    @jaxtyped(typechecker=beartype)
    def interact(self, eps: int) -> float:
        """
        Interact with the environment using random actions until learning starts, afterward the policy is sampled.

        Args:
            eps: The current episode number.

        Returns:
            float: The average reward collected during the episode.
        """
        if eps < self.learning_starts:
            action = self.env.action_set.sample()
        else:
            with torch.no_grad():
                action = self.policy(self.buffer.observations[self.buffer.t])

        observation, reward, terminated, truncated, info = self.env.step(action)
        terminal = terminated | truncated

        safe_action = self.env.safe_action if hasattr(self.env, "safe_action") else None
        safeguard_metrics  = self.env.safeguard_metrics()  if hasattr(self.env, "safeguard_metrics") else None
        self.buffer.add(observation, reward, terminal, action=action, safe_action=safe_action, additional_metrics=safeguard_metrics)

        return reward.mean().item()

    @jaxtyped(typechecker=beartype)
    def update_value_function(self, sample: CoupledBufferBatch) -> float:
        """
        Update the value function by the mse of the estimated next value and the
        predicted next value.

        Args:
            sample: The batch of samples to train on.
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

            next_value = sample.rewards + self.GAMMA * ~sample.next_terminals * min_next_target

        values_1 = self.value_function(sample.observations, sample.actions).view(-1)
        values_2 = self.value_function2(sample.observations, sample.actions).view(-1)

        mse1 = torch.nn.functional.mse_loss(values_1, next_value)
        mse2 = torch.nn.functional.mse_loss(values_2, next_value)
        value_loss = mse1 + mse2

        self.value_function_optim.zero_grad()
        value_loss.backward()
        self.value_function_optim.step()

        return value_loss.item()

    @jaxtyped(typechecker=beartype)
    def update_policy(self, observation: Float[Tensor, "{self.buffer.batch_size} {self.env.obs_dim}"]) -> float:
        """
        Update the policy using the entropy-regularized policy gradient.
        Args:
            observation: The observations.

        Returns:
            The policy loss.
        """
        actions = self.policy(observation)
        log_probs = self.policy.log_prob(actions)

        value1 = self.value_function(observation, actions).view(-1)
        value2 = self.value_function2(observation, actions).view(-1)
        min_value = torch.min(value1, value2)

        policy_loss = (self.alpha * log_probs - min_value).mean()
        
        if self.buffer.store_safe_actions:
            policy_loss += self.env.regularisation(self.buffer.actions.tensor, 
                                                   self.buffer.safe_actions.tensor,
                                                   safeguard_metrics= self.buffer.additional_metrics)
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return policy_loss.item()

    @jaxtyped(typechecker=beartype)
    def update_alpha(self, observation: Float[Tensor, "{self.buffer.batch_size} {self.env.obs_dim}"]):
        """
        Update the entropy regularization coefficient.

        Args:
            observation: The observations.
        """
        with torch.no_grad():
            actions = self.policy(observation)
            log_probs = self.policy.log_prob(actions)
        alpha_loss = (- torch.log(self.alpha) * (log_probs + self.env.action_dim)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    @jaxtyped(typechecker=beartype)
    def update_target(self):
        """
        Polyak update the target value function.
        """
        for target_param, param in zip(self.target_value_function1.parameters(), self.value_function.parameters()):
            target_param.data.mul_(self.polyak_target)
            target_param.data.add_((1 - self.polyak_target) * param.data)
        for target_param, param in zip(self.target_value_function2.parameters(), self.value_function2.parameters()):
            target_param.data.mul_(self.polyak_target)
            target_param.data.add_((1 - self.polyak_target) * param.data)
