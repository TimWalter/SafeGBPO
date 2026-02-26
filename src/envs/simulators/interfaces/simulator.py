from abc import ABC, abstractmethod
from typing import Optional, Any

import torch
from torch import Tensor
from beartype import beartype
from gymnasium.vector import VectorEnv
from jaxtyping import Float, Bool, jaxtyped

import src.sets as sets

class Simulator(ABC, VectorEnv):
    """
    Base class for vectorized torch environments.
    """
    EVAL_ENVS: int = 1

    @jaxtyped(typechecker=beartype)
    def __init__(
            self,
            action_dim: int,
            state_set: sets.AxisAlignedBox,
            noise_set: sets.AxisAlignedBox,
            observation_set: sets.AxisAlignedBox,
            num_envs: int
    ):
        """Base class for vectorized environments.

        Args:
            action_dim: The dimension of the action space.
            state_set: Set of feasible states.
            noise_set: Set of noise that disturbs the dynamics.
            observation_set: Set of observations that can be made.
            num_envs: Number of environments to vectorize.
        """
        self.action_dim = action_dim
        self.state_dim = state_set.center.shape[1]
        self.obs_dim = observation_set.center.shape[1]
        self.num_envs = num_envs

        self.action_set = sets.AxisAlignedBox(
            torch.zeros(num_envs, action_dim),
            torch.diag_embed(torch.ones(num_envs, action_dim))
        )
        self.state_set = state_set
        self.noise_set = noise_set
        self.observation_set = observation_set

        self.state: Tensor = torch.empty((num_envs, self.state_dim))
        self.steps: int = 0
        self.scheduled_reset: Tensor = torch.ones(num_envs, dtype=torch.bool)

        self.dynamics = torch.vmap(self.unbatched_dynamics)

    @jaxtyped(typechecker=beartype)
    def reset(self, seed: Optional[int] = None) -> tuple[
        Float[Tensor, "{self.num_envs} {self.obs_dim}"],
        dict[str, Any]
    ]:
        """Reset all parallel environments and return a batch of initial observations
        and info.

        Args:
            seed: The environment reset seeds

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        if seed is not None:
            torch.manual_seed(seed)

        self.state = self.state_set.sample()

        self.steps = 0

        self.scheduled_reset = torch.zeros_like(self.scheduled_reset)

        return self.observation, {}

    @jaxtyped(typechecker=beartype)
    def step(self, action: Float[Tensor, "{self.num_envs} {self.action_dim}"]) -> tuple[
        Float[Tensor, "{self.num_envs} {self.obs_dim}"],
        Float[Tensor, "{self.num_envs}"],
        Bool[Tensor, "{self.num_envs}"],
        Bool[Tensor, "{self.num_envs}"],
        dict[str, Any]
    ]:
        """Execute an action for each parallel environment.

        Args:
            action: Batch of actions.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)

        Note:
            Resetting behaviour according to https://gymnasium.farama.org/gymnasium_release_notes/index.html#:~:text=To%20increase%20the%20efficiency,first%20observations%20of%20episodes.
        """
        self.execute_action(action)

        self.steps += 1
        if self.scheduled_reset.any():
            self.reset()
        terminated, truncated = self.episode_ending()
        self.scheduled_reset = terminated | truncated

        return self.observation, self.reward(action), terminated, truncated, {}

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[Tensor, "{self.num_envs} {self.obs_dim}"]:
        """
        Get the current observation of the environment.

        Returns:
            The current observation of the environment as a batch of observations.
        """
        return self.state

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def reward(self, action: Float[Tensor, "{self.num_envs} {self.action_dim}"]) \
            -> Float[Tensor, "{self.num_envs}"]:
        """
        Compute the reward for the given action.

        Args:
            action: Action executed in the environment.

        Returns:
            Reward.
        """
        pass

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def episode_ending(self) -> tuple[
        Bool[Tensor, "{self.num_envs}"],
        Bool[Tensor, "{self.num_envs}"],
    ]:
        """
        Check if the episode is ending for each environment.

        Returns:
            terminated: Whether the episode is terminated for each environment.
            truncated: Whether the episode is truncated for each environment.

        Notes:
            Termination
            refers to the episode ending after reaching a terminal state
            that is defined as part of the environment definition.
            Examples are - task success, task failure, robot falling down etc.
            Notably, this also includes episodes ending in finite-horizon environments
            due to a time-limit inherent to the environment. Note that to preserve
            Markov property, a representation of the remaining time must be present in
            the agent’s observation in finite-horizon environments

            Truncation
            refers to the episode ending after an externally defined condition (that is
            outside the scope of the Markov Decision Process). This could be a
            time-limit, a robot going out of bounds etc. An infinite-horizon environment
            is an obvious example of where this is needed. We cannot wait forever for
            the episode to complete, so we set a practical time-limit after which we
            forcibly halt the episode. The last state in this case is not a terminal
            state since it has a non-zero transition probability of moving to another
            state as per the Markov Decision Process that defines the RL problem. This
            is also different from time-limits in finite horizon environments as the
            agent in this case has no idea about this time-limit.
        """
        pass

    @jaxtyped(typechecker=beartype)
    def execute_action(self, action: Float[Tensor, "{self.num_envs} {self.action_dim}"]):
        """
        Execute the action in the environment by updating the state.

        Args:
            action: Action to execute in the environment.
        """
        self.state = self.dynamics(self.state, action, self.noise_set.sample())
        self.state = torch.clamp(self.state, self.state_set.min, self.state_set.max)

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def unbatched_dynamics(self,
                           state: Float[Tensor, "{self.state_dim}"],
                           action: Float[Tensor, "{self.action_dim}"],
                           noise: Float[Tensor, "{self.state_dim}"]) \
            -> Float[Tensor, "{self.state_dim}"]:
        """
        Unbatched dynamics function that computes the next state given the current state,
        action, and noise. We batch this function automatically using torch.vmap for vectorized execution.

        Args:
            state: Current state of the environment.
            action: Action to execute in the environment.
            noise: Noise sample to perturb the dynamics.

        Returns:
            Next state.
        """
        pass

    @jaxtyped(typechecker=beartype)
    def linear_dynamics(self) -> tuple[
        Float[Tensor, "{self.num_envs} {self.state_dim}"],
        Float[Tensor, "{self.num_envs} {self.state_dim} {self.state_dim}"],
        Float[Tensor, "{self.num_envs} {self.state_dim} {self.action_dim}"],
        Float[Tensor, "{self.num_envs} {self.state_dim} {self.state_dim}"]
    ]:
        """
        Compute the linearised dynamics around the current state, action set centre and noise set centre.

        Returns:
            constant_matrix: f(lin_state, lin_action, lin_noise)
            state_matrix: d_f(lin_state, lin_action, lin_noise)/ d_noise
            action_matrix: d_f(lin_state, lin_action, lin_noise)/ d_action
            noise_matrix: d_f(lin_state, lin_action, lin_noise)/ d_noise
        """
        lin_state, lin_action, lin_noise = self.state, self.action_set.center, self.noise_set.center

        constant_mat = self.dynamics(lin_state, lin_action, lin_noise)

        unbatched_linear_dynamics = torch.func.jacrev(self.unbatched_dynamics, argnums=(0, 1, 2))
        state_mat, action_mat, noise_mat = torch.vmap(unbatched_linear_dynamics)(
            lin_state, lin_action, lin_noise
        )

        return constant_mat, state_mat, action_mat, noise_mat

    @jaxtyped(typechecker=beartype)
    def reachable_set(self) -> sets.Zonotope:
        """
        Compute the one step reachable set for the linearised dynamics.

        Returns:
            The one step reachable set.
        """
        center, state_mat, action_mat, noise_mat = self.linear_dynamics()

        generator = torch.bmm(action_mat, self.action_set.generator)
        return sets.Zonotope(center, torch.cat([generator], dim=2))

    @jaxtyped(typechecker=beartype)
    def eval_reset(self) -> tuple[
        Float[Tensor, "{self.num_envs} {self.obs_dim}"],
        dict[str, Any]
    ]:
        """
        Reset all parallel environments and return a batch of initial observations
        and info for evaluation.

        Returns:
            A batch of observations and info from the vectorized environment.

        Note:
            If one does not reset the rng_state a deterministic evaluation (desired) will lead to a deterministic
            training (undesired).
        """
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            obs, info = self.reset(seed=42)
        return obs, info

    @jaxtyped(typechecker=beartype)
    def cut_computation_graph(self):
        """
        Cut the computation graph for the trajectory.
        """
        self.state = self.state.detach()

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def render(self) -> list[Tensor]:
        """
        Render all environments.

        Returns:
            A ist of rendered frames for each environment.
        """
        pass
