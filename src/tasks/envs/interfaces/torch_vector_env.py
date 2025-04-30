from abc import ABC, abstractmethod
from typing import Literal, Optional, Any

import gymnasium as gym
import numpy as np
import pygame
import torch
from beartype import beartype
from gymnasium.spaces import Box
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from jaxtyping import Float, Bool, jaxtyped
from torch import Tensor

import src.sets as sets


class TorchVectorEnv(ABC, VectorEnv):
    """
    Base class for vectorized torch environments to run multiple independent copies
    of the same environment in parallel. Moreover, a linearisation is
    expected and the stochasticity is supposed to be contained and bounded in a convex
    noise set.

    Attributes:
        device: Device to run the environment on.
        num_envs: Number of environments in the vectorized environment.
        observation_space: Observation space of a single environment.
        action_space: Action space of a single environment.
        state: Current state of the environment.
        steps: Number of steps taken in the environment.
        scheduled_reset: Whether the environment is scheduled to reset.
        _torch_random: Random number generator
        surf: The surf object for rendering the environment.
    """

    _torch_random: torch.random.Generator | None = None

    action_space: Box
    observation_space: Box
    single_action_space: Box
    single_observation_space: Box

    dt: float = None

    stochastic: bool = False
    noise: sets.Box = None

    eval_eps: int = 1

    screen_width: int
    screen_height: int

    def __init__(
            self,
            device: Literal["cpu", "cuda"],
            num_envs: int,
            observation_space: Box,
            action_space: Box,
            state_bounds: list,
            stochastic: bool,
            render_mode: Optional[str] = None
    ):
        """Base class for vectorized environments.

        Args:
            device: Device to run the environment on.
            num_envs: Number of environments in the vectorized environment.
            observation_space: Observation space of a single environment.
            action_space: Action space of a single environment.
            state_bounds: Bounds of the state space.
            stochastic: Whether the environment is stochastic.
            render_mode: Mode to render the environment in.
        """
        self.num_envs = num_envs
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.observation_space = batch_space(observation_space, num_envs)
        self.action_space = batch_space(action_space, num_envs)

        self.stochastic = stochastic

        self.device = device
        self.noise.center = self.noise.center.to(device, dtype=torch.float64)
        self.noise.generator = self.noise.generator.to(device, dtype=torch.float64)
        self.noise.device = self.device

        self.state_bounds = torch.tensor(state_bounds, device=device,
                                         dtype=torch.float64).unsqueeze(0).expand(
            self.num_envs, -1, -1)
        center = (self.state_bounds[..., 0] + self.state_bounds[..., 1]) / 2
        center[center.isnan()] = 0.0
        self.feasible_set = sets.Box(
            center,
            torch.diag_embed((self.state_bounds[..., 1] - self.state_bounds[..., 0]) / 2)
        )
        self.action_set = sets.Box(
            torch.zeros(*self.action_space.shape, device=device, dtype=torch.float64),
            torch.diag_embed(torch.ones(*self.action_space.shape, dtype=torch.float64,device=device))
        )

        self.state: Tensor = torch.empty((num_envs, len(state_bounds)), device=device, dtype=torch.float64)
        self.steps: Tensor = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.scheduled_reset: Tensor = torch.ones(num_envs, device=device, dtype=torch.bool)

        self.surf = None
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.is_open = True

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[
        Tensor, "{self.num_envs} {self.observation_space.shape[1]}"]:
        """
        If this is overwritten still the first observation should always be the state
        """
        return self.state

    @jaxtyped(typechecker=beartype)
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> tuple[
        Float[Tensor, "{self.num_envs} {self.observation_space.shape[1]}"],
        dict[str, Any]
    ]:
        """Reset all parallel environments and return a batch of initial observations
        and info.

        Args:
            seed: The environment reset seeds
            options: If to return the options

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)

        self.state = torch.zeros_like(self.state)

        self.steps = torch.zeros_like(self.steps)

        self.scheduled_reset = torch.zeros_like(self.scheduled_reset)

        if self.render_mode == "human":
            self.render()

        if seed is not None:
            torch.set_rng_state(rng_state)

        return self.observation, {}

    @jaxtyped(typechecker=beartype)
    def eval_reset(self, eps: int) -> tuple[
        Float[Tensor, "{self.num_envs} {self.observation_space.shape[1]}"],
        dict[str, Any]
    ]:
        return self.reset(seed=eps)

    @jaxtyped(typechecker=beartype)
    def step(self,
             action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]) \
            -> tuple[
                Float[Tensor, "{self.num_envs} {self.observation_space.shape[1]}"],
                Float[Tensor, "{self.num_envs}"],
                Bool[Tensor, "{self.num_envs}"],
                Bool[Tensor, "{self.num_envs}"],
                dict[str, Any]
            ]:
        """Take an action for each parallel environment.

        Args:
            action: element of :attr:`action_space` Batch of actions.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)

        Note:
            As the vector environments auto-reset for a terminating and truncating
            sub-environments, the returned observation and info is not the final steps
            observation or info which is instead stored in info as `"final_observation"`
            and `"final_info"`.
        """
        if self.num_envs == 1:
            assert self.action_space.contains(
                action.detach().cpu().numpy()
            ), f"{action.detach().cpu().numpy()!r} ({type(action.detach().cpu().numpy())}) invalid"
        assert self.state is not None, "Call reset before using step method."

        self.timestep(action)
        # Enforce feasibility of the state
        self.state = torch.clamp(self.state, self.state_bounds[..., 0],
                                 self.state_bounds[..., 1])

        self.steps += 1

        if self.scheduled_reset.any():
            self.reset()

        terminated, truncated = self.episode_ending()

        self.scheduled_reset = terminated | truncated

        if self.render_mode == "human":
            self.render()

        return self.observation, self.reward(action), terminated, truncated, {}

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def dynamics(self,
                 action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]) \
            -> Float[Tensor, "{self.num_envs} {self.state.shape[1]}"]:
        pass

    @jaxtyped(typechecker=beartype)
    def timestep(self,
                 action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]):
        """
        Perform a single timestep of the environment.

        Args:
            action: Action to take in the environment.
        """
        self.state = self.dynamics(action)

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def reward(self,
               action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]) \
            -> Float[Tensor, "{self.num_envs}"]:
        """
        Compute the reward for the given action.

        Args:
            action: Action taken in the environment.

        Returns:
            Reward for the given action.
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

    @abstractmethod
    def linear_dynamics(self,
                        lin_state: Float[Tensor, "{self.state.shape[1]}"],
                        lin_action: Float[Tensor, "{self.action_space.shape[1]}"],
                        lin_noise: Float[Tensor, "{self.noise.center.shape[1]"]
                        ) -> tuple[
        Float[Tensor, "{self.state.shape[1]}"],
        Float[Tensor, "{self.state.shape[1]} {self.state.shape[1]}"],
        Float[Tensor, "{self.state.shape[1]} {self.action_space.shape[1]}"],
        Float[Tensor, "{self.state.shape[1]} {self.noise.center.shape[1]}"]
    ]:
        """
        Compute the linearised dynamics around the given state, action and noise.
        f(state, action, noise) = constant_matrix + state_matrix @ state +
        action_matrix @ action + noise_matrix @ noise.

        Args:
            lin_state: State around which the dynamics are linearised.
            lin_action: Action around which the dynamics are linearised.
            lin_noise: Noise around which the dynamics are linearised.

        Returns:
            constant_matrix: f(lin_state, lin_action, lin_noise)
            state_matrix: d_f(lin_state, lin_action, lin_noise)/ d_noise
            action_matrix: d_f(lin_state, lin_action, lin_noise)/ d_action
            noise_matrix: d_f(lin_state, lin_action, lin_noise)/ d_noise
        """
        pass

    @abstractmethod
    def reachable_set(self) -> sets.Zonotope:
        """
        Compute the one step reachable set.

        Returns:
            The one step reachable set.
        """
        pass

    def clear_computation_graph(self):
        """Clear the computation graph for the environment."""
        self.state = self.state.detach()

    @jaxtyped(typechecker=beartype)
    def _rand(self, size: int) -> Float[Tensor, "{self.num_envs} {size}"]:
        return torch.rand([self.num_envs, size],
                          generator=self._torch_random,
                          dtype=torch.float64,
                          device=self.device)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.screen is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        self.draw()

        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    @abstractmethod
    def draw(self):
        """
        Actual drawing in pygame
        """
        pass

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.is_open = False
        self.closed = True
