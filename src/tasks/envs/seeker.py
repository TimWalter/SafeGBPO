from typing import Optional, Literal, Any

import numpy as np
import pygame
import torch
from beartype import beartype
from gymnasium import spaces
from jaxtyping import jaxtyped, Float, Bool
from pygame import gfxdraw
from torch import Tensor

import src.sets as sets
from src.tasks.envs.interfaces.torch_vector_env import TorchVectorEnv


class SeekerEnv(TorchVectorEnv):
    """
    ## Description

    2D environment without dynamics

    ## Action Space

    The action is a `Tensor` with shape `(num_envs, 2)` indicating the velocity.

    | Num |   Action   |  Min | Max |
    |-----|------------|------|-----|
    | 1   | Velocity X | -1.0 | 1.0 |
    | 1   | Velocity Y | -1.0 | 1.0 |

    ## Observation & State Space

    The observation is a `Tensor` with shape `(num_envs, 4)` representing horizontal and
    vertical position. In addition, the goal state is appended to the observation.

    | Num | Observation              | Min              | Max             |
    |-----|--------------------------|------------------|-----------------|
    | 0   | Horizontal Position      | -domain_width/2  | domain_width/2  |
    | 1   | Vertical Position        | -domain_height/2 | domain_height/2 |
    | 2   | Goal Horizontal Position | -domain_width/2  | domain_width/2  |
    | 3   | Goal Vertical Position   | -domain_height/2 | domain_height/2 |
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    dt: float = 0.1

    noise: sets.Box = sets.Box(torch.zeros(1, 2),
                     torch.tensor([0.0, 0.0]) * torch.eye(2).unsqueeze(0))

    screen_width = 500
    screen_height = 500

    def __init__(self,
                 device: Literal["cpu", "cuda"] = "cpu",
                 num_envs: int = 1,
                 stochastic: bool = False,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 1000,
                 domain_width: float = 16,
                 domain_height: float = 16,
                 additional_obs: np.ndarray = None
                 ):
        boundaries = np.array([
            [-domain_width / 2, domain_width / 2],
            [-domain_height / 2, domain_height / 2],
            [-domain_width / 2, domain_width / 2],
            [-domain_height / 2, domain_height / 2]
        ], dtype=np.float64)
        if additional_obs is not None and additional_obs.any():
            boundaries = np.concatenate([boundaries, additional_obs], axis=0)
        action_space = spaces.Box(-np.ones(2, dtype=np.float64),
                                  np.ones(2, dtype=np.float64), dtype=np.float64)
        observation_space = spaces.Box(boundaries[:, 0], boundaries[:, 1],
                                       dtype=np.float64)
        super().__init__(device, num_envs, observation_space, action_space,
                         [
                             [-domain_width / 2, domain_width / 2],
                             [-domain_height / 2, domain_height / 2]
                         ],
                         stochastic, render_mode)
        self.max_episode_steps = max_episode_steps
        self.goal = torch.empty((num_envs, 2), device=device, dtype=torch.float64)

        self.screen_height = domain_height * 50
        self.screen_width = domain_width * 50

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[
        Tensor, "{self.num_envs} {self.observation_space.shape[1]}"]:
        return torch.cat([self.state, self.goal], dim=1)

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
        self.state = torch.zeros_like(self.state)
        self.goal = self.state[:, 0:2].clone()

        self.steps = torch.zeros_like(self.steps)

        self.scheduled_reset = torch.zeros_like(self.scheduled_reset)

        if self.render_mode == "human":
            self.render()

        return self.observation, {}

    @jaxtyped(typechecker=beartype)
    def dynamics(self,
                 action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]) \
            -> Float[Tensor, "{self.num_envs} {self.state.shape[1]}"]:
        return self.state + self.dt * action

    @jaxtyped(typechecker=beartype)
    def timestep(self,
                 action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]):
        """
        Perform a single timestep of the environment.

        Args:
            action: Action to take in the environment.
        """
        self.state = self.dynamics(action)

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
        return torch.zeros(self.num_envs, device=self.device)

    @jaxtyped(typechecker=beartype)
    def episode_ending(self) -> tuple[
        Bool[Tensor, "{self.num_envs}"],
        Bool[Tensor, "{self.num_envs}"],
    ]:
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        truncated = self.steps >= self.max_episode_steps
        return terminated, truncated

    def linear_dynamics(self,
                        lin_state: Float[Tensor, "{self.state.shape[1]}"],
                        lin_action: Float[Tensor, "{self.action_space.shape[1]}"],
                        lin_noise: Float[Tensor, "{self.noise.center.shape[0]"]
                        ) -> tuple[
        Float[Tensor, "{self.state.shape[1]}"],
        Float[Tensor, "{self.state.shape[1]} {self.state.shape[1]}"],
        Float[Tensor, "{self.state.shape[1]} {self.action_space.shape[1]}"],
        Float[Tensor, "{self.state.shape[1]} {self.noise.center.shape[1]}"]
    ]:
        """
        Compute the linearised dynamics around the given state, action and noise by
        computing a first order taylor series of the update.

        Args:
            lin_state: Linearisation point for the state.
            lin_action: Linearisation point for the action.
            lin_noise: Linearisation point for the noise.

        Returns:
            constant_matrix: The constant matrix in the linear dynamics.
            state_matrix: The state matrix in the linear dynamics.
            action_matrix: The action matrix in the linear dynamics.
            noise_matrix: The noise matrix in the linear dynamics.
        """
        constant_mat = torch.tensor([
            lin_state[0] + self.dt * lin_action[0],
            lin_state[1] + self.dt * lin_action[1]
        ], dtype=torch.float64, device=self.device)

        state_mat = torch.eye(2, dtype=torch.float64, device=self.device)

        action_mat = torch.eye(2, dtype=torch.float64, device=self.device) * self.dt

        noise_mat = torch.zeros((self.state.shape[1], self.noise.center.shape[1]),
                                dtype=torch.float64, device=self.device)

        return constant_mat, state_mat, action_mat, noise_mat

    def reachable_set(self) -> sets.Zonotope:
        """
        Assuming linearised dynamics around 0 action.

        Returns:
            The one-step reachable set that is non-obstacles intersecting
        """
        center = self.dynamics(self.action_set.center)
        action_mat = torch.diag_embed(
            torch.ones(*self.action_space.shape, dtype=torch.float64,
                       device=self.device)) * self.dt

        generator = torch.bmm(action_mat, self.action_set.generator)
        return sets.Zonotope(center, generator)

    def draw(self):
        world_width = np.ceil(
            self.single_observation_space.high[0] - self.single_observation_space.low[
                0]).item() + 1.0
        world_height = np.ceil(
            self.single_observation_space.high[1] - self.single_observation_space.low[
                1]).item() + 1.0
        scale_x = self.screen_width / world_width
        scale_y = self.screen_height / world_height

        radius = 5

        if self.state is None:
            return None

        x, y = self.state[0, :].detach().cpu().numpy()
        # Convert position to screen coordinates
        x = int((x + world_width / 2) * scale_x)
        y = int(self.screen_height - (y + world_height / 2) * scale_y)

        # Draw coordinate grid
        grid_spacing = 0.5  # meters
        for i in np.arange(-world_width / 2, world_width / 2 + grid_spacing,
                           grid_spacing):
            grid_x = int((i + world_width / 2) * scale_x)
            gfxdraw.vline(self.surf, grid_x, 0, self.screen_height, (200, 200, 200))
            # Add x-axis labels
            if abs(i) > 0.01:  # Don't label zero
                pygame.font.init()
                font = pygame.font.SysFont("calibri", 12)
                text_surface = font.render(f"{i:.1f}", True, (100, 100, 100))
                label_x = grid_x - text_surface.get_width() // 2
                label_y = self.screen_height - text_surface.get_height() - 5
                self.surf.blit(text_surface, (label_x, label_y))

        for i in np.arange(-world_height / 2, world_height / 2 + grid_spacing,
                           grid_spacing):
            grid_y = int(self.screen_height - (i + world_height / 2) * scale_y)
            gfxdraw.hline(self.surf, 0, self.screen_width, grid_y, (200, 200, 200))
            # Add y-axis labels
            if abs(i) > 0.01:  # Don't label zero
                pygame.font.init()
                font = pygame.font.SysFont("calibri", 12)
                text_surface = font.render(f"{i:.1f}", True, (100, 100, 100))
                self.surf.blit(text_surface,
                               (5, grid_y - text_surface.get_height() // 2))



        gfxdraw.aacircle(self.surf, x, y, radius, (255, 0, 0))
        gfxdraw.filled_circle(self.surf, x, y, radius,(255, 0, 0))

        # Draw the target position if balance_state exists
        if self.goal is not None:
            target_x = int((self.goal[0, 0].item() + world_width / 2) * scale_x)
            target_y = int(self.screen_height - (
                    self.goal[0, 1].item() + world_height / 2) * scale_y)
            gfxdraw.aacircle(self.surf, target_x, target_y, 5, (0, 255, 0))
            gfxdraw.filled_circle(self.surf, target_x, target_y, 5, (0, 255, 0))
