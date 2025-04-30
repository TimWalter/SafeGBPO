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


class QuadrotorEnv(TorchVectorEnv):
    """
    ## Description

    Nonlinear longitudinal model of a quadrotor based on [1].

    [1] I. M. Mitchell et al. "Invariant, Viability and Discriminating
        Kernel Under-Approximation via Zonotope Scaling", 2019,
        Proceedings of the 22nd ACM International Conference on Hybrid
        Systems: Computation and Control, pp. 268-269

    ## Action Space

    The action is a `Tensor` with shape `(num_envs, 2)` indicating the total thrust and
    desired roll angle.

    | Num |    Action          |         Min         |         Max        |
    |-----|--------------------|---------------------|--------------------|
    | 0   | Total Thrust       | -1.5 + gravity/gain | 1.5 + gravity/gain |
    | 1   | Desired Roll Angle | -pi/12              | pi/12              |

    ## Observation & State Space

    The observation is a `Tensor` with shape `(num_envs, 6)` representing horizontal and
    vertical position as well as the respective velocities, as well as the roll and roll
    velocity. In addition, the balancing state is appended to the observation.

    | Num | Observation              | Min              | Max             |
    |-----|--------------------------|------------------|-----------------|
    | 0   | Horizontal Position      | -domain_width/2  | domain_width/2  |
    | 1   | Vertical Position        | -domain_height/2 | domain_height/2 |
    | 2   | Roll                     | -pi/12           | pi/12           |
    | 3   | Horizontal Velocity      | -0.8             | 0.8             |
    | 4   | Vertical Velocity        | -1.0             | 1.0             |
    | 5   | Roll Velocity            | -pi/2            | pi/2            |
    | 6   | Goal Horizontal Position | -domain_width/2  | domain_width/2  |
    | 7   | Goal Vertical Position   | -domain_height/2 | domain_height/2 |

    ## Starting State

    The starting state is centred in the domain.

    ## Episode Truncation

    The episode truncates at 1000 time steps.

    ## Arguments
    device: Literal["cpu", "cuda"] = "cpu" (device to use for torch tensors)
    num_envs: int = 1 (number of environments to run in parallel)
    stochastic: bool = True (whether to introduce stochastic dampening)
    max_episode_steps: int = 1000 (maximum number of steps in an episode)
    render_mode: Optional[str] = None (render mode for the environment)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    gravity: float = 9.81
    pd_gain0: float = 70
    pd_gain1: float = 17
    pd_gain2: float = 55

    thrust_mag = 6/7
    roll_angle_mag = torch.pi / 12

    dt: float = 0.05

    # Dim==state_shape[1] is required for the linearisation, such that the requirements
    # for a full dimensional box are given.
    noise: sets.Box = sets.Box(torch.zeros(1, 6),
                     torch.tensor([0.1, 0.1, 0.0, 0.0, 0.0, 0.0]) * torch.eye(
                         6).unsqueeze(0))

    screen_width = 500
    screen_height = 500

    def __init__(self,
                 device: Literal["cpu", "cuda"] = "cpu",
                 num_envs: int = 1,
                 stochastic: bool = True,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 1000,
                 domain_width: float = 16.0,
                 domain_height: float = 16.0,
                 additional_obs: np.ndarray = None
                 ):
        boundaries = np.array([
            [-domain_width / 2, domain_width / 2],
            [-domain_height / 2, domain_height / 2],
            [-np.pi / 12, np.pi / 12],
            [-0.8, 0.8],
            [-1.0, 1.0],
            [-np.pi / 2, np.pi / 2],
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
                             [-domain_height / 2, domain_height / 2],
                             [-np.pi / 12, np.pi / 12],
                             [-0.8, 0.8],
                             [-1.0, 1.0],
                             [-np.pi / 2, np.pi / 2],
                         ],
                         stochastic, render_mode)
        self.max_episode_steps = max_episode_steps
        self.goal = torch.empty((num_envs, 2), device=device, dtype=torch.float64)

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[
        Tensor, "{self.num_envs} {self.observation_space.shape[1]}"]:
        return torch.cat(
            [self.state, self.goal], dim=1)

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
        thrust, roll_angle = action.split(1, dim=1)
        thrust = thrust * self.thrust_mag + self.gravity
        roll_angle = roll_angle * self.roll_angle_mag

        x, z, roll, x_dot, z_dot, roll_dot = self.state.split(1, dim=1)

        x_ddot = thrust * torch.sin(roll)
        z_ddot = thrust * torch.cos(roll) - self.gravity
        roll_ddot = (roll_angle * self.pd_gain2
                     - self.pd_gain0 * roll
                     - self.pd_gain1 * roll_dot)

        if self.stochastic:
            noise = self.noise.sample()
            x_ddot = x_ddot + noise[:, 0:1]
            z_ddot = z_ddot + noise[:, 1:2]

        x_dot = x_dot + self.dt * x_ddot
        z_dot = z_dot + self.dt * z_ddot
        roll_dot = roll_dot + self.dt * roll_ddot
        x = x + self.dt * x_dot
        z = z + self.dt * z_dot
        roll = roll + self.dt * roll_dot

        roll = torch.atan2(torch.sin(roll), torch.cos(roll))  # more precise numerically
        return torch.cat([x, z, roll, x_dot, z_dot, roll_dot], dim=1)

    def reachable_set(self) -> sets.Zonotope:
        """
        Assuming linearised dynamics around 0 action.

        Returns:
            The one-step reachable set that is non-obstacles intersecting
        """
        center = self.dynamics(self.action_set.center)

        d_x_ddot_d_action0 = self.thrust_mag * torch.sin(self.state[:, 2])
        d_z_ddot_d_action0 = self.thrust_mag * torch.cos(self.state[:, 2])
        d_roll_ddot_d_action1 = self.roll_angle_mag * self.pd_gain2

        action_mat = torch.zeros((*self.state.shape, self.action_space.shape[1]),
                                 dtype=torch.float64, device=self.device)

        action_mat[:, 3, 0] += self.dt * d_x_ddot_d_action0
        action_mat[:, 4, 0] += self.dt * d_z_ddot_d_action0
        action_mat[:, 5, 1] += self.dt * d_roll_ddot_d_action1
        action_mat[:, 0, 0] += self.dt ** 2 * d_x_ddot_d_action0
        action_mat[:, 1, 0] += self.dt ** 2 * d_z_ddot_d_action0
        action_mat[:, 2, 1] += self.dt ** 2 * d_roll_ddot_d_action1

        generator = torch.bmm(action_mat, self.action_set.generator)
        return sets.Zonotope(center, generator)

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
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)

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
        thrust = lin_action[0] * self.thrust_mag + self.gravity
        roll_angle = lin_action[1] * self.roll_angle_mag

        x_ddot = thrust * torch.sin(lin_state[2]) \
                 + lin_noise[0] * self.stochastic
        z_ddot = thrust * torch.cos(lin_state[2]) - self.gravity \
                 + lin_noise[1] * self.stochastic
        roll_ddot = roll_angle * self.pd_gain2 \
                    - self.pd_gain0 * lin_state[2] \
                    - self.pd_gain1 * lin_state[5]

        d_x_ddot_d_roll = thrust * torch.cos(lin_state[2])
        d_x_ddot_d_action0 = self.thrust_mag * torch.sin(lin_state[2])
        d_x_ddot_d_noise = self.stochastic
        d_z_ddot_d_roll = -thrust * torch.sin(lin_state[2])
        d_z_ddot_d_action0 = self.thrust_mag * torch.cos(lin_state[2])
        d_z_ddot_d_noise = self.stochastic
        d_roll_ddot_d_roll = -self.pd_gain0
        d_roll_ddot_d_roll_dot = -self.pd_gain1
        d_roll_ddot_d_action1 = self.roll_angle_mag * self.pd_gain2

        constant_mat = torch.tensor([
            lin_state[0] + self.dt * (lin_state[3] + self.dt * x_ddot),
            lin_state[1] + self.dt * (lin_state[4] + self.dt * z_ddot),
            lin_state[2] + self.dt * (lin_state[5] + self.dt * roll_ddot),
            lin_state[3] + self.dt * x_ddot,
            lin_state[4] + self.dt * z_ddot,
            lin_state[5] + self.dt * roll_ddot
        ], dtype=torch.float64, device=self.device)

        state_mat = torch.eye(6, dtype=torch.float64, device=self.device)
        state_mat[0, 3] = state_mat[1, 4] = state_mat[2, 5] = self.dt
        state_mat[3, 2] += self.dt * d_x_ddot_d_roll
        state_mat[4, 2] += self.dt * d_z_ddot_d_roll
        state_mat[5, 2] += self.dt * d_roll_ddot_d_roll
        state_mat[5, 5] += self.dt * d_roll_ddot_d_roll_dot
        state_mat[0, 2] += self.dt ** 2 * d_x_ddot_d_roll
        state_mat[1, 2] += self.dt ** 2 * d_z_ddot_d_roll
        state_mat[2, 2] += self.dt ** 2 * d_roll_ddot_d_roll
        state_mat[2, 5] += self.dt ** 2 * d_roll_ddot_d_roll_dot

        action_mat = torch.zeros((6, 2), dtype=torch.float64, device=self.device)
        action_mat[3, 0] += self.dt * d_x_ddot_d_action0
        action_mat[4, 0] += self.dt * d_z_ddot_d_action0
        action_mat[5, 1] += self.dt * d_roll_ddot_d_action1
        action_mat[0, 0] += self.dt ** 2 * d_x_ddot_d_action0
        action_mat[1, 0] += self.dt ** 2 * d_z_ddot_d_action0
        action_mat[2, 1] += self.dt ** 2 * d_roll_ddot_d_action1

        noise_mat = torch.zeros((self.state.shape[1], self.noise.center.shape[1]),
                                dtype=torch.float64, device=self.device)
        noise_mat[3, 0] += self.dt * d_x_ddot_d_noise
        noise_mat[4, 1] += self.dt * d_z_ddot_d_noise
        noise_mat[0, 0] += self.dt ** 2 * d_x_ddot_d_noise
        noise_mat[1, 1] += self.dt ** 2 * d_z_ddot_d_noise

        return constant_mat, state_mat, action_mat, noise_mat

    def draw(self):
        world_width = np.ceil(
            self.single_observation_space.high[0] - self.single_observation_space.low[
                0]).item() + 1.0
        world_height = np.ceil(
            self.single_observation_space.high[1] - self.single_observation_space.low[
                1]).item() + 1.0
        scale_x = self.screen_width / world_width
        scale_y = self.screen_height / world_height

        quad_width = 30.0
        quad_height = 10.0

        if self.state is None:
            return None

        x, z, roll, _, _, _ = self.state[0, :].detach().cpu().numpy()

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

        # Convert quadrotor position to screen coordinates
        quad_x = int((x + world_width / 2) * scale_x)
        quad_y = int(self.screen_height - (z + world_height / 2) * scale_y)

        if np.abs(x) < world_width / 2 and np.abs(z) < world_height / 2:
            # Draw quadrotor body
            quad_points = []
            radius = np.sqrt((quad_height / 2) ** 2 + (quad_width / 2) ** 2)
            angle = np.arctan2(quad_height / 2, quad_width / 2)
            angles = [angle, -angle + np.pi, angle + np.pi, -angle]

            for angle_i in angles:
                x_offset = radius * np.cos(angle_i + roll)
                y_offset = radius * np.sin(angle_i + roll)
                quad_points.append((quad_x + x_offset, quad_y + y_offset))

            # Draw quadrotor
            gfxdraw.aapolygon(self.surf, quad_points, (0, 0, 0))
            gfxdraw.filled_polygon(self.surf, quad_points, (0, 0, 255))

            # Draw rotors
            rotor_radius = 5
            rotor_positions = [
                (quad_width / 2, quad_height / 2),
                (quad_width / 2, -quad_height / 2),
                (-quad_width / 2, quad_height / 2),
                (-quad_width / 2, -quad_height / 2)
            ]

            for rx, ry in rotor_positions:
                # Rotate rotor position by roll angle
                rotated_x = rx * np.cos(roll) - ry * np.sin(roll)
                rotated_y = rx * np.sin(roll) + ry * np.cos(roll)

                rotor_x = int(quad_x + rotated_x)
                rotor_y = int(quad_y + rotated_y)

                gfxdraw.aacircle(self.surf, rotor_x, rotor_y, rotor_radius, (255, 0, 0))
                gfxdraw.filled_circle(self.surf, rotor_x, rotor_y, rotor_radius,
                                      (255, 0, 0))

        # Draw the target position if balance_state exists
        if self.goal is not None:
            target_x = int((self.goal[0, 0].item() + world_width / 2) * scale_x)
            target_y = int(self.screen_height - (
                    self.goal[0, 1].item() + world_height / 2) * scale_y)
            gfxdraw.aacircle(self.surf, target_x, target_y, 5, (0, 255, 0))
            gfxdraw.filled_circle(self.surf, target_x, target_y, 5, (0, 255, 0))
