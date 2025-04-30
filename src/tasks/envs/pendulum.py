import os
from typing import Optional, Literal

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


class PendulumEnv(TorchVectorEnv):
    """
    ## Description

    The system consists of a pendulum attached at one end to a fixed point, and
    he other end being free.

    The diagram below specifies the coordinate system used for the implementation of the
    pendulum's dynamic equations.

    ![Pendulum Coordinate System](/src/assets/pendulum.png)

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Space

    The action is a `Tensor` with shape `(num_envs, 1)` which can take values `[-1,
    1]` indicating the torque and direction applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -1.0 | 1.0 |

    ## Observation Space

    The observation is a `Tensor` with shape `(num_envs, 3)` representing the x-y
    coordinates of the pendulum's free end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | sin(theta)       | -1.0 | 1.0 |
    | 1   | cos(theta)       | -1.0 | 1.0 |
    | 2   | Angular Velocity | -inf | inf |

    ## Arguments

    stochastic: bool = True (whether to introduce stochastic dampening)
    max_episode_steps: int = 200 (maximum number of steps in an episode)
    device: Literal["cpu", "cuda"] = "cpu" (device to use for torch tensors)
    render_mode: Optional[str] = None (render mode for the environment)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    gravity: float = 9.81
    length: float = 1
    mass: float = 1
    torque_mag: float = 30.0

    dt: float = 0.05

    # Dim==state_shape[1] is required for the linearisation, such that the requirements
    # for a full dimensional box are given.
    noise: sets.Box = sets.Box(torch.zeros(1, 2), torch.tensor([[[0.1, 0.0], [0.0, 0.0]]]))

    screen_width = 500
    screen_height = 500

    def __init__(self,
                 device: Literal["cpu", "cuda"] = "cpu",
                 num_envs: int = 1,
                 stochastic: bool = True,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 240,
                 ):
        high = np.array([1.0, 1.0, 1000], dtype=np.float64)
        action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        observation_space = spaces.Box(-high, high, dtype=np.float64)
        super().__init__(device, num_envs, observation_space, action_space,
                         [
                             [-np.pi, np.pi],
                             [-1000, 1000]
                         ],
                         stochastic, render_mode)

        self.max_episode_steps = max_episode_steps

        self.last_torque = None

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[
        Tensor, "{self.num_envs} {self.observation_space.shape[1]}"]:
        return torch.cat([
            torch.sin(self.state[:, 0:1]),
            torch.cos(self.state[:, 0:1]),
            self.state[:, 1:2]
        ], dim=1)

    @jaxtyped(typechecker=beartype)
    def dynamics(self,
                 action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]) \
            -> Float[Tensor, "{self.num_envs} {self.state.shape[1]}"]:
        theta, theta_dot = self.state.split(1, dim=1)

        torque = self.torque_mag * action
        self.last_torque = torque[0, 0].detach().cpu().numpy()  # for rendering

        gravitational_force = 1.5 * self.gravity / self.length * torch.sin(theta)
        driving_force = 3.0 / self.mass / self.length ** 2 * torque
        theta_ddot = gravitational_force + driving_force

        if self.stochastic:
            theta_ddot = theta_ddot + self.noise.sample()[:, 0]

        theta_dot = theta_dot + self.dt * theta_ddot
        theta = theta + self.dt * theta_dot

        theta = torch.atan2(torch.sin(theta), torch.cos(theta))
        return torch.cat([theta, theta_dot], dim=1)

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

    @jaxtyped(typechecker=beartype)
    def linear_dynamics(self,
                        lin_state: Float[Tensor, "{self.state.shape[1]}"],
                        lin_action: Float[Tensor, "{self.action_space.shape[1]}"],
                        lin_noise: Float[Tensor, "{self.noise.center.shape[1]}"]
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
        torque = self.torque_mag * lin_action[0]

        gravitational_force = 1.5 * self.gravity / self.length * torch.sin(lin_state[0])
        driving_force = 3.0 / self.mass / self.length ** 2 * torque

        theta_ddot = gravitational_force + driving_force
        if self.stochastic:
            theta_ddot = theta_ddot + lin_noise[0]

        d_gravitational_force_d_theta = 1.5 * self.gravity / self.length * torch.cos(
            lin_state[0])
        d_driving_force_d_action = 3.0 / self.mass / self.length ** 2 * self.torque_mag
        d_theta_ddot_d_theta = d_gravitational_force_d_theta
        d_theta_ddot_d_action = d_driving_force_d_action
        d_theta_ddot_d_noise = self.stochastic

        constant_mat = torch.tensor([
            lin_state[0] + self.dt * (lin_state[1] + self.dt * theta_ddot),
            lin_state[1] + self.dt * theta_ddot
        ], dtype=torch.float64, device=self.device)

        state_mat = torch.eye(2, dtype=torch.float64, device=self.device)
        state_mat[0, 1] = self.dt
        state_mat[1, 0] += self.dt * d_theta_ddot_d_theta
        state_mat[0, 0] += self.dt ** 2 * d_theta_ddot_d_theta

        action_mat = torch.zeros((2, 1), dtype=torch.float64, device=self.device)
        action_mat[1, 0] += self.dt * d_theta_ddot_d_action
        action_mat[0, 0] += self.dt ** 2 * d_theta_ddot_d_action

        noise_mat = torch.zeros((self.state.shape[1], self.noise.center.shape[1]),
                                dtype=torch.float64, device=self.device)
        noise_mat[1, 0] += self.dt * d_theta_ddot_d_noise
        noise_mat[0, 0] += self.dt ** 2 * d_theta_ddot_d_noise

        return constant_mat, state_mat, action_mat, noise_mat

    @jaxtyped(typechecker=beartype)
    def reachable_set(self) -> sets.Zonotope:
        """
        Compute the one step reachable set.

        Returns:
            The one step reachable set.
        """
        center = self.dynamics(self.action_set.center)

        d_driving_force_d_action = 3.0 / self.mass / self.length ** 2 * self.torque_mag
        d_theta_ddot_d_action = d_driving_force_d_action

        action_mat = torch.zeros((*self.state.shape, self.action_space.shape[1]),
                                 dtype=torch.float64, device=self.device)
        action_mat[:, 1, 0] += self.dt * d_theta_ddot_d_action
        action_mat[:, 0, 0] += self.dt ** 2 * d_theta_ddot_d_action

        generator = torch.bmm(action_mat, self.action_set.generator)
        return sets.Zonotope(center, torch.cat([generator], dim=2))

    def draw(self):
        bound = 2.2
        scale = self.screen_width / (bound * 2)
        offset = self.screen_width // 2
        pole_length = 1 * scale
        pole_width = 0.2 * scale

        if self.state is None:
            return None

        theta = self.state[0, 0].detach().cpu().numpy()

        l, r, t, b = 0, pole_length, pole_width / 2, -pole_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(theta + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(pole_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(pole_width / 2), (204, 77, 77)
        )

        pole_end = (pole_length, 0)
        pole_end = pygame.math.Vector2(pole_end).rotate_rad(theta + np.pi / 2)
        pole_end = (int(pole_end[0] + offset), int(pole_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, pole_end[0], pole_end[1], int(pole_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, pole_end[0], pole_end[1], int(pole_width / 2), (204, 77, 77)
        )

        img = pygame.image.load(
            os.path.join(os.path.dirname(__file__), "..", "..", "assets",
                         "clockwise.png"))
        if self.last_torque is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (
                    float(scale * np.abs(self.last_torque) / 2),
                    float(scale * np.abs(self.last_torque) / 2),
                ),
            )
            is_flip = bool(self.last_torque > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
