"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

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
from tasks.envs.interfaces.torch_vector_env import TorchVectorEnv


class CartPoleEnv(TorchVectorEnv):
    """
    ## Description

    A pole is attached by an un-actuated joint to a cart, which moves along a
    frictionless track.

    ## Action Space

    The action is a `Tensor` with shape `(num_envs, 1)` which can take values `[-1,
    1]` indicating the force and direction the cart is pushed in.

    - -1: Push cart to the left with maximal force
    - 1: Push cart to the right with maximal force

    **Note**: The velocity that is reduced or increased by the applied force is not
    fixed, and it depends on the angle the pole is pointing. The center of gravity of
    the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `Tensor` with shape `(num_envs, 5)` with the values
    corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -10                 | 10                |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Sin Pole Angle        | -1                  | 1                 |
    | 2   | Cos Pole Angle        | -1                  | 1                 |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |


    ## Starting State All observations are assigned to zero.

    ## Episode Truncation

    The episode truncates at 240 time steps.

    ## Arguments

    max_episode_steps: int = 240 (maximum steps per episode)
    device: Literal["cpu", "cuda"] = "cpu" (device to use for torch tensors)
    render_mode: Optional[str] = None (mode to render the environment)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    g: float = 9.8  # gravity
    m_c: float = 1.0  # mass cart
    m_p: float = 0.1  # mass pole
    m_t: float = m_p + m_c  # mass total
    l: float = 0.5  # actually half the pole's length
    force_mag: float = 100.0
    dt: float = 0.015

    # Dim==state_shape[1] is required for the linearisation, such that the requirements
    # for a full dimensional box are given.
    noise: sets.Box = sets.Box(torch.zeros(1, 4), torch.zeros(1, 4, 4))

    screen_width: int = 1200
    screen_height: int = 200

    def __init__(
            self,
            device: Literal["cpu", "cuda"] = "cpu",
            num_envs: int = 1,
            stochastic: bool = False,
            render_mode: Optional[str] = None,
            max_episode_steps: int = 240,
    ):
        high = np.array(
            [
                10,
                1.0,
                1.0,
                1000,
                1000,
            ],
            dtype=np.float64,
        )
        action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        observation_space = spaces.Box(-high, high, dtype=np.float64)
        super().__init__(device, num_envs, observation_space, action_space,
                         [
                             [-10, 10],
                             [-np.pi, np.pi],
                             [-1000, 1000],
                             [-1000, 1000],
                         ], stochastic,
                         render_mode)
        self.max_episode_steps = max_episode_steps

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[
        Tensor, "{self.num_envs} {self.observation_space.shape[1]}"]:
        return torch.cat([
            self.state[:, 0:1],
            torch.sin(self.state[:, 1:2]),
            torch.cos(self.state[:, 1:2]),
            self.state[:, 2:4]
        ], dim=1)

    @jaxtyped(typechecker=beartype)
    @jaxtyped(typechecker=beartype)
    def dynamics(self,
                 action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]) \
            -> Float[Tensor, "{self.num_envs} {self.state.shape[1]}"]:

        x, theta, x_dot, theta_dot = self.state.split(1, dim=1)

        force = self.force_mag * action

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = 1 / self.m_t * (
            (force + self.m_p * self.l * torch.square(theta_dot) * sin_theta)
        )
        theta_ddot = 1 / self.l * (
                (self.g * sin_theta - temp * cos_theta) /
                (4.0 / 3.0 - self.m_p / self.m_t * torch.square(cos_theta))
        )
        x_ddot = temp - self.m_p * self.l / self.m_t * cos_theta * theta_ddot

        x_dot = x_dot + self.dt * x_ddot
        theta_dot = theta_dot + self.dt * theta_ddot
        x = x + self.dt * x_dot
        theta = theta + self.dt * theta_dot

        theta = torch.atan2(torch.sin(theta), torch.cos(theta))
        return torch.cat([x, theta, x_dot, theta_dot], dim=1)

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
        force = self.force_mag * lin_action[0]

        sin = torch.sin(lin_state[1])
        cos = torch.cos(lin_state[1])

        temp = 1 / self.m_t * (
            (force + self.m_p * self.l * torch.square(lin_state[3]) * sin)
        )
        nominator = self.g * sin - temp * cos
        denominator = (4.0 / 3.0 - self.m_p / self.m_t * torch.square(cos))
        theta_ddot = 1 / self.l * (nominator / denominator)
        x_ddot = temp - self.m_p * self.l / self.m_t * cos * theta_ddot

        d_temp_d_theta = self.m_p * self.l * torch.square(lin_state[3]) * cos / self.m_t
        d_temp_d_theta_dot = 2 * self.m_p * self.l * lin_state[3] * sin / self.m_t
        d_temp_d_action = self.force_mag / self.m_t
        d_nominator_d_theta = self.g * cos - d_temp_d_theta * cos + temp * sin
        d_nominator_d_theta_dot = -d_temp_d_theta_dot * cos
        d_nominator_d_action = -d_temp_d_action * cos
        d_denominator_d_theta = self.m_p / self.m_t * 2 * cos * sin
        d_theta_ddot_d_theta = 1 / self.l * (
                (
                        d_nominator_d_theta * denominator - nominator * d_denominator_d_theta) /
                torch.square(denominator)
        )
        d_theta_ddot_d_theta_dot = 1 / self.l * d_nominator_d_theta_dot / denominator
        d_theta_ddot_d_action = 1 / self.l * d_nominator_d_action / denominator
        d_x_ddot_d_theta = d_temp_d_theta + self.m_p * self.l / self.m_t * (
                sin * theta_ddot - cos * d_theta_ddot_d_theta)
        d_x_ddot_d_theta_dot = d_temp_d_theta_dot - self.m_p * self.l / self.m_t * cos * d_theta_ddot_d_theta_dot
        d_x_ddot_d_action = d_temp_d_action - self.m_p * self.l / self.m_t * cos * d_theta_ddot_d_action

        constant_mat = torch.tensor([
            lin_state[0] + self.dt * (lin_state[2] + self.dt * x_ddot),
            lin_state[1] + self.dt * (lin_state[3] + self.dt * theta_ddot),
            lin_state[2] + self.dt * x_ddot,
            lin_state[3] + self.dt * theta_ddot
        ], dtype=torch.float64, device=self.device)

        state_mat = torch.eye(4, dtype=torch.float64, device=self.device)
        state_mat[0, 2] = state_mat[1, 3] = self.dt
        state_mat[2, 1] += self.dt * d_x_ddot_d_theta
        state_mat[2, 3] += self.dt * d_x_ddot_d_theta_dot
        state_mat[3, 1] += self.dt * d_theta_ddot_d_theta
        state_mat[3, 3] += self.dt * d_theta_ddot_d_theta_dot
        state_mat[0, 1] += self.dt ** 2 * d_x_ddot_d_theta
        state_mat[0, 3] += self.dt ** 2 * d_x_ddot_d_theta_dot
        state_mat[1, 1] += self.dt ** 2 * d_theta_ddot_d_theta
        state_mat[1, 3] += self.dt ** 2 * d_theta_ddot_d_theta_dot

        action_mat = torch.zeros((4, 1), dtype=torch.float64, device=self.device)
        action_mat[2, 0] += self.dt * d_x_ddot_d_action
        action_mat[3, 0] += self.dt * d_theta_ddot_d_action
        action_mat[0, 0] += self.dt ** 2 * d_x_ddot_d_action
        action_mat[1, 0] += self.dt ** 2 * d_theta_ddot_d_action

        noise_mat = torch.zeros((self.state.shape[1], self.noise.center.shape[1]),
                                dtype=torch.float64, device=self.device)
        return constant_mat, state_mat, action_mat, noise_mat

    def reachable_set(self) -> sets.Zonotope:
        """
        Compute the one step reachable set.

        Returns:
            The one step reachable set.
        """
        center = self.dynamics(self.action_set.center)

        cos = torch.cos(self.state[:, 1])
        denominator = (4.0 / 3.0 - self.m_p / self.m_t * torch.square(cos))
        d_temp_d_action = self.force_mag / self.m_t
        d_nominator_d_action = -d_temp_d_action * cos
        d_theta_ddot_d_action = 1 / self.l * d_nominator_d_action / denominator
        d_x_ddot_d_action = d_temp_d_action - self.m_p * self.l / self.m_t * cos * d_theta_ddot_d_action


        action_mat = torch.zeros((*self.state.shape, self.action_space.shape[1]),
                                 dtype=torch.float64, device=self.device)
        action_mat[:, 2, 0] += self.dt * d_x_ddot_d_action
        action_mat[:, 3, 0] += self.dt * d_theta_ddot_d_action
        action_mat[:, 0, 0] += self.dt ** 2 * d_x_ddot_d_action
        action_mat[:, 1, 0] += self.dt ** 2 * d_theta_ddot_d_action

        generator = torch.bmm(action_mat, self.action_set.generator)
        return sets.Zonotope(center, torch.cat([generator], dim=2))


    def draw(self):
        world_width = 20
        scale = self.screen_width / world_width
        pole_width = 10.0
        pole_length = scale * (2 * self.l)
        cart_width = 50.0
        cart_height = 30.0

        if self.state is None:
            return None

        x = self.state[0, :].detach().cpu().numpy()

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cart_width / 2, cart_width / 2, cart_height / 2, -cart_height / 2
        offset = cart_height / 4.0

        cart_x = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        cart_y = self.screen_height // 2  # CENTER OF SCREEN

        # Draw x-axis
        axis_y = int(cart_y - cart_height // 2 - 10)
        gfxdraw.hline(self.surf, 0, self.screen_width, axis_y, (0, 0, 0))

        # Draw ticks on x-axis
        num_ticks = 21  # Odd number to have a center tick
        tick_spacing = self.screen_width // (num_ticks - 1)
        for i in range(num_ticks):
            tick_x = i * tick_spacing
            tick_height = 5 if i == num_ticks // 2 else 3  # Larger tick in the center
            gfxdraw.vline(self.surf, tick_x, axis_y, axis_y - tick_height, (0, 0, 0))

            # Add tick labels
            if 0 < i < num_ticks - 1:
                label = f"{i - num_ticks // 2}"
                font = pygame.font.SysFont("calibri", 15)
                text_surface = font.render(label, True, (0, 0, 0))
                text_surface = pygame.transform.flip(text_surface, False, True)
                self.surf.blit(text_surface, (
                    tick_x - text_surface.get_width() // 2, axis_y - tick_height - 20))

        if np.abs(x[0]) < world_width // 2:
            # Draw cart
            cart_coords = [(l, b), (l, t), (r, t), (r, b)]
            cart_coords = [(c[0] + cart_x, c[1] + cart_y) for c in cart_coords]
            gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
            gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

            # Draw pole
            l, r, t, b = (
                -pole_width / 2,
                pole_width / 2,
                pole_length - pole_width / 2,
                -pole_width / 2,
            )
            pole_coords = []
            for coord in [(l, b), (l, t), (r, t), (r, b)]:
                coord = pygame.math.Vector2(coord).rotate_rad(-x[1])
                coord = (coord[0] + cart_x, coord[1] + cart_y + offset)
                pole_coords.append(coord)
            gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
            gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

            # Draw axle
            gfxdraw.aacircle(
                self.surf,
                int(cart_x),
                int(cart_y + offset),
                int(pole_width / 2),
                (129, 132, 203),
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cart_x),
                int(cart_y + offset),
                int(pole_width / 2),
                (129, 132, 203),
            )

            # Draw black stripe indicating cart's x position
            gfxdraw.vline(self.surf, int(cart_x), axis_y,
                          int(cart_y - cart_height // 2),
                          (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
