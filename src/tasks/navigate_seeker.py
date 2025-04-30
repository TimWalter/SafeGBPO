from typing import Literal, Optional, Any

import cvxpy as cp
import numpy as np
import pygame
import torch
from beartype import beartype
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import jaxtyped, Float, Bool
from pygame import gfxdraw
from torch import Tensor

import src.sets as sets
from src.algorithms.components.coupled_tensor import CoupledTensor
from tasks.envs.seeker import SeekerEnv
from tasks.interfaces.safe_action_task import SafeActionTask


class NavigateSeekerTask(SeekerEnv, SafeActionTask):
    """
    ## The reward describes a navigation task, where a goal has to be reached,
    without colliding with obstacles. The distance to the goal is punished and
    collisions are punished, reaching the goal is rewarded.

    ## Starting State

    The starting state and goal positions are sampled uniformly from the domain.

    ## Episode Truncation

    The episode truncates at 400 time steps.
    """

    collision_penalty: float = 10.0
    distance_penalty: float = 1.0
    success_reward: float = 100.0

    last_safe_state_set: sets.Zonotope | None = None

    eval_eps = 6

    def __init__(self,
                 device: Literal["cpu", "cuda"] = "cpu",
                 num_envs: int = 1,
                 stochastic: bool = False,
                 max_episode_steps: int = 400,
                 render_mode: Optional[str] = None,
                 num_obstacles: int = 3,
                 domain_width: int = 16,
                 domain_height: int = 16,
                 min_radius: float = 1.0,
                 max_radius: float = 3.0,
                 draw_safe_state_set: bool = False
                 ):
        additional_obs = np.array(
            [
                [-domain_width / 2, domain_width / 2],  # Obstacle center x
                [-domain_height / 2, domain_height / 2],  # Obstacle center z
                [min_radius, max_radius],  # Obstacle radius
            ] * num_obstacles, dtype=np.float64)
        SeekerEnv.__init__(self, device, num_envs, False, render_mode,
                           max_episode_steps, domain_width, domain_height,
                           additional_obs)
        self.num_obstacles = num_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.draw_safe_state_set = draw_safe_state_set

        self.obstacles: list[sets.Ball] = []
        self.obstacle_centers = CoupledTensor((num_obstacles, self.num_envs, 2),
                                              torch.float64, self.device)
        self.obstacle_radii = CoupledTensor((num_obstacles, self.num_envs),
                                            torch.float64, self.device)

        self.collided = torch.zeros((self.num_envs, num_obstacles),
                                    device=self.device, dtype=torch.bool)
        self.initial_goal_distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self.reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.calculate_generator = self.construct_generator_function()
        SafeActionTask.__init__(self, device, 2)

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[
        Tensor, "{self.num_envs} {self.observation_space.shape[1]}"]:
        obs = torch.cat([self.state, self.goal], dim=1)
        for i in range(self.num_obstacles):
            obs = torch.cat(
                [
                    obs,
                    self.obstacles[i].center,
                    #self.obstacles[i].center - self.state,
                    self.obstacles[i].radius.unsqueeze(1)
                    #torch.norm(self.state - self.obstacles[i].center, dim=1, keepdim=True) - self.obstacles[i].radius.unsqueeze(1)
                ],
                dim=1)
        return obs

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
        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
        self.state = self.feasible_set.sample()
        too_close = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        while too_close.any():
            self.goal[too_close] = self.feasible_set.sample()[too_close]
            too_close[:] = torch.norm(self.goal - self.state, dim=1) < self.max_radius * 2
        self.sample_obstacles()

        self.steps = torch.zeros_like(self.steps)
        self.scheduled_reset = torch.zeros_like(self.scheduled_reset)
        self.collided = torch.zeros_like(self.collided)
        self.reached = torch.zeros_like(self.reached)
        self.initial_goal_distance = torch.norm(self.goal - self.state, dim=1)

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
        if self.num_obstacles == 1:
            self.reset()
            configs = [
                [(-4.0, 0.0), (4.0, 0.0), (0.0, 0.0), self.min_radius],
                [(4.0, 0.0), (-4.0, 0.0), (0.0, 0.0), self.min_radius],
                [(0.0, 4.0), (0.0, -4.0), (0.0, 0.0), self.min_radius],
                [(0.0, -4.0), (0.0, 4.0), (0.0, 0.0), self.min_radius],
                [(-4.0, 0.0), (4.0, 1.0), (0.0, 0.0), self.min_radius],
                [(-4.0, 0.0), (4.0, 0.0), (1.0, 1.0), 1.0]
            ]
            self.state[:, 0], self.state[:, 1] = configs[eps][0]
            self.goal[:, 0], self.goal[:, 1] = configs[eps][1]
            self.obstacles[0].center[:, 0], self.obstacles[0].center[:, 1] = configs[eps][2]
            self.obstacles[0].radius[:] = configs[eps][3]
            self.obstacle_centers[0] = self.obstacles[0].center
            self.obstacle_radii[0] = self.obstacles[0].radius
        else:
            self.reset(seed=eps)

        return self.observation, {}

    def sample_obstacles(self):
        self.obstacles = []
        if self.num_obstacles > 0:
            ray = (self.goal - self.state) / torch.norm(self.goal - self.state, dim=1, keepdim=True)
            normal_ray = torch.zeros_like(ray)
            normal_ray[:, 0] = ray[:, 1]
            normal_ray[:, 1] = -ray[:, 0]
            obs = sets.Ball(
                (self.state + self.goal) / 2 + self._rand(1) * normal_ray * self.min_radius / 2,
                self._rand(1).squeeze(dim=1) * (self.max_radius - self.min_radius) + self.min_radius
            )
            self.obstacles.append(obs)
            self.obstacle_centers[0] = obs.center
            self.obstacle_radii[0] = obs.radius

        for i in range(1, self.num_obstacles):
            obs = sets.Ball(
                self.feasible_set.sample(),
                self._rand(1).squeeze(dim=1) * (
                        self.max_radius - self.min_radius) + self.min_radius
            )
            obstructing = self.check_obstruction(obs)
            while obstructing.any():
                center = self.feasible_set.sample()
                radius = self._rand(1).squeeze(dim=1) * (
                        self.max_radius - self.min_radius) + self.min_radius
                obs.center[obstructing, :] = center[obstructing, :]
                obs.radius[obstructing] = radius[obstructing]
                obstructing = self.check_obstruction(obs)

            self.obstacles.append(obs)
            self.obstacle_centers[i] = obs.center
            self.obstacle_radii[i] = obs.radius

    def check_obstruction(self, obs):
        obstructing = obs.contains(self.state)
        obstructing |= obs.contains(self.goal)
        for other in self.obstacles:
            obstructing |= obs.intersects(other)
        return obstructing

    @jaxtyped(typechecker=beartype)
    def timestep(self,
                 action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]):
        """
        Perform a single timestep of the environment.

        Args:
            action: Action to take in the environment.
        """
        free_state = self.dynamics(action)
        if self.num_obstacles:
            self.collided = self.collision_check(free_state)
            if self.collided.any():
                self.collision_operator(free_state)
            else:
                self.state = free_state
        else:
            self.state = free_state

    def collision_check(self,
                        state: Float[Tensor, "{self.num_envs} {self.state.shape[1]}"]) \
            -> Bool[Tensor, "{self.num_envs} {self.num_obstacles}"]:
        """
        Check if the state collides with any obstacles.

        Args:
            state: The state to check for collisions.

        Returns:
            True if the state collides with any obstacles, False otherwise.
        """
        collided = torch.zeros_like(self.collided)
        for i in range(len(self.obstacles)):
            collided[:, i] = self.obstacles[i].contains(state)
        return collided

    def collision_operator(self, free_state: Float[
        Tensor, "{self.num_envs} {self.state.shape[1]}"]):
        """
        Apply an elastic collision operator by simulating the actual path intersection
        and bounce for colliding states.

        Args:
            free_state: The state to correct for collisions.
        """
        state_mask = self.collided.any(dim=1)
        obstacle_mask = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        nonzero = self.collided.nonzero()
        obstacle_mask[nonzero[:, 0]] = nonzero[:, 1].type(torch.int32)

        center = self.obstacle_centers[obstacle_mask][state_mask]
        radius = self.obstacle_radii[obstacle_mask][state_mask].unsqueeze(1)

        to_start = self.state[state_mask] - center

        # Solve quadratic equation for intersection time
        free_vel = free_state[state_mask] - self.state[state_mask]
        direction = free_vel
        a = torch.sum(direction * direction, dim=1, keepdim=True)
        b = 2 * torch.sum(to_start * direction, dim=1, keepdim=True)
        c = torch.sum(to_start * to_start, dim=1, keepdim=True) - radius ** 2
        discriminant = b ** 2 - 4 * a * c
        t = (-b - torch.sqrt(discriminant)) / (2 * a)  # First intersection time

        intersection = self.state[state_mask] + t * direction

        # Calculate normal at intersection point
        normal = intersection - self.obstacle_centers[obstacle_mask][state_mask]
        normal = normal / torch.norm(normal, dim=1, keepdim=True)

        # Reflect velocity about normal vector
        vel_dot_normal = torch.sum(free_vel * normal, dim=1, keepdim=True)
        reflected_vel = free_vel - 2 * vel_dot_normal * normal

        # Calculate remaining time after collision
        remaining_time = (1 - t) * self.dt

        collided_pos = self.state.clone()
        collided_pos[state_mask] = intersection + reflected_vel * remaining_time

        collided_state = torch.where(state_mask.unsqueeze(1), collided_pos, self.state)

        self.state = torch.where(state_mask.unsqueeze(1), collided_state, free_state)

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
        goal_distance = torch.norm(self.goal - self.state, dim=1)
        loss = -self.distance_penalty * goal_distance / self.initial_goal_distance
        for obs in self.obstacles:
            signed_distance = torch.norm(obs.center - self.state, dim=1) - obs.radius
            threshold = obs.radius * 1.1
            one = torch.ones_like(loss)
            zero = torch.zeros_like(loss)
            obstacle_loss = torch.max(zero,
                                      (torch.exp(
                                          -signed_distance / threshold) - torch.exp(
                                          -one)) / (1 - torch.exp(-one)))
            loss = loss - obstacle_loss * self.collision_penalty
        return loss

    @jaxtyped(typechecker=beartype)
    def safe_action_set(self) -> sets.Zonotope:
        """
        Get the safe state set for the current state.

        Returns:
            A convex set representing the safe state set.

        Note:
            Cache the result if it is expensive to compute.
        """
        with torch.no_grad():
            generator = self.calculate_generator(self.state)

            self.last_safe_state_set = sets.Zonotope(self.action_set.center, generator)
            return self.last_safe_state_set

    @jaxtyped(typechecker=beartype)
    def construct_generator_function(self):
        length = cp.Variable(2, nonneg=True)

        center = cp.Parameter(2)  # Reachability
        unscaled_generator = cp.Parameter((2, 2))
        generator = unscaled_generator @ cp.diag(length)
        parameters = [center, unscaled_generator]

        constraints = [
            # State Feasibility
            self.state_bounds[0, :, 0].cpu().numpy() <= center - cp.abs(generator).sum(
                axis=1),
            self.state_bounds[0, :, 1].cpu().numpy() >= center + cp.abs(generator).sum(
                axis=1),
            # Action Feasibility
            length <= np.ones(2),
        ]
        for i in range(self.num_obstacles):
            # Products for DPP compliance
            unscaled_generator_x_direction = cp.Parameter(2)
            distance = cp.Parameter(1)
            parameters += [unscaled_generator_x_direction, distance]

            # Collision avoidance
            support_pos = cp.sum(cp.abs(unscaled_generator_x_direction @ cp.diag(length)))

            constraints += [support_pos <= distance]

        objective = cp.Maximize(cp.geo_mean(length))

        problem = cp.Problem(objective, constraints)

        length_layer = CvxpyLayer(problem, parameters=parameters, variables=[length])

        def length_fn(center):
            parameters = [center]
            unscaled_generator = torch.diag_embed(
                torch.ones(*self.action_space.shape, dtype=torch.float64, device=self.device))
            min_distance = (self.state.abs() - self.state_bounds[:, :, 1]).norm(dim=1, keepdim=True)
            for i, ball in enumerate(self.obstacles):
                direction = ball.center - center
                distance = torch.norm(direction, dim=1, keepdim=True)
                to_obs = direction / distance
                distance -= ball.radius.unsqueeze(1)
                normal = torch.zeros_like(to_obs)
                normal[:, 0] = to_obs[:, 1]
                normal[:, 1] = -to_obs[:, 0]
                mask = (distance < min_distance).squeeze(dim=1)
                unscaled_generator[mask, :, 0] = to_obs[mask]
                unscaled_generator[mask, :, 1] = normal[mask]
                min_distance = torch.where(distance < min_distance, distance, min_distance)
            parameters += [unscaled_generator]

            for i, ball in enumerate(self.obstacles):
                direction = ball.center - center
                distance = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / distance
                distance -= ball.radius.unsqueeze(1)

                parameters += [torch.sum(direction.unsqueeze(2) * unscaled_generator, dim=1), distance]

            length = length_layer(*parameters, solver_args={"solve_method": "Clarabel"})[0]
            return unscaled_generator * length.unsqueeze(1)

        return length_fn

    def draw(self):
        super().draw()
        world_width = np.ceil(
            self.single_observation_space.high[0] - self.single_observation_space.low[
                0]).item() + 1.0
        world_height = np.ceil(
            self.single_observation_space.high[1] - self.single_observation_space.low[
                1]).item() + 1.0
        scale_x = self.screen_width / world_width
        scale_y = self.screen_height / world_height

        for obstacle in self.obstacles:
            center = obstacle.center[0].cpu().numpy()
            radius = obstacle.radius[0].item()

            # Convert center point to screen coordinates
            x = int((center[0] + world_width / 2) * scale_x)
            y = int(self.screen_height - (center[1] + world_height / 2) * scale_y)
            screen_radius = int(radius * self.screen_width / world_width)

            gfxdraw.aacircle(self.surf, x, y, screen_radius, (0, 0, 0))
            gfxdraw.filled_circle(self.surf, x, y, screen_radius, (0, 0, 0))

        if self.draw_safe_state_set:
            if self.last_safe_state_set is None:
                self.safe_action_set()

            vertices = sets.Zonotope(self.state, self.last_safe_state_set.generator).vertices().cpu().numpy()
            try:
                # Convert vertices to screen coordinates
                screen_vertices = [
                    (
                        int((vertex[0] + world_width / 2) * scale_x),
                        int(self.screen_height - (
                                vertex[1] + world_height / 2) * scale_y)
                    )
                    for vertex in vertices.T
                ]

                # Create a new surface with an alpha channel
                shade_surface = pygame.Surface(
                    (self.screen_width, self.screen_height),
                    pygame.SRCALPHA)
                shade_color = (255, 0, 0, 128)  # Red color with 50% opacity

                # Draw the safe state set zonotope on the new surface
                gfxdraw.aapolygon(shade_surface, screen_vertices, shade_color)
                gfxdraw.filled_polygon(shade_surface, screen_vertices, shade_color)

                # Blit the new surface onto the main surface
                self.surf.blit(shade_surface, (0, 0))
            except OverflowError:
                pass
            except ValueError:
                pass
            self.last_safe_state_set = None  # Invalidate to remove drawing
