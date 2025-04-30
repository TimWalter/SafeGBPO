from typing import Literal, Optional, Any, Callable

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
from tasks.envs.quadrotor import QuadrotorEnv
from tasks.interfaces.safe_state_task import SafeStateTask


class NavigateQuadrotorTask(QuadrotorEnv, SafeStateTask):
    """
    ## The reward describes a navigation task, where a goal has to be reached,
    without colliding with obstacles. The distance to the goal is punished and
    collisions are punished, reaching the goal is rewarded.

    ## Starting State

    The starting state and goal positions are sampled uniformly from the domain. The
    velocities and roll are zero.

    ## Episode Truncation

    The episode truncates at 400 time steps.
    """

    collision_penalty: float = 10.0
    distance_penalty: float = 1.0
    success_reward: float = 100.0

    last_safe_state_set: sets.Zonotope = None

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
                [max_radius, np.sqrt(domain_width ** 2 + domain_height ** 2)],  # Obstacle distance
            ] * num_obstacles, dtype=np.float64)
        QuadrotorEnv.__init__(self, device, num_envs, False, render_mode,
                              max_episode_steps, domain_width, domain_height,
                              additional_obs)
        self.num_obstacles = num_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.draw_safe_state_set = draw_safe_state_set

        self.domain = sets.Box(
            torch.zeros(num_envs, 2, device=device, dtype=torch.float64),
            torch.diag_embed(
                torch.tensor([[domain_width / 2, domain_height / 2]] * num_envs,
                             dtype=torch.float64, device=device))
        )
        # Roll bound pi/12 is not reachable, ~pi/16 is reachable
        self.max_acceleration = torch.tensor([
            (self.thrust_mag + self.gravity) * np.sin(np.pi / 16),
            self.thrust_mag,
        ], dtype=torch.float64, device=self.device)

        self.thrust_mag = 10.0

        self.screen_height = domain_height * 50
        self.screen_width = domain_width * 50

        self.obstacles: list[sets.Ball] = []
        self.obstacle_centers = CoupledTensor((num_obstacles, self.num_envs, 2),
                                              torch.float64, self.device)
        self.obstacle_radii = CoupledTensor((num_obstacles, self.num_envs),
                                            torch.float64, self.device)

        self.collided = torch.zeros((self.num_envs, num_obstacles),
                                    device=self.device, dtype=torch.bool)
        self.initial_goal_distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self.reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.roll_generators = torch.zeros(self.num_envs, 6, 2, dtype=torch.float64, device=self.device)
        self.roll_generators[:, 2, 0] = torch.pi / 12
        self.roll_generators[:, 5, 1] = torch.pi / 2

        self.max_support_boundary_layer = self.construct_max_support_boundary_layer()
        self.calculate_pos_generators = self.construct_pos_generators_function()
        self.calculate_vel_generators = self.construct_vel_generators_function()
        SafeStateTask.__init__(self, device, 6)

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
                    torch.norm(self.state[:, :2] - self.obstacles[i].center, dim=1, keepdim=True) - self.obstacles[
                        i].radius.unsqueeze(1)
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
        start = self.domain.sample()
        self.state = torch.cat(
            [start, torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float64)], dim=1)
        too_close = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        while too_close.any():
            self.goal[too_close] = self.domain.sample()[too_close]
            too_close[:] = torch.norm(self.goal - self.state[:, :2], dim=1) < self.max_radius * 2
        self.sample_obstacles()

        self.steps = torch.zeros_like(self.steps)
        self.scheduled_reset = torch.zeros_like(self.scheduled_reset)
        self.collided = torch.zeros_like(self.collided)
        self.reached = torch.zeros_like(self.reached)
        self.initial_goal_distance = torch.norm(self.goal - self.state[:, :2], dim=1)

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
            ray = (self.goal - self.state[:, :2]) / torch.norm(self.goal - self.state[:, :2], dim=1, keepdim=True)
            normal_ray = torch.zeros_like(ray)
            normal_ray[:, 0] = ray[:, 1]
            normal_ray[:, 1] = -ray[:, 0]
            obs = sets.Ball(
                (self.state[:, :2] + self.goal) / 2 + self._rand(1) * normal_ray * self.min_radius / 2,
                self._rand(1).squeeze(dim=1) * (self.max_radius - self.min_radius) + self.min_radius
            )
            self.obstacles.append(obs)
            self.obstacle_centers[0] = obs.center
            self.obstacle_radii[0] = obs.radius
        for i in range(1, self.num_obstacles):
            obs = sets.Ball(
                self.domain.sample(),
                self._rand(1).squeeze(dim=1) * (
                        self.max_radius - self.min_radius) + self.min_radius
            )
            obstructing = self.check_obstruction(obs)
            while obstructing.any():
                center = self.domain.sample()
                radius = self._rand(1).squeeze(dim=1) * (
                        self.max_radius - self.min_radius) + self.min_radius
                obs.center[obstructing, :] = center[obstructing, :]
                obs.radius[obstructing] = radius[obstructing]
                obstructing = self.check_obstruction(obs)

            self.obstacles.append(obs)
            self.obstacle_centers[i] = obs.center
            self.obstacle_radii[i] = obs.radius

    def check_obstruction(self, obs):
        obstructing = obs.contains(self.state[:, :2])
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
            collided[:, i] = self.obstacles[i].contains(state[:, :2])
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

        x, z, roll, x_dot, z_dot, roll_dot = self.state.split(1, dim=1)
        pos = torch.cat([x, z], dim=1)
        vel = torch.cat([x_dot, z_dot], dim=1)

        free_vel = free_state[:, 3:5]

        to_start = pos[state_mask] - self.obstacle_centers[obstacle_mask][state_mask]
        radius = self.obstacle_radii[obstacle_mask][state_mask].unsqueeze(1)

        # Solve quadratic equation for intersection time
        direction = free_vel[state_mask] * self.dt
        a = torch.sum(direction * direction, dim=1, keepdim=True)
        b = 2 * torch.sum(to_start * direction, dim=1, keepdim=True)
        c = torch.sum(to_start * to_start, dim=1, keepdim=True) - radius ** 2
        discriminant = b ** 2 - 4 * a * c
        t = (-b - torch.sqrt(discriminant)) / (2 * a)  # First intersection time

        intersection = pos[state_mask] + t * direction

        # Calculate normal at intersection point
        normal = intersection - self.obstacle_centers[obstacle_mask][state_mask]
        normal = normal / torch.norm(normal, dim=1, keepdim=True)

        # Reflect velocity about normal vector
        vel_dot_normal = torch.sum(free_vel[state_mask] * normal, dim=1, keepdim=True)
        reflected_vel = free_vel[state_mask] - 2 * vel_dot_normal * normal

        # Calculate remaining time after collision
        remaining_time = (1 - t) * self.dt

        collided_pos = pos.clone()
        collided_pos[state_mask] = intersection + reflected_vel * remaining_time

        collided_vel = vel.clone()
        collided_vel[state_mask] = reflected_vel

        pos = torch.where(state_mask.unsqueeze(1), collided_pos, pos)
        vel = torch.where(state_mask.unsqueeze(1), collided_vel, vel)
        collided_state = torch.cat([pos, roll, vel, roll_dot], dim=1)

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
        goal_distance = torch.norm(self.goal - self.state[:, :2], dim=1)

        just_reached = ~self.reached & (goal_distance < self.dt / 10)
        self.reached[just_reached] = True

        return (self.success_reward * just_reached
                - self.distance_penalty * goal_distance / self.initial_goal_distance
                - self.collision_penalty * self.collided.any(dim=1))

    @jaxtyped(typechecker=beartype)
    def safe_state_set(self) -> sets.Zonotope:
        """
        Get the safe state set for the current state.

        Returns:
            A convex set representing the safe state set.

        Note:
            Cache the result if it is expensive to compute.
        """
        with torch.no_grad():
            reachable_set = self.reachable_set()
            center, pos_mask, vel_mask = self.safe_center(reachable_set)

            # Separate as geo_mean is not maximising correctly if one length is 0
            pos_generators = torch.ones(self.num_envs, 6, 2, device=self.device, dtype=torch.float64) * 1e-2  # Failsafe
            if pos_mask.any():
                pos_generators[pos_mask] = self.calculate_pos_generators(center[pos_mask][:, :2], pos_mask)
            vel_generators = self.calculate_vel_generators(center, reachable_set, vel_mask)

            generator = torch.cat([pos_generators, vel_generators, self.roll_generators], dim=2)
            self.last_safe_state_set = sets.Zonotope(center, generator)
            return self.last_safe_state_set

    def safe_center(self, reachable_set: sets.Zonotope) \
            -> tuple[
                Float[Tensor, "{self.num_envs} {self.state.shape[1]}"],
                Bool[Tensor, "{self.num_envs}"],
                Bool[Tensor, "{self.num_envs}"]
            ]:
        """
        Ensure the center of the safe set is feasible and not colliding.

        Args:
            reachable_set: The one step reachable set.

        Returns:
            Safeguarded center, Mask indicating for which this worked
        """
        center = reachable_set.center

        infeasible = (self.state_bounds[..., 0] > reachable_set.center) | \
                     (self.state_bounds[..., 1] < reachable_set.center)
        if infeasible.any():
            center = self.feasiblest_center(reachable_set, infeasible)

        collisions = self.collision_check(center)
        if collisions.any():
            center = self.non_intersecting_center(center, reachable_set, collisions)

        pos_mask = (self.state_bounds[:, :2, 0] > reachable_set.center[:, :2]).any(dim=1) | \
                   (self.state_bounds[:, :2, 1] < reachable_set.center[:, :2]).any(dim=1) | \
                   self.collision_check(center).any(dim=1)
        vel_mask = (self.state_bounds[:, 3:5, 0] > reachable_set.center[:, 3:5]).any(dim=1) | \
                   (self.state_bounds[:, 3:5, 1] < reachable_set.center[:, 3:5]).any(dim=1)
        return center, ~pos_mask, ~vel_mask

    def feasiblest_center(self, reachable_set: sets.Zonotope, infeasible: Tensor):
        """
        If the center is infeasible try to make it as feasible as possible
        To allow the safe state to get away from the boundary, center cant be at
        boundary, alleviate by treating as collision
        """
        state_mask = infeasible.any(dim=1)
        unsafe_center = reachable_set.center[state_mask]
        generator = reachable_set.generator[state_mask]

        direction = torch.zeros_like(unsafe_center)
        too_low = self.state_bounds[state_mask, :, 0] > unsafe_center
        too_high = self.state_bounds[state_mask, :, 1] < unsafe_center

        # Roll and roll velocity determine x velocity
        too_low[:, 2] |= too_low[:, 3]
        too_high[:, 2] |= too_high[:, 3]
        too_low[:, 5] |= too_low[:, 3]
        too_high[:, 5] |= too_high[:, 3]

        direction[too_low] = -self.state_bounds[state_mask][too_low][:, 0]
        direction[too_high] = -self.state_bounds[state_mask][too_high][:, 1]

        shifted_center = reachable_set.center.clone()
        shifted_center[state_mask] = self.max_support_boundary_layer(unsafe_center,
                                                                     generator,
                                                                     direction)

        center = torch.where(state_mask.unsqueeze(1), shifted_center,
                             reachable_set.center)
        return center

    def non_intersecting_center(self, center, reachable_set, collisions):
        """
        Ensure center is not lying in an obstacle, if so go as far away from it as possible
        """
        state_mask = collisions.any(dim=1)
        unsafe_center = center[state_mask]
        generator = reachable_set.generator[state_mask]

        obstacle_mask = torch.zeros(self.num_envs, dtype=torch.int,
                                    device=self.device)
        nonzero = collisions.nonzero()
        obstacle_mask[nonzero[:, 0]] = nonzero[:, 1].type(torch.int32)
        direction = unsafe_center[:, :2] - self.obstacle_centers[obstacle_mask][
            state_mask]
        direction = torch.cat([direction, torch.zeros_like(unsafe_center)[:, :4]],
                              dim=1)

        center[state_mask] = self.max_support_boundary_layer(unsafe_center, generator,
                                                             direction)
        return center

    @jaxtyped(typechecker=beartype)
    def construct_max_support_boundary_layer(self) -> Callable[
        [
            Float[Tensor, "batch_dim {self.state.shape[1])}"],
            Float[Tensor, "batch_dim {self.state.shape[1])}"],
            Float[
                Tensor, "batch_dim {self.state.shape[1])} {self.action_space.shape[1]}"],
        ],
        Float[Tensor, "batch_dim {self.action_dim}"]
    ]:
        """
        Construct the cvxpylayer to compute the boundary of a given set which has the
        highest projection value among the given direction (max. support)
        """
        center = cp.Parameter(self.state.shape[1])
        generator = cp.Parameter((self.state.shape[1], self.action_space.shape[1]))
        dir_x_gen = cp.Parameter(self.action_space.shape[1])

        parameters = [center, generator, dir_x_gen]

        beta = cp.Variable(self.action_space.shape[1])
        max_support_boundary = center + generator @ beta
        objective = cp.Maximize(cp.sum(cp.multiply(dir_x_gen, beta)))

        constraints = [
            # Feasibility
            self.state_bounds[0, :, 0].cpu().numpy() <= max_support_boundary,
            self.state_bounds[0, :, 1].cpu().numpy() >= max_support_boundary,
            cp.norm(beta, "inf") <= 1
        ]

        problem = cp.Problem(objective, constraints)

        flow_layer = CvxpyLayer(problem, parameters=parameters, variables=[beta])

        def max_support_boundary_fn(center, generator, direction) \
                -> Float[Tensor, "batch_dim {self.state.shape[1])}"]:
            dir_x_gen = torch.sum(direction.unsqueeze(2) * generator, dim=1)
            params = [center, generator, dir_x_gen]
            beta = flow_layer(*params, solver_args={"solve_method": "Clarabel"})[0]
            max_support_boundary = center + torch.sum(generator * beta.unsqueeze(1),
                                                      dim=2)
            return max_support_boundary

        return max_support_boundary_fn

    @jaxtyped(typechecker=beartype)
    def construct_pos_generators_function(self):
        length = cp.Variable(2, nonneg=True)

        center = cp.Parameter(2)  # Reachability
        unscaled_generator = cp.Parameter((2,2))
        generator = unscaled_generator @ cp.diag(length)
        parameters = [center, unscaled_generator]

        constraints = [
            # Feasibility
            self.state_bounds[0, :2, 0].cpu().numpy() <= center - cp.abs(generator).sum(
                axis=1),
            self.state_bounds[0, :2, 1].cpu().numpy() >= center + cp.abs(generator).sum(
                axis=1),
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

        def length_fn(center, mask):
            parameters = [center]
            unscaled_generator = torch.diag_embed(
                torch.ones((mask.sum(), 2), dtype=torch.float64, device=self.device))
            min_distance = (center.abs() - self.state_bounds[mask][:, :2, 1]).norm(dim=1, keepdim=True)
            for i, ball in enumerate(self.obstacles):
                direction = ball.center[mask] - center
                distance = torch.norm(direction, dim=1, keepdim=True)
                to_obs = direction / distance
                distance -= ball.radius[mask].unsqueeze(1)
                normal = torch.zeros_like(to_obs)
                normal[:, 0] = to_obs[:, 1]
                normal[:, 1] = -to_obs[:, 0]
                dist_mask = (distance < min_distance).squeeze(dim=1)
                unscaled_generator[dist_mask, :, 0] = to_obs[dist_mask]
                unscaled_generator[dist_mask, :, 1] = normal[dist_mask]
                min_distance = torch.where(distance < min_distance, distance, min_distance)
            parameters += [unscaled_generator]
            for i, ball in enumerate(self.obstacles):
                direction = ball.center[mask] - center
                distance = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / distance
                distance -= ball.radius[mask].unsqueeze(1)

                parameters += [torch.sum(direction.unsqueeze(2) * unscaled_generator, dim=1), distance]

            length = length_layer(*parameters, solver_args={"solve_method": "Clarabel"})[0]
            # Fallback Length for unavoidable collisions or infeasibility
            length[length.isnan().any(dim=1) | length.isinf().any(dim=1)] = 1e-2
            generator = unscaled_generator * length.unsqueeze(1)
            padding = torch.zeros((mask.sum(), 4, 2), dtype=torch.float64, device=self.device)
            return torch.cat((generator, padding), dim=1)

        return length_fn

    @jaxtyped(typechecker=beartype)
    def construct_vel_generators_function(self):
        length = cp.Variable(2, nonneg=True)

        center = cp.Parameter(2)  # Reachability
        unscaled_generator = cp.Parameter((2, 2))
        generator = unscaled_generator @ cp.diag(length)
        parameters = [unscaled_generator]

        constraints = [
            # Feasibility is removed to enable larger sets at the boundary? (symmetry)
            # self.state_bounds[0, 3:5, 0].cpu().numpy() <= center - cp.abs(generator).sum(axis=1),
            # self.state_bounds[0, 3:5, 1].cpu().numpy() >= center + cp.abs(generator).sum(axis=1),
        ]
        # Do not crash into walls
        lower_wall_bounds = cp.Parameter(2)
        upper_wall_bounds = cp.Parameter(2)
        parameters += [lower_wall_bounds, upper_wall_bounds]
        constraints += [
            lower_wall_bounds <= -cp.abs(generator).sum(axis=1),
            upper_wall_bounds >= cp.abs(generator).sum(axis=1)
        ]

        for i in range(self.num_obstacles):
            # Products for DPP compliance
            unscaled_generator_x_direction = cp.Parameter(2)
            vel_bound = cp.Parameter(1)
            parameters += [unscaled_generator_x_direction, vel_bound]

            # Collision avoidance / Breaking
            support_vel = cp.sum(cp.abs(unscaled_generator_x_direction @ cp.diag(length)))

            constraints += [support_vel <= vel_bound]

        objective = cp.Maximize(cp.geo_mean(length))

        problem = cp.Problem(objective, constraints)

        length_layer = CvxpyLayer(problem, parameters=parameters, variables=[length])

        def length_fn(center, reachable_set: sets.Zonotope, mask):
            parameters = []

            reachable_box = reachable_set.box()
            lower_reach = reachable_box.center[:, :2] - reachable_box.generator[:, :2,
                                                        :2].abs().sum(dim=2)
            upper_reach = reachable_box.center[:, :2] + reachable_box.generator[:, :2,
                                                        :2].abs().sum(dim=2)
            lower_distance = torch.abs(self.state_bounds[:, :2, 0] - lower_reach)
            upper_distance = torch.abs(self.state_bounds[:, :2, 1] - upper_reach)

            # for z direction assuming maximum acceleration is pretty accurate (cos(r)=1)
            # for x assume it takes us 10 timesteps to reach maximum acceleration
            # average velocity during these is roughly half of the current
            lower_distance[:, 0] += 5 * self.dt * center[:, 3]
            upper_distance[:, 0] -= 5 * self.dt * center[:, 3]

            lower_distance[lower_distance < 0] = 1e-2  # Failsafe
            upper_distance[upper_distance < 0] = 1e-2  # Failsafe

            lower_wall_bounds = -torch.sqrt(
                lower_distance * 2 * self.max_acceleration.unsqueeze(0)
            ) - center[:, 3:5]
            lower_wall_bounds[lower_wall_bounds > 0] = 0.0  # Failsafe
            upper_wall_bounds = torch.sqrt(
                upper_distance * 2 * self.max_acceleration.unsqueeze(0)
            ) - center[:, 3:5]
            upper_wall_bounds[upper_wall_bounds < 0] = 0.0  # Failsafe

            unscaled_generator = torch.diag_embed(torch.ones((self.num_envs, 2), dtype=torch.float64, device=self.device))
            min_distance = (center[:, 3:5].abs() - self.state_bounds[:, 3:5, 1]).norm(dim=1, keepdim=True)
            lower_distance = lower_wall_bounds.abs().min(dim=1, keepdim=True).values
            min_distance = torch.where(lower_distance < min_distance, lower_distance, min_distance)
            upper_distance = upper_wall_bounds.abs().min(dim=1, keepdim=True).values
            min_distance = torch.where(upper_distance < min_distance, upper_distance, min_distance)
            for i, ball in enumerate(self.obstacles):
                direction = ball.center - center[:, :2]
                distance = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / distance

                speed_in_dir = torch.sum(center[:, 3:5] * direction, dim=1,
                                         keepdim=True)

                distance -= 5 * self.dt * speed_in_dir
                distance[distance < 0] = 1e-2  # Failsafe
                vel_bound = torch.sqrt(
                    distance * torch.sum(
                        2 * direction * self.max_acceleration.unsqueeze(0), dim=1,
                        keepdim=True).abs()
                ) - speed_in_dir

                # Negative bounds lead to infeasible problem
                vel_bound[vel_bound < 0] = 0.0

                normal = torch.zeros_like(direction)
                normal[:, 0] = direction[:, 1]
                normal[:, 1] = -direction[:, 0]
                dist_mask = (vel_bound < min_distance).squeeze(dim=1)
                unscaled_generator[dist_mask, :, 0] = direction[dist_mask]
                unscaled_generator[dist_mask, :, 1] = normal[dist_mask]
                min_distance = torch.where(vel_bound < min_distance, vel_bound, min_distance)

            parameters += [unscaled_generator, lower_wall_bounds, upper_wall_bounds]

            for i, ball in enumerate(self.obstacles):
                direction = ball.center - center[:, :2]
                # Instead of center point with minimum distance in the reachable set
                # would be safe but requires one more solve, therefore omit for now
                distance = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / distance
                distance -= ball.radius.unsqueeze(1)

                speed_in_dir = torch.sum(center[:, 3:5] * direction, dim=1,
                                         keepdim=True)

                distance -= 5 * self.dt * speed_in_dir
                distance[distance < 0] = 1e-2  # Failsafe

                vel_bound = torch.sqrt(
                    distance * torch.sum(
                        2 * direction * self.max_acceleration.unsqueeze(0), dim=1,
                        keepdim=True).abs()
                ) - speed_in_dir

                # Negative bounds lead to infeasible problem
                vel_bound[vel_bound < 0] = 0.0

                parameters += [direction, vel_bound]

            length = length_layer(*parameters, solver_args={"solve_method": "Clarabel"})[0]
            # Fallback Length for unavoidable collisions or infeasibility
            length[length.isnan().any(dim=1) | length.isinf().any(dim=1)] = 1e-2
            generator = unscaled_generator * length.unsqueeze(1)
            padding1 = torch.zeros((self.num_envs, 3, 2), dtype=torch.float64, device=self.device)
            padding2 = torch.zeros((self.num_envs, 1, 2), dtype=torch.float64, device=self.device)
            return torch.cat((padding1, generator, padding2), dim=1)

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
                self.safe_state_set()
            pos_zonotope = sets.Zonotope(self.last_safe_state_set.center[:, :2],
                                         self.last_safe_state_set.generator[:, :2, :2])
            pos_vertices = pos_zonotope.vertices().cpu().numpy()
            vel_zonotope = sets.Zonotope(self.last_safe_state_set.center[:, :2] + self.last_safe_state_set.center[:, 3:5],
                                         self.last_safe_state_set.generator[:, 3:5, 2:4])
            vel_vertices = vel_zonotope.vertices().cpu().numpy()
            try:
                for i, vertices in enumerate([pos_vertices, vel_vertices]):
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
                    if i == 0:
                        shade_color = (255, 0, 0, 128)  # Red color with 50% opacity
                    else:
                        shade_color = (0, 0, 255, 128)

                    # Draw the safe state set zonotope on the new surface
                    gfxdraw.aapolygon(shade_surface, screen_vertices, shade_color)
                    gfxdraw.filled_polygon(shade_surface, screen_vertices, shade_color)

                    # Blit the new surface onto the main surface
                    self.surf.blit(shade_surface, (0, 0))
            except OverflowError:
                pass
            self.last_safe_state_set = None  # Invalidate to remove drawing
