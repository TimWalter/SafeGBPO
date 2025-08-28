from typing import Optional, Any

import torch
import cvxpy as cp
import numpy as np
from torch import Tensor
from beartype import beartype
from PIL import Image, ImageDraw
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import jaxtyped, Float, Bool
from torchvision.transforms.functional import to_tensor

import src.sets as sets
from envs.simulators.quadrotor import QuadrotorEnv
from envs.interfaces.safe_state_env import SafeStateEnv
from src.learning_algorithms.components.coupled_tensor import CoupledTensor


class NavigateQuadrotorEnv(QuadrotorEnv, SafeStateEnv):
    """
    The quadrotor has to navigate through an environment with obstacles to reach a goal.

    ## Reward
    Since the goal is to reach the goal position while avoiding obstacles,
    the reward punishes for:
    - Collisions with obstacles
    - Distance to the goal position
    and rewards:
    - Reaching the goal position

    ## Starting State
    The starting state and goal positions are sampled uniformly from the domain. The
    velocities and roll are zero. The obstacles are sampled uniformly, ensuring that they do not overlap.
    The first obstacle is placed between the start and goal position.

    ## Safety
    The safe state set is computed based on the reachable set of the quadrotor, such that collisions are likely to be
    avoided, however the safe state set might become empty.
    """
    EVAL_ENVS: int = 6
    COLLISION_PENALTY: float = 10.0
    DISTANCE_PENALTY: float = 1.0
    GOAL_REWARD: float = 100.0

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 num_envs: int,
                 num_steps: int,
                 num_obstacles: int,
                 min_radius: float,
                 max_radius: float,
                 draw_safe_state_set: bool
                 ):
        """
        Initialize the NavigateQuadrotor environment.

        Args:
            num_envs: Number of environments to vectorize.
            num_steps: Number of steps till the episode ends.
            num_obstacles: Number of obstacles in the environment.
            min_radius: Minimum radius of the obstacles.
            max_radius: Maximum radius of the obstacles.
            draw_safe_state_set: Whether to draw the safe state set in the environment.
        """
        SafeStateEnv.__init__(self, num_state_gens=6)

        self.additional_observation_set = sets.AxisAlignedBox(
            torch.tensor([0.0, 0.0, min_radius + (max_radius - min_radius) / 2] * num_obstacles).unsqueeze(0).repeat(
                num_envs, 1),
            torch.diag_embed(
                torch.tensor([8.0, 8.0, (max_radius - min_radius) / 2] * num_obstacles).repeat(num_envs, 1))
        )
        QuadrotorEnv.__init__(self, num_envs, num_steps, self.additional_observation_set)

        self.num_obstacles = num_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.draw_safe_state_set = draw_safe_state_set

        # Roll bound pi/12 is not reachable, ~pi/16 is reachable
        self.max_acceleration = torch.tensor([
            (self.THRUST_MAG + self.GRAVITY) * np.sin(np.pi / 16),
            self.THRUST_MAG,
        ])

        self.obstacles: list[sets.Ball] = [sets.Ball(torch.empty((num_envs, 2)), torch.empty(num_envs)) for _ in
                                           range(num_obstacles)]
        self.obstacle_centers = CoupledTensor(num_obstacles, self.num_envs, 2)
        self.obstacle_radii = CoupledTensor(num_obstacles, self.num_envs)

        self.collided = torch.zeros((self.num_envs, num_obstacles), dtype=torch.bool)
        self.initial_goal_distance = torch.zeros(self.num_envs)
        self.reached = torch.zeros(self.num_envs, dtype=torch.bool)

        self.roll_generators = torch.zeros(self.num_envs, 6, 2)
        self.roll_generators[:, 2, 0] = torch.pi / 12
        self.roll_generators[:, 5, 1] = torch.pi / 2

        self.support_point_layer = None
        self.position_generator_layer = None
        self.velocity_generator_layer = None
        self.last_safe_state_set: sets.Zonotope = sets.Zonotope(torch.zeros(self.num_envs, self.state_dim),
                                                                torch.zeros(self.num_envs, self.state_dim,
                                                                            self.num_state_gens))

    @jaxtyped(typechecker=beartype)
    def reset(self, seed: Optional[int] = None) -> tuple[
        Float[Tensor, "{self.num_envs} {self.obs_dim}"],
        dict[str, Any]
    ]:
        """
        Reset all parallel environments and return a batch of initial observations
        and info.

        Args:
            seed: The environment reset seeds

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        super().reset(seed)
        self.sample_goal()
        self.sample_obstacles()

        self.collided = torch.zeros_like(self.collided)
        self.reached = torch.zeros_like(self.reached)
        self.initial_goal_distance = torch.norm(self.goal - self.state[:, :2], dim=1)

        return self.observation, {}

    @jaxtyped(typechecker=beartype)
    def sample_goal(self):
        """
        Sample a goal position uniformly from the domain, ensuring it is not too close to the starting position,
        such that one obstacle can be placed between the start and goal position.
        """
        too_close = torch.ones(self.num_envs, dtype=torch.bool)
        while too_close.any():
            self.goal[too_close] = torch.rand(int(too_close.sum()), 2) * 16.0 - 8.0
            too_close[:] = torch.norm(self.goal - self.state[:, :2], dim=1) < self.max_radius * 2

    @jaxtyped(typechecker=beartype)
    def sample_obstacles(self):
        """
        Sample obstacles uniformly, ensuring that they do not overlap with each other or the start/goal positions.
        """
        for i in range(0, self.num_obstacles):
            obstructing = torch.ones(self.num_envs, dtype=torch.bool)
            while obstructing.any():
                self.sample_one_obstacle(i, obstructing)
                obstructing = self.check_obstruction(i)
            self.obstacle_centers[i] = self.obstacles[i].center
            self.obstacle_radii[i] = self.obstacles[i].radius

    @jaxtyped(typechecker=beartype)
    def sample_one_obstacle(self, i: int, obstructing: Bool[Tensor, "{self.num_envs}"]):
        """
        Sample one obstacle uniformly. The first obstacle is placed between the start and goal position.

        Args:
            i: Index of the obstacle to sample.
            obstructing: A boolean tensor indicating which environments should sample a new obstacle.
        """
        sample = self.additional_observation_set.sample()
        if i == 0:
            ray = (self.goal - self.state[:, :2]) / torch.norm(self.goal - self.state[:, :2], dim=1, keepdim=True)
            normal_ray = torch.zeros_like(ray)
            normal_ray[:, 0] = ray[:, 1]
            normal_ray[:, 1] = -ray[:, 0]
            center = (self.state[:, :2] + self.goal) / 2 + torch.rand(self.num_envs,
                                                                      1) * normal_ray * self.min_radius / 2
            radius = torch.rand(self.num_envs) * (self.max_radius - self.min_radius) + self.min_radius
        else:
            center = sample[:, i * 3:i * 3 + 2]
            radius = sample[:, i * 3 + 2]

        self.obstacles[i].center[obstructing, :] = center
        self.obstacles[i].radius[obstructing] = radius

    @jaxtyped(typechecker=beartype)
    def check_obstruction(self, i: int) -> Bool[Tensor, "{self.num_envs}"]:
        """
        Check if the i-th obstacle obstructs the start or goal position, or intersects with any previous obstacles.

        Args:
            i: Index of the obstacle to check for obstruction.

        Returns:
            A boolean tensor indicating which environments have the i-th obstacle obstructing.
        """

        obstructing = self.obstacles[i].contains(self.state[:, :2])
        obstructing |= self.obstacles[i].contains(self.goal)
        for other in self.obstacles[:i]:
            obstructing |= self.obstacles[i].intersects(other)
        return obstructing

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[Tensor, "{self.num_envs} {self.obs_dim}"]:
        """
        Get the current observation of the environment.

        Returns:
            The current observation of the environment as a batch of observations.
        """
        return torch.cat(
            [
                self.state, self.goal,
                *[
                    torch.cat([self.obstacles[i].center, self.obstacles[i].radius.unsqueeze(1)], dim=1)
                    for i in range(self.num_obstacles)
                ]
            ],
            dim=1)

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
        goal_distance = torch.norm(self.goal - self.state[:, :2], dim=1)

        just_reached = ~self.reached & (goal_distance < 0.1)
        self.reached[just_reached] = True

        return (self.GOAL_REWARD * just_reached
                - self.DISTANCE_PENALTY * goal_distance / self.initial_goal_distance
                - self.COLLISION_PENALTY * self.collided.any(dim=1))

    @jaxtyped(typechecker=beartype)
    def execute_action(self, action: Float[Tensor, "{self.num_envs} {self.action_dim}"]):
        """
        Execute the action in the environment by updating the state.

        Args:
            action: Action to execute in the environment.
        """
        free_state = self.dynamics(self.state, action, self.noise_set.sample())
        if self.num_obstacles:
            self.collided = self.collision_check(free_state)
            if self.collided.any():
                self.collision_operator(free_state)
            else:
                self.state = free_state
        else:
            self.state = free_state
        self.state = torch.clamp(self.state, self.state_set.min, self.state_set.max)

    @jaxtyped(typechecker=beartype)
    def collision_check(self, state: Float[Tensor, "{self.num_envs} {self.state_dim}"]) \
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

    @jaxtyped(typechecker=beartype)
    def collision_operator(self, free_state: Float[Tensor, "{self.num_envs} {self.state_dim}"]):
        """
        Apply an elastic collision operator by simulating the actual path intersection
        and bounce for colliding states.

        Args:
            free_state: The state to correct for collisions.
        """
        state_mask = self.collided.any(dim=1)
        obstacle_mask = torch.zeros(self.num_envs, dtype=torch.int)
        nonzero = self.collided.nonzero()
        obstacle_mask[nonzero[:, 0]] = nonzero[:, 1].type(torch.int32)

        x, z, roll, x_dot, z_dot, roll_dot = self.state.split(1, dim=1)
        pos = torch.cat([x, z], dim=1)
        vel = torch.cat([x_dot, z_dot], dim=1)

        free_vel = free_state[:, 3:5]

        to_start = pos[state_mask] - self.obstacle_centers[obstacle_mask][state_mask]
        radius = self.obstacle_radii[obstacle_mask][state_mask].unsqueeze(1)

        # Solve quadratic equation for intersection time
        direction = free_vel[state_mask] * self.DT
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
        remaining_time = (1 - t) * self.DT

        collided_pos = pos.clone()
        collided_pos[state_mask] = intersection + reflected_vel * remaining_time

        collided_vel = vel.clone()
        collided_vel[state_mask] = reflected_vel

        pos = torch.where(state_mask.unsqueeze(1), collided_pos, pos)
        vel = torch.where(state_mask.unsqueeze(1), collided_vel, vel)
        collided_state = torch.cat([pos, roll, vel, roll_dot], dim=1)

        self.state = torch.where(state_mask.unsqueeze(1), collided_state, free_state)

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
        """
        if self.num_obstacles == 1 and self.num_envs == 6:
            self.reset()
            self.state[:, 0:2] = torch.tensor([
                [-4.0, 0.0],
                [4.0, 0.0],
                [0.0, 4.0],
                [0.0, -4.0],
                [-4.0, 1.0],
                [4.0, 1.0]
            ])
            self.goal = torch.tensor([
                [4.0, 0.0],
                [-4.0, 0.0],
                [0.0, -4.0],
                [0.0, 4.0],
                [4.0, 1.0],
                [-4.0, 0.0]
            ])
            self.obstacles[0].center = torch.tensor([[0.0, 0.0]] * self.num_envs)
            self.obstacles[0].center[-1, :] = 1.0
            self.obstacles[0].radius = torch.tensor([self.min_radius] * self.num_envs)
            self.obstacles[0].radius[-1] = 1.0
            self.obstacle_centers[0] = self.obstacles[0].center
            self.obstacle_radii[0] = self.obstacles[0].radius
        else:
            super().eval_reset()

        return self.observation, {}

    @jaxtyped(typechecker=beartype)
    def render(self) -> list[Tensor]:
        """
        Render all environments.

        Returns:
            A list of rendered frames for each environment.
        """
        state = self.state.detach().cpu()
        goal = self.goal.detach().cpu()

        frames = []
        for i in range(self.num_envs):
            img, draw = self.draw_quadrotor(state[i], goal[i])

            world_bound = 8.5
            scale = self.SCREEN_WIDTH / (world_bound * 2)
            offset_x = self.SCREEN_WIDTH / 2
            offset_y = self.SCREEN_HEIGHT / 2

            for obstacle in self.obstacles:
                center = obstacle.center[i].cpu().numpy()
                radius = obstacle.radius[i].item()

                x = center[0] * scale + offset_x
                y = -center[1] * scale + offset_y
                screen_radius = radius * scale

                draw.ellipse((x - screen_radius, y - screen_radius, x + screen_radius, y + screen_radius),
                             fill=(0, 0, 0))

            if self.draw_safe_state_set:
                if self.last_safe_state_set.generator.sum() == 0:
                    self.safe_state_set()

                overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)

                pos_zonotope = sets.Zonotope(self.last_safe_state_set.center[i:i + 1, :2],
                                             self.last_safe_state_set.generator[i:i + 1, :2, :2])
                pos_vertices = pos_zonotope.vertices().cpu().numpy()

                vel_zonotope = sets.Zonotope(
                    self.last_safe_state_set.center[i:i + 1, :2] + self.last_safe_state_set.center[i:i + 1, 3:5],
                    self.last_safe_state_set.generator[i:i + 1, 3:5, 2:4])
                vel_vertices = vel_zonotope.vertices().cpu().numpy()

                for j, vertices in enumerate([pos_vertices, vel_vertices]):
                    screen_vertices = [
                        (v[0] * scale + offset_x, -v[1] * scale + offset_y)
                        for v in vertices.T
                    ]
                    color = (255, 0, 0, 64) if j == 0 else (0, 0, 255, 64)
                    overlay_draw.polygon(screen_vertices, fill=color)

                img = Image.alpha_composite(img, overlay)

            frames.append((to_tensor(img) * 255).to(torch.uint8))

        # invalidate the cached safe state set
        self.last_safe_state_set.generator *= 0.0

        return frames

    @jaxtyped(typechecker=beartype)
    def safe_state_set(self) -> sets.Zonotope:
        """
        Get the safe state set for the current state.

        Returns:
            A convex set representing the safe state set.

        Note:
            Cache the result if it is expensive to compute.
        """
        with (torch.no_grad()):
            reachable_set = self.reachable_set()
            center, safe_position, safe_velocity = self.safe_center(reachable_set)

            # Separate as geo_mean is not maximising correctly if one length is 0
            pos_generator = torch.where(safe_position.unsqueeze(1).unsqueeze(2),
                                         self.compute_position_generator(center),
                                         torch.ones(self.num_envs, self.state_dim, 2) * 1e-2)
            vel_generator = torch.where(safe_velocity.unsqueeze(1).unsqueeze(2),
                                         self.compute_velocity_generator(center, reachable_set),
                                         torch.ones(self.num_envs, self.state_dim, 2) * 1e-2)
            generator = torch.cat([pos_generator, vel_generator, self.roll_generators], dim=2)

            self.last_safe_state_set = sets.Zonotope(center, generator)
            return self.last_safe_state_set

    def safe_center(self, reachable_set: sets.Zonotope) \
            -> tuple[
                Float[Tensor, "{self.num_envs} {self.state_dim}"],
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
        center = self.feasible_center(reachable_set)
        center = self.collision_free_center(center, reachable_set)

        safe_position = ~((self.state_set.min[:, :2] > reachable_set.center[:, :2]).any(dim=1) |
                          (self.state_set.max[:, :2] < reachable_set.center[:, :2]).any(dim=1) |
                          self.collision_check(center).any(dim=1))
        safe_velocity = ~((self.state_set.min[:, 3:5] > reachable_set.center[:, 3:5]).any(dim=1) |
                          (self.state_set.max[:, 3:5] < reachable_set.center[:, 3:5]).any(dim=1))
        return center, safe_position, safe_velocity

    @jaxtyped(typechecker=beartype)
    def feasible_center(self, reachable_set: sets.Zonotope) -> Float[Tensor, "{self.num_envs} {self.state_dim}"]:
        """
        Try to ensure a feasible center. Infeasible centers are shifted as far away from the boundary as possible.

        Args:
            reachable_set: The one step reachable set.

        Returns:
            A (hopefully) feasible center for the safe state set.
        """
        too_low = self.state_set.min > reachable_set.center
        too_high = self.state_set.max < reachable_set.center

        # Roll and roll velocity determine x velocity
        too_low[:, 2] |= too_low[:, 3]
        too_high[:, 2] |= too_high[:, 3]
        too_low[:, 5] |= too_low[:, 3]
        too_high[:, 5] |= too_high[:, 3]

        direction = torch.ones_like(reachable_set.center)
        direction[too_low] = -self.state_set.min[too_low]
        direction[too_high] = -self.state_set.max[too_high]

        center = torch.where(~self.state_set.contains(reachable_set.center).unsqueeze(1),
                             self.compute_support_point(reachable_set, direction),
                             reachable_set.center)
        return center

    @jaxtyped(typechecker=beartype)
    def collision_free_center(self, center: Float[Tensor, "{self.num_envs} {self.state_dim}"], reachable_set) \
            -> Float[Tensor, "{self.num_envs} {self.state_dim}"]:
        """
        Ensure center is not lying in an obstacle, if so go as far away from it as possible.

        Args:
            center: The center of the safe state set.
            reachable_set: The one step reachable set.
        Returns:
            A center that is not colliding with any obstacles.
        """
        collisions = self.collision_check(center)

        obstacle_mask = torch.zeros(self.num_envs, dtype=torch.int)
        nonzero = collisions.nonzero()
        obstacle_mask[nonzero[:, 0]] = nonzero[:, 1].type(torch.int32)

        direction = center[:, :2] - self.obstacle_centers[obstacle_mask]
        direction = torch.cat([direction, torch.zeros_like(center)[:, :4]], dim=1)

        shifted_center = self.compute_support_point(reachable_set, direction)
        center = torch.where(collisions.any(dim=1, keepdim=True), shifted_center, center)
        return center

    @jaxtyped(typechecker=beartype)
    def compute_support_point(self,
                              zonotope: sets.Zonotope,
                              direction: Float[Tensor, "{self.num_envs} {self.state_dim}"]
                              ) -> Float[Tensor, "{self.num_envs} {self.state_dim}"]:
        """
        Compute the point of a zonotope that maximises its support function in a given direction.

        Args:
            zonotope: The zonotope to compute the support point for.
            direction: The direction in which to compute the support point.

        Returns:
            The support point of the zonotope in the given direction.
        """

        if self.support_point_layer is None:
            center = cp.Parameter(self.state.shape[1])
            generator = cp.Parameter((self.state.shape[1], self.action_dim))
            direction_times_generator = cp.Parameter(self.action_dim)
            parameters = [center, generator, direction_times_generator]

            beta = cp.Variable(self.action_dim)

            support_point = center + generator @ beta
            objective = cp.Maximize(cp.sum(cp.multiply(direction_times_generator, beta)))

            constraints = [
                # Feasibility
                self.state_set.min[0, :].cpu().numpy() <= support_point,
                self.state_set.max[0, :].cpu().numpy() >= support_point,
                cp.norm(beta, "inf") <= 1
            ]

            problem = cp.Problem(objective, constraints)
            self.support_point_layer = CvxpyLayer(problem, parameters=parameters, variables=[beta])

        direction_times_generator = torch.sum(direction.unsqueeze(2) * zonotope.generator, dim=1)
        parameters = [zonotope.center, zonotope.generator, direction_times_generator]
        beta = self.support_point_layer(*parameters, solver_args={"solve_method": "Clarabel"})[0]

        return zonotope.center + torch.sum(zonotope.generator * beta.unsqueeze(1), dim=2)

    @jaxtyped(typechecker=beartype)
    def compute_position_generator(self, full_center: Float[Tensor, "{self.num_envs} {self.state_dim}"]) \
            -> Float[Tensor, "{self.num_envs} {self.state_dim} 2"]:
        """
        Compute the position generators for the safe state set, such that they are orthogonal to the closest obstacle
        and ensure feasibility and collision avoidance.

        Args:
            full_center: The center of the safe state set.

        Returns:
            The position generators for the safe state set.
        """
        if self.position_generator_layer is None:
            center = cp.Parameter(2)  # Reachability
            unscaled_generator = cp.Parameter((2, 2))
            parameters = [center, unscaled_generator]
            for i in range(self.num_obstacles):
                # Products for DPP compliance
                unscaled_generator_times_direction = cp.Parameter(2)
                distance = cp.Parameter(1)
                parameters += [unscaled_generator_times_direction, distance]

            length = cp.Variable(2, nonneg=True)

            objective = cp.Maximize(cp.geo_mean(length))

            generator = unscaled_generator @ cp.diag(length)
            constraints = [
                # Feasibility
                self.state_set.min[0, :2].cpu().numpy() <= center - cp.abs(generator).sum(axis=1),
                self.state_set.max[0, :2].cpu().numpy() >= center + cp.abs(generator).sum(axis=1),
            ]
            for i in range(self.num_obstacles):
                # Collision avoidance
                support_pos = cp.sum(cp.abs(parameters[2 + i * 2] @ cp.diag(length)))
                constraints += [support_pos <= parameters[2 + i * 2 + 1]]

            problem = cp.Problem(objective, constraints)
            self.position_generator_layer = CvxpyLayer(problem, parameters=parameters, variables=[length])

        center = full_center[:, :2]
        # Determine directions orthogonal to the closest obstacle
        unscaled_generator = torch.diag_embed(torch.ones((self.num_envs, 2)))
        min_distance = (center.abs() - self.state_set.max[:, :2]).norm(dim=1, keepdim=True)
        for i, ball in enumerate(self.obstacles):
            direction = ball.center - center
            distance = torch.norm(direction, dim=1, keepdim=True)
            to_obs = direction / distance
            distance -= ball.radius.unsqueeze(1)
            normal = torch.zeros_like(to_obs)
            normal[:, 0] = to_obs[:, 1]
            normal[:, 1] = -to_obs[:, 0]
            dist_mask = (distance < min_distance).squeeze(dim=1)
            unscaled_generator[dist_mask, :, 0] = to_obs[dist_mask]
            unscaled_generator[dist_mask, :, 1] = normal[dist_mask]
            min_distance = torch.where(distance < min_distance, distance, min_distance)
        parameters = [center, unscaled_generator]
        for i, ball in enumerate(self.obstacles):
            direction = ball.center - center
            distance = torch.norm(direction, dim=1, keepdim=True)
            direction = direction / distance
            distance -= ball.radius.unsqueeze(1)
            parameters += [torch.sum(direction.unsqueeze(2) * unscaled_generator, dim=1), distance]

        length = self.position_generator_layer(*parameters, solver_args={"solve_method": "Clarabel"})[0]
        # Fallback Length for unavoidable collisions or infeasibility
        length[length.isnan().any(dim=1) | length.isinf().any(dim=1)] = 1e-2
        generator = unscaled_generator * length.unsqueeze(1)
        padding = torch.zeros((self.num_envs, 4, 2))
        return torch.cat((generator, padding), dim=1)

    @jaxtyped(typechecker=beartype)
    def compute_velocity_generator(self,
                                   center: Float[Tensor, "{self.num_envs} {self.state_dim}"],
                                   reachable_set: sets.Zonotope) \
            -> Float[Tensor, "{self.num_envs} {self.state_dim} 2"]:
        """
        Compute the velocity generators for the safe state set, ensuring they are orthogonal to the closest obstacle
        and ensure feasibility and collision avoidance.

        Args:
            center: The center of the safe state set.
            reachable_set: The one step reachable set.

        Returns:
            The velocity generators for the safe state set.
        """
        if self.velocity_generator_layer is None:
            # center = cp.Parameter(2)  # Reachability
            unscaled_generator = cp.Parameter((2, 2))
            lower_wall_bounds = cp.Parameter(2)
            upper_wall_bounds = cp.Parameter(2)
            parameters = [unscaled_generator, lower_wall_bounds, upper_wall_bounds]
            for i in range(self.num_obstacles):
                # Products for DPP compliance
                unscaled_generator_times_direction = cp.Parameter(2)
                vel_bound = cp.Parameter(1)
                parameters += [unscaled_generator_times_direction, vel_bound]

            length = cp.Variable(2, nonneg=True)

            objective = cp.Maximize(cp.geo_mean(length))

            generator = unscaled_generator @ cp.diag(length)
            constraints = [
                # Feasibility is removed to enable larger sets at the boundary? (symmetry)
                # self.state_set.min[0, 3:5].cpu().numpy() <= center - cp.abs(generator).sum(axis=1),
                # self.state_set.max[0, 3:5].cpu().numpy() >= center + cp.abs(generator).sum(axis=1),
                # Do not crash into walls
                lower_wall_bounds <= -cp.abs(generator).sum(axis=1),
                upper_wall_bounds >= cp.abs(generator).sum(axis=1)
            ]
            for i in range(self.num_obstacles):
                # Collision avoidance / Breaking
                support_pos = cp.sum(cp.abs(parameters[3 + i * 2] @ cp.diag(length)))
                constraints += [support_pos <= parameters[3 + i * 2 + 1]]

            problem = cp.Problem(objective, constraints)
            self.velocity_generator_layer = CvxpyLayer(problem, parameters=parameters, variables=[length])

        reachable_box = reachable_set.box()
        lower_reach = reachable_box.center[:, :2] - reachable_box.generator[:, :2, :2].abs().sum(dim=2)
        upper_reach = reachable_box.center[:, :2] + reachable_box.generator[:, :2, :2].abs().sum(dim=2)
        lower_distance = torch.abs(self.state_set.min[:, :2] - lower_reach)
        upper_distance = torch.abs(self.state_set.max[:, :2] - upper_reach)

        # for z direction assuming maximum acceleration is pretty accurate (cos(r)=1)
        # for x assume it takes us 10 timesteps to reach maximum acceleration
        # average velocity during these is roughly half of the current
        lower_distance[:, 0] += 5 * self.DT * center[:, 3]
        upper_distance[:, 0] -= 5 * self.DT * center[:, 3]

        lower_distance[lower_distance < 0] = 1e-2  # Failsafe
        upper_distance[upper_distance < 0] = 1e-2  # Failsafe

        lower_wall_bounds = -torch.sqrt(lower_distance * 2 * self.max_acceleration.unsqueeze(0)) - center[:, 3:5]
        lower_wall_bounds[lower_wall_bounds > 0] = 0.0  # Failsafe
        upper_wall_bounds = torch.sqrt(upper_distance * 2 * self.max_acceleration.unsqueeze(0)) - center[:, 3:5]
        upper_wall_bounds[upper_wall_bounds < 0] = 0.0  # Failsafe

        unscaled_generator = torch.diag_embed(torch.ones((self.num_envs, 2)))
        min_distance = (center[:, 3:5].abs() - self.state_set.max[:, 3:5]).norm(dim=1, keepdim=True)
        lower_distance = lower_wall_bounds.abs().min(dim=1, keepdim=True).values
        min_distance = torch.where(lower_distance < min_distance, lower_distance, min_distance)
        upper_distance = upper_wall_bounds.abs().min(dim=1, keepdim=True).values
        min_distance = torch.where(upper_distance < min_distance, upper_distance, min_distance)
        for i, ball in enumerate(self.obstacles):
            direction = ball.center - center[:, :2]
            distance = torch.norm(direction, dim=1, keepdim=True)
            direction = direction / distance
            speed_in_dir = torch.sum(center[:, 3:5] * direction, dim=1, keepdim=True)
            distance -= 5 * self.DT * speed_in_dir
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
        parameters = [unscaled_generator, lower_wall_bounds, upper_wall_bounds]

        for i, ball in enumerate(self.obstacles):
            direction = ball.center - center[:, :2]
            # Instead of center point with minimum distance in the reachable set
            # would be safe but requires one more solve, therefore omit for now
            distance = torch.norm(direction, dim=1, keepdim=True)
            direction = direction / distance
            distance -= ball.radius.unsqueeze(1)
            speed_in_dir = torch.sum(center[:, 3:5] * direction, dim=1, keepdim=True)
            distance -= 5 * self.DT * speed_in_dir
            distance[distance < 0] = 1e-2  # Failsafe
            vel_bound = torch.sqrt(
                distance * torch.sum(
                    2 * direction * self.max_acceleration.unsqueeze(0), dim=1,
                    keepdim=True).abs()
            ) - speed_in_dir
            # Negative bounds lead to infeasible problem
            vel_bound[vel_bound < 0] = 0.0
            parameters += [direction, vel_bound]

        length = self.velocity_generator_layer(*parameters, solver_args={"solve_method": "Clarabel"})[0]
        # Fallback Length for unavoidable collisions or infeasibility
        length[length.isnan().any(dim=1) | length.isinf().any(dim=1)] = 1e-2
        generator = unscaled_generator * length.unsqueeze(1)
        padding1 = torch.zeros((self.num_envs, 3, 2))
        padding2 = torch.zeros((self.num_envs, 1, 2))
        return torch.cat((padding1, generator, padding2), dim=1)
