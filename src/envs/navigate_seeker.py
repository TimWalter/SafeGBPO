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
from envs.simulators.seeker import SeekerEnv
from envs.interfaces.safe_action_env import SafeActionEnv
from src.learning_algorithms.components.coupled_tensor import CoupledTensor


class NavigateSeekerEnv(SeekerEnv, SafeActionEnv):
    """
    The seeker has to navigate through a set of obstacles to reach a goal position.

    ## Reward
    Since the goal is to reach the goal position, the reward punishes for:
    - Distance to the goal position
    - Collision with obstacles
    - The reward is positive if the goal is reached.

    ## Starting State
    The starting state and goal position are sampled uniformly from the state set.
    The obstacles are sampled uniformly, ensuring that they do not overlap.
    The first obstacle is placed between the start and goal position.

    ## Safety
    The safe action set is computed such that collisions with obstacles are avoided.
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
                 draw_safe_action_set: bool
                 ):
        """
        Initialize the NavigateSeekerEnv.

        Args:
            num_envs: Number of environments to vectorize.
            num_steps: Number of steps till the episode ends.
            num_obstacles: Number of obstacles in the environment.
            min_radius: Minimum radius of the obstacles.
            max_radius: Maximum radius of the obstacles.
            draw_safe_action_set: Whether to draw the safe action set in the environment.
        """
        SafeActionEnv.__init__(self, num_action_gens=2)

        self.additional_observation_set = sets.AxisAlignedBox(
            torch.tensor([0.0, 0.0, min_radius + (max_radius - min_radius) / 2] * num_obstacles).unsqueeze(0).repeat(
                num_envs, 1),
            torch.diag_embed(
                torch.tensor([8.0, 8.0, (max_radius - min_radius) / 2] * num_obstacles).repeat(num_envs, 1))
        )
        SeekerEnv.__init__(self, num_envs, num_steps, self.additional_observation_set)

        self.num_obstacles = num_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.draw_safe_action_set = draw_safe_action_set

        self.obstacles: list[sets.Ball] = [sets.Ball(torch.empty((num_envs, 2)), torch.empty(num_envs)) for _ in
                                           range(num_obstacles)]
        self.obstacle_centers = CoupledTensor(num_obstacles, self.num_envs, 2)
        self.obstacle_radii = CoupledTensor(num_obstacles, self.num_envs)

        self.collided = torch.zeros((self.num_envs, num_obstacles), dtype=torch.bool)
        self.initial_goal_distance = torch.zeros(self.num_envs)
        self.reached = torch.zeros(self.num_envs, dtype=torch.bool)

        self.generator_layer = None
        self.last_safe_action_set: sets.Zonotope = sets.Zonotope(torch.zeros(self.num_envs, self.state_dim),
                                                                 torch.zeros(self.num_envs, self.state_dim,
                                                                             self.num_action_gens))

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

        self.obstacles[i].center[obstructing, :] = center[obstructing, :]
        self.obstacles[i].radius[obstructing] = radius[obstructing]

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
    def reward(self,
               action: Float[Tensor, "{self.num_envs} {self.action_dim}"]) \
            -> Float[Tensor, "{self.num_envs}"]:
        """
        Compute the reward for the given action.

        Args:
            action: Action executed in the environment.

        Returns:
            Reward.
        """
        goal_distance = torch.norm(self.goal - self.state, dim=1)
        loss = -self.DISTANCE_PENALTY * goal_distance / self.initial_goal_distance
        for obs in self.obstacles:
            signed_distance = torch.norm(obs.center - self.state, dim=1) - obs.radius
            threshold = obs.radius * 1.1
            zero = torch.zeros_like(loss)
            obstacle_loss = torch.max(zero, (torch.exp(-signed_distance / threshold) - 1 / torch.e) / (1 - 1 / torch.e))
            loss = loss - obstacle_loss * self.COLLISION_PENALTY
        return loss

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
            collided[:, i] = self.obstacles[i].contains(state)
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
        remaining_time = (1 - t) * self.DT

        collided_pos = self.state.clone()
        collided_pos[state_mask] = intersection + reflected_vel * remaining_time

        collided_state = torch.where(state_mask.unsqueeze(1), collided_pos, self.state)

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
            img, draw = self.draw_seeker(state[i], goal[i])

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

            if self.draw_safe_action_set:
                if self.last_safe_action_set.generator.sum() == 0:
                    self.safe_action_set()

                overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)

                draw_set = sets.Zonotope(self.last_safe_action_set.center[i:i + 1, :] + self.state[i:i + 1, :],
                                         self.last_safe_action_set.generator[i:i + 1, :, :])
                vertices = draw_set.vertices().cpu().numpy()

                screen_vertices = [
                    (v[0] * scale + offset_x, -v[1] * scale + offset_y)
                    for v in vertices.T
                ]
                color = (255, 0, 0, 64)
                overlay_draw.polygon(screen_vertices, fill=color)

                img = Image.alpha_composite(img, overlay)

            frames.append((to_tensor(img) * 255).to(torch.uint8))

        # invalidate the cached safe state set
        self.last_safe_action_set.generator *= 0.0

        return frames

    @jaxtyped(typechecker=beartype)
    def safe_action_set(self) -> sets.Zonotope:
        """
        Get the safe action set for the current state.

        Returns:
            A convex set representing the safe action set.

        Note:
            Cache the result if it is expensive to compute.
        """
        with torch.no_grad():
            generator = self.compute_generator()
            self.last_safe_action_set = sets.Zonotope(self.action_set.center, generator)

            return self.last_safe_action_set

    @jaxtyped(typechecker=beartype)
    def compute_generator(self) -> Float[Tensor, "{self.num_envs} {self.state_dim} {self.state_dim}"]:
        """
        Compute the generator for the safe action set, such that they are orthogonal to the closest obstacle
        and ensure feasibility and collision avoidance.


        Returns:
            The generator for the safe action set.
        """
        if self.generator_layer is None:
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
                # State Feasibility
                self.state_set.min[0, :2].cpu().numpy() <= center - cp.abs(generator).sum(axis=1),
                self.state_set.max[0, :2].cpu().numpy() >= center + cp.abs(generator).sum(axis=1),
                # Action Feasibility
                length <= np.ones(2),
            ]
            for i in range(self.num_obstacles):
                # Collision avoidance
                support_pos = cp.sum(cp.abs(parameters[2 + i * 2] @ cp.diag(length)))
                constraints += [support_pos <= parameters[2 + i * 2 + 1]]

            problem = cp.Problem(objective, constraints)
            self.generator_layer = CvxpyLayer(problem, parameters=parameters, variables=[length])

        unscaled_generator = torch.diag_embed(torch.ones((self.num_envs, self.state_dim)))
        min_distance = (self.state.abs() - self.state_set.max[:, :2]).norm(dim=1, keepdim=True)
        for i, ball in enumerate(self.obstacles):
            direction = ball.center - self.state
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
        parameters = [self.state, unscaled_generator]
        for i, ball in enumerate(self.obstacles):
            direction = ball.center - self.state
            distance = torch.norm(direction, dim=1, keepdim=True)
            direction = direction / distance
            distance -= ball.radius.unsqueeze(1)
            parameters += [torch.sum(direction.unsqueeze(2) * unscaled_generator, dim=1), distance]

        length = self.generator_layer(*parameters, solver_args={"solve_method": "Clarabel"})[0]
        return unscaled_generator * length.unsqueeze(1)
