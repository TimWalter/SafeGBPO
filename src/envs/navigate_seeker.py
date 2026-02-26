from typing import Optional, Any, Union

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
    DISTANCE_PENALTY: float = 2.0

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 num_envs: int,
                 num_steps: int,
                 num_obstacles: int,
                 min_radius: float,
                 max_radius: float,
                 draw_safe_action_set: bool,
                 safe_action_polytope: bool,
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
        self.safe_action_polytope = safe_action_polytope

        self.obstacles: list[sets.Ball] = [sets.Ball(torch.empty((num_envs, 2)), torch.empty(num_envs)) for _ in
                                           range(num_obstacles)]
        self.obstacle_centers = CoupledTensor(num_obstacles, self.num_envs, 2)
        self.obstacle_radii = CoupledTensor(num_obstacles, self.num_envs)

        self.collided = torch.zeros((self.num_envs, num_obstacles), dtype=torch.bool)
        self.initial_goal_distance = torch.zeros(self.num_envs)
        self.reached = torch.zeros(self.num_envs, dtype=torch.bool)

        self.generator_layer = None

        if not self.safe_action_polytope:
            self.last_safe_action_set: sets.Zonotope = sets.Zonotope(torch.zeros(self.num_envs, self.state_dim),
                                                                     torch.zeros(self.num_envs, self.state_dim,
                                                                                 self.num_action_gens))
        else:
            self.last_safe_action_set: sets.Polytope = sets.Polytope(A=torch.zeros(self.num_envs, self.state_dim, self.state_dim),
                                                                     b=torch.zeros(self.num_envs, self.state_dim))

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
        return -self.DISTANCE_PENALTY * torch.norm(self.goal - self.state, dim=1)

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
        For every action -> render is called.

        Returns:
            A list of rendered frames for each environment.
        """
        state = self.state.detach().cpu()
        goal = self.goal.detach().cpu()

        frames = []

        # Restart the safe set
        if self.draw_safe_action_set:
            self.safe_action_set()

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
                overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                if self.safe_action_polytope:
                    draw_set = sets.Polytope(A = self.last_safe_action_set.A[i].unsqueeze(0),
                                            b = (self.last_safe_action_set.b[i] + torch.matmul(self.last_safe_action_set.A[i], self.state[i])).unsqueeze(0))
                else:
                    draw_set = sets.Zonotope(self.last_safe_action_set.center[i:i + 1, :] + self.state[i:i + 1, :],
                                            self.last_safe_action_set.generator[i:i + 1, :, :])
                
                vertices = draw_set.vertices().cpu().detach().numpy()

                screen_vertices = [
                    (v[0] * scale + offset_x, -v[1] * scale + offset_y)
                    for v in vertices.T
                ]
                
                color = (255, 0, 0, 64)
                overlay_draw.polygon(screen_vertices, fill=color)

                img = Image.alpha_composite(img, overlay)

            frames.append((to_tensor(img) * 255).to(torch.uint8))

        return frames



    @jaxtyped(typechecker=beartype)
    def safe_action_set(self) -> Union[sets.Zonotope, sets.Polytope]:
        """
        Get the safe action set for the current state.

        Returns:
            A convex set representing the safe action set.

        Note:
            Cache the result if it is expensive to compute.
        """
        with torch.no_grad():
            if self.safe_action_polytope:
                A, b = self.compute_polytope_generator()
                self.last_safe_action_set = sets.Polytope(A=A, b=b)
            else:
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

    @jaxtyped(typechecker=beartype)
    def compute_polytope_generator(
        self
    ) -> tuple[Float[Tensor, "batch_dim num_constraints dim"], Float[Tensor, "batch_dim num_constraints"]]:
        """
        Compute the safe input set polytope for a batch of environments based on acorl repo (envs.constraints.seeker.compute_relevant_input_set()).
        Returns A, b for each batch: { x | A x <= b }.
        """
        batch = self.num_envs
        dim = self.state_dim
        asi = self.action_set.generator[0].diag()

        agent_pos = self.state
        noise = self.noise_set.sample()
        
        # Boundary constraints
        boundary_size = (self.state_set.max - self.state_set.min) / 2
        b_upper = boundary_size - agent_pos
        b_lower = boundary_size + agent_pos
        b_boundary = torch.cat([b_upper, b_lower], dim=1)

        I = torch.eye(dim, device=self.state.device)
        A_boundary_single = torch.cat([I, -I], dim=0)
        A_boundary = A_boundary_single.unsqueeze(0).repeat(batch, 1, 1)

        # Action constraints
        b_action = torch.cat([asi, asi], dim=0).unsqueeze(0).repeat(batch, 1)
        A_action_single = torch.cat([-I, I], dim=0)
        A_action = A_action_single.unsqueeze(0).repeat(batch, 1, 1)

        # Combine fixed constraints
        b_fixed = torch.cat([b_boundary, b_action], dim=1)
        A_fixed = torch.cat([A_boundary, A_action], dim=1)

        # Obstacle constraints
        max_obs_constraints = self.obstacle_centers.tensor.shape[0]
        total_constraints = 4*dim + max_obs_constraints

        A_padded = torch.zeros(batch, total_constraints, dim, device=self.state.device)
        b_padded = torch.full((batch, total_constraints), float('inf'), device=self.state.device)

        A_padded[:, :4*dim, :] = A_fixed
        b_padded[:, :4*dim] = b_fixed
        threshold = (torch.sqrt(torch.tensor(dim, device=self.state.device)) * (asi[0] + noise[:, 0]))

        B, D = agent_pos.shape
        centers = self.obstacle_centers.tensor.permute(1, 0, 2)
        radii = self.obstacle_radii.tensor.permute(1, 0)

        # Distance to obstacle
        agent = agent_pos[:, None, :] 
        a = centers - agent
        norm_a = torch.linalg.norm(a, dim=-1, keepdim=True)
        dist = torch.linalg.norm(a, dim=-1)
        a_normalized = torch.zeros_like(a)
        valid = norm_a > 1e-8
        a_normalized[valid.expand_as(a)] = (a / norm_a)[valid.expand_as(a)]

        # Threshold condition
        # far_mask = True means obstacle is too far and should be skipped
        far_mask = (dist - radii) > threshold[:, None]
        near_mask = ~far_mask

        b_val = dist - radii - noise[:, 0:1]

        A_obs = torch.zeros((B, max_obs_constraints, D), device=agent_pos.device, dtype=agent_pos.dtype)
        b_obs = torch.full((B, max_obs_constraints), float('inf'), device=agent_pos.device, dtype=agent_pos.dtype)

        A_obs[near_mask] = a_normalized[near_mask]
        b_obs[near_mask] = b_val[near_mask]

        start = 4 * dim
        end = start + max_obs_constraints
        A_padded[:, start:end, :] = A_obs
        b_padded[:, start:end] = b_obs

        return A_padded, b_padded