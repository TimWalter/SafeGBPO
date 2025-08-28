from typing import Optional, Any

import torch
from torch import Tensor
from beartype import beartype
from PIL import Image, ImageDraw
from jaxtyping import jaxtyped, Float, Bool
from torchvision.transforms.functional import to_tensor

import src.sets as sets
from envs.simulators.interfaces.simulator import Simulator


class SeekerEnv(Simulator):
    """
    2D environment where we try to reach a goal.

    ## Action Set
    | Num |   Action   |  Min | Max |
    |-----|------------|------|-----|
    | 1   | Velocity X | -1.0 | 1.0 |
    | 1   | Velocity Y | -1.0 | 1.0 |

    ## State Set
    | Num | State                    | Min | Max |
    |-----|--------------------------|-----|-----|
    | 0   | Horizontal Position      | -8  | 8   |
    | 1   | Vertical Position        | -8  | 8   |

    ## Observation Set
    | Num | Observation              | Min | Max |
    |-----|--------------------------|-----|-----|
    | 0   | Horizontal Position      | -8  | 8   |
    | 1   | Vertical Position        | -8  | 8   |
    | 1   | Goal Horizontal Position | -8  | 8   |
    | 1   | Goal Vertical Position   | -8  | 8   |
    """
    DT: float = 0.1

    SCREEN_WIDTH = 500
    SCREEN_HEIGHT = 500

    @jaxtyped(typechecker=beartype)
    def __init__(self, num_envs: int = 1, num_steps: int = 1000,
                 additional_observation_set: sets.AxisAlignedBox | None = None,
                 ):
        """
        Initialize the Quadrotor environment.

        Args:
            num_envs: Number of environments to vectorize.
            num_steps: Number of steps till the episode ends.
            additional_observation_set: Additional observation set to append to the observation set.
        """
        state_set = sets.AxisAlignedBox(
            torch.zeros(num_envs, 2),
            torch.diag_embed(torch.tensor([8.0, 8.0]).repeat(num_envs, 1))
        )
        noise_set = sets.AxisAlignedBox(
            torch.zeros(num_envs, 2),
            torch.diag_embed(torch.tensor([0.05, 0.05]).repeat(num_envs, 1))
        )
        observation_set = sets.AxisAlignedBox(
            torch.zeros(num_envs, 4),
            torch.diag_embed(
                torch.tensor([8.0, 8.0, 8.0, 8.0]).repeat(num_envs, 1))
        )
        if additional_observation_set is not None:
            padding1 = torch.zeros(num_envs, additional_observation_set.center.shape[1],
                                   observation_set.generator.shape[2])
            padding2 = torch.zeros(num_envs, observation_set.center.shape[1],
                                   additional_observation_set.generator.shape[2])
            observation_set.center = torch.cat([observation_set.center, additional_observation_set.center], dim=1)
            observation_set.generator = torch.cat([
                torch.cat([observation_set.generator, padding1], dim=1),
                torch.cat([padding2, additional_observation_set.generator], dim=1)
            ], dim=2)

        super().__init__(2, state_set, noise_set, observation_set, num_envs)

        self.num_steps = num_steps

        self.goal = torch.empty((num_envs, 2))

    @jaxtyped(typechecker=beartype)
    def reset(self, seed: Optional[int] = None) -> tuple[
        Float[Tensor, "{self.num_envs} {self.obs_dim}"],
        dict[str, Any]
    ]:
        """Reset all parallel environments and return a batch of initial observations
        and info.

        Args:
            seed: The environment reset seeds

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        super().reset(seed)
        self.goal = self.state[:, 0:2].clone()

        return self.observation, {}

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[Tensor, "{self.num_envs} {self.obs_dim}"]:
        """
        Get the current observation of the environment.

        Returns:
            The current observation of the environment as a batch of observations.
        """
        return torch.cat([self.state, self.goal], dim=1)

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
        return torch.zeros(self.num_envs)

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
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = (self.steps >= self.num_steps) * torch.ones(self.num_envs, dtype=torch.bool)
        return terminated, truncated

    @jaxtyped(typechecker=beartype)
    def unbatched_dynamics(self,
                           state: Float[Tensor, "{self.state_dim}"],
                           action: Float[Tensor, "{self.action_dim}"],
                           noise: Float[Tensor, "{self.state_dim}"]) \
            -> Float[Tensor, "{self.state_dim}"]:
        """
        Unbatched dynamics function that computes the next state given the current state,
        action, and noise. We batch this function automatically using torch.vmap for vectorized execution.

        Args:
            state: Current state of the environment.
            action: Action to execute in the environment.
            noise: Noise sample to perturb the dynamics.

        Returns:
            Next state.
        """
        return state + self.DT * action + noise

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

            frames.append((to_tensor(img) * 255).to(torch.uint8))

        return frames

    @jaxtyped(typechecker=beartype)
    def draw_seeker(self, state: Float[Tensor, "{self.state_dim}"], goal: Float[Tensor, "2"]) \
            -> tuple[Image.Image, ImageDraw.ImageDraw]:
        """
        Draw the quadrotor and the goal on a blank image. Outsourced for reusability.

        Args:
            state: Current state of the quadrotor.
            goal: Goal position.

        Returns:
            img: The image with the quadrotor and goal drawn on it.
            draw: The ImageDraw object used to draw on the image.
        """
        img = Image.new("RGBA", (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), "white")
        draw = ImageDraw.Draw(img)

        world_bound = 8.5
        scale = self.SCREEN_WIDTH / (world_bound * 2)
        offset_x = self.SCREEN_WIDTH / 2
        offset_y = self.SCREEN_HEIGHT / 2

        for j in range(int(-world_bound), int(world_bound) + 1):
            line_x = j * scale + offset_x
            draw.line([(line_x, 0), (line_x, self.SCREEN_HEIGHT)], fill=(230, 230, 230))
            line_y = j * scale + offset_y
            draw.line([(0, line_y), (self.SCREEN_WIDTH, line_y)], fill=(230, 230, 230))

        draw.line([(offset_x, 0), (offset_x, self.SCREEN_HEIGHT)], fill=(150, 150, 150))
        draw.line([(0, offset_y), (self.SCREEN_WIDTH, offset_y)], fill=(150, 150, 150))

        agent_x, agent_y = state
        screen_agent_x = agent_x * scale + offset_x
        screen_agent_y = -agent_y * scale + offset_y

        goal_x, goal_y = goal
        screen_goal_x = goal_x * scale + offset_x
        screen_goal_y = -goal_y * scale + offset_y

        radius = 0.1 * scale

        draw.ellipse(
            (screen_goal_x - radius, screen_goal_y - radius, screen_goal_x + radius, screen_goal_y + radius),
            fill=(0, 255, 0), outline="black"
        )

        draw.ellipse(
            (screen_agent_x - radius, screen_agent_y - radius, screen_agent_x + radius, screen_agent_y + radius),
            fill=(255, 0, 0), outline="black"
        )

        return img, draw
