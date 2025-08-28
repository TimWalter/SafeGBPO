import torch
from torch import Tensor
from beartype import beartype
from PIL import Image, ImageDraw
from jaxtyping import jaxtyped, Float, Bool
from torchvision.transforms.functional import to_tensor

import src.sets as sets
from envs.simulators.interfaces.simulator import Simulator


class CartPoleEnv(Simulator):
    r"""
    A pole is attached by an un-actuated joint to a cart, which moves along a
    frictionless track.

    ## Action Set
    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Force | -1.0 | 1.0 |

    ## State Set
    | Num | State          | Min  | Max |
    |-----|----------------|------|-----|
    | 0   | Position       | -10  | 10  |
    | 1   | Theta          | -pi  | pi  |
    | 2   | \dot{Position} | -10  | 10  |
    | 3   | \dot{Theta}    | -10  | 10  |

    ## Observation Set
    | Num | Observation    | Min  | Max |
    |-----|----------------|------|-----|
    | 0   | Position       | -10  | 10  |
    | 1   | sin(theta)     | -1.0 | 1.0 |
    | 2   | cos(theta)     | -1.0 | 1.0 |
    | 2   | \dot{Position} | -10  | 10  |
    | 3   | \dot{Theta}    | -10  | 10  |
    """
    DT: float = 0.02
    GRAVITY: float = 9.8
    HALF_LENGTH: float = 0.5
    MASS_CART: float = 1.0
    MASS_POLE: float = 0.1
    FORCE_MAG: float = 10.0

    SCREEN_WIDTH: int = 1200
    SCREEN_HEIGHT: int = 200

    @jaxtyped(typechecker=beartype)
    def __init__(self, num_envs: int, num_steps: int):
        """
        Initialize the Cartpole environment.

        Args:
            num_envs: Number of environments to vectorize.
            num_steps: Number of steps till the episode ends.
        """
        state_set = sets.AxisAlignedBox(
            torch.zeros(num_envs, 4),
            torch.diag_embed(torch.tensor([10.0, torch.pi, 10.0, 10.0]).repeat(num_envs, 1))
        )
        noise_set = sets.AxisAlignedBox(
            torch.zeros(num_envs, 4),
            torch.diag_embed(torch.tensor([0.0, 0.0, 0.0, 0.1]).repeat(num_envs, 1))
        )
        observation_set = sets.AxisAlignedBox(
            torch.zeros(num_envs, 5),
            torch.diag_embed(torch.tensor([10.0, 1.0, 1.0, 10.0, 10.0]).repeat(num_envs, 1))
        )
        super().__init__(1, state_set, noise_set, observation_set, num_envs)

        self.num_steps = num_steps

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[Tensor, "{self.num_envs} {self.obs_dim}"]:
        """
        Get the current observation of the environment.

        Returns:
            The current observation of the environment as a batch of observations.
        """
        return torch.cat([
            self.state[:, 0:1],
            torch.sin(self.state[:, 1:2]),
            torch.cos(self.state[:, 1:2]),
            self.state[:, 2:4]
        ], dim=1)

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
        x, theta, x_dot, theta_dot = state.split(1)
        force = self.FORCE_MAG * action

        temp = 1 / (self.MASS_POLE + self.MASS_CART) * (
            (force + self.MASS_POLE * self.HALF_LENGTH * torch.square(theta_dot) * torch.sin(theta))
        )
        theta_ddot = 1 / self.HALF_LENGTH * (
                (self.GRAVITY * torch.sin(theta) + temp * torch.cos(theta)) /
                (4.0 / 3.0 - self.MASS_POLE / (self.MASS_POLE + self.MASS_CART) * torch.square(
                    torch.cos(theta))) + noise[0:1]
        )
        x_ddot = temp - self.MASS_POLE * self.HALF_LENGTH / (self.MASS_POLE + self.MASS_CART) * torch.cos(
            theta) * theta_ddot

        x_dot = x_dot + self.DT * x_ddot
        theta_dot = theta_dot + self.DT * theta_ddot
        x = x + self.DT * x_dot
        theta = theta + self.DT * theta_dot

        theta = torch.atan2(torch.sin(theta), torch.cos(theta))
        return torch.cat([x, theta, x_dot, theta_dot], dim=0)

    @jaxtyped(typechecker=beartype)
    def render(self) -> list[Tensor]:
        """
        Render all environments.

        Returns:
            A list of rendered frames for each environment.
        """
        state = self.state.detach().cpu()

        frames = []
        for i in range(self.num_envs):
            img = Image.new("RGB", (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), "white")
            draw = ImageDraw.Draw(img)

            world_width = self.state_set.generator[i, 0, 0].item() * 2.2
            scale = self.SCREEN_WIDTH / world_width
            cart_y = self.SCREEN_HEIGHT / 2  # Vertical center of the screen

            draw.line([(0, cart_y), (self.SCREEN_WIDTH, cart_y)], fill="black")

            x, theta, _, _ = state[i]

            cart_x = x * scale + self.SCREEN_WIDTH / 2.0
            cart_width = 0.5 * scale
            cart_height = 0.3 * scale

            l, r, t, b = -cart_width / 2, cart_width / 2, cart_height / 2, -cart_height / 2
            cart_coords = [(l + cart_x, t + cart_y), (r + cart_x, t + cart_y),
                           (r + cart_x, b + cart_y), (l + cart_x, b + cart_y)]
            draw.polygon(cart_coords, fill="black")

            pole_length = 2 * self.HALF_LENGTH * scale
            pole_width = 0.1 * scale
            pole_pivot_y = cart_y - cart_height / 4

            l, r, t, b = -pole_width / 2, pole_width / 2, 0, -pole_length
            pole_coords = torch.tensor([(l, b), (l, t), (r, t), (r, b)])

            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

            rotated_pole = pole_coords @ rotation_matrix
            transformed_pole = rotated_pole + torch.tensor([cart_x, pole_pivot_y])
            draw.polygon(transformed_pole.flatten().tolist(), fill=(202, 152, 101))

            axle_radius = pole_width / 2
            draw.ellipse(
                (cart_x - axle_radius, pole_pivot_y - axle_radius,
                 cart_x + axle_radius, pole_pivot_y + axle_radius),
                fill=(129, 132, 203)
            )

            frames.append((to_tensor(img) * 255).to(torch.uint8))

        return frames