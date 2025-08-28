from pathlib import Path

import torch
from torch import Tensor
from beartype import beartype
from PIL import Image, ImageDraw
from jaxtyping import jaxtyped, Float, Bool
from torchvision.transforms.functional import to_tensor

import src.sets as sets
from envs.simulators.interfaces.simulator import Simulator


class PendulumEnv(Simulator):
    r"""
    The environment consists of a pendulum attached at one end to a fixed point, and
    the other end being free.

    The diagram below specifies the coordinate system used for the implementation of the
    pendulum's dynamic equations.

    ![Pendulum Coordinate System](/src/assets/pendulum.png)

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Set
    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -1.0 | 1.0 |

    ## State Set
    | Num | State       | Min  | Max |
    |-----|-------------|------|-----|
    | 0   | Theta       | -pi  | pi  |
    | 1   | \dot{Theta} | -8   |  8  |

    ## Observation Set
    | Num | Observation | Min  | Max |
    |-----|-------------|------|-----|
    | 0   | sin(theta)  | -1.0 | 1.0 |
    | 1   | cos(theta)  | -1.0 | 1.0 |
    | 2   | \dot{Theta} | -8   | 8   |
    """
    DT: float = 0.05
    GRAVITY: float = 9.81
    LENGTH: float = 1
    MASS: float = 1
    TORQUE_MAG: float = 30.0

    SCREEN_WIDTH: int = 500
    SCREEN_HEIGHT: int = 500
    CLOCKWISE_IMG = Image.open(Path(__file__).parent.parent.parent / "assets" / "clockwise.png").convert("RGBA")

    @jaxtyped(typechecker=beartype)
    def __init__(self, num_envs: int, num_steps: int):
        """
        Initialize the Pendulum environment.

        Args:
            num_envs: Number of environments to vectorize.
            num_steps: Number of steps till the episode ends.
        """
        state_set = sets.AxisAlignedBox(
            torch.zeros(num_envs, 2),
            torch.diag_embed(torch.tensor([torch.pi, 8.0]).repeat(num_envs, 1))
        )
        noise_set = sets.AxisAlignedBox(
            torch.zeros(num_envs, 2),
            torch.diag_embed(torch.tensor([0.0, 0.1]).repeat(num_envs, 1))
        )
        observation_set = sets.AxisAlignedBox(
            torch.zeros(num_envs, 3),
            torch.diag_embed(torch.tensor([1.0, 1.0, 8.0]).repeat(num_envs, 1))
        )
        super().__init__(1, state_set, noise_set, observation_set, num_envs)

        self.num_steps = num_steps

        self.last_torque = None

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[Tensor, "{self.num_envs} {self.obs_dim}"]:
        """
        Get the current observation of the environment.

        Returns:
            The current observation of the environment as a batch of observations.
        """
        return torch.cat([
            torch.sin(self.state[:, 0:1]),
            torch.cos(self.state[:, 0:1]),
            self.state[:, 1:2]
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
    def execute_action(self, action: Float[Tensor, "{self.num_envs} {self.action_dim}"]):
        """
        Execute the action in the environment by updating the state.

        Args:
            action: Action to execute in the environment.
        """
        self.last_torque = action[:, 0].clone().detach() * self.TORQUE_MAG
        super().execute_action(action)

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
        theta, theta_dot = state.split(1)
        torque = self.TORQUE_MAG * action

        gravitational_force = torch.sin(theta) * 1.5 * self.GRAVITY / self.LENGTH
        driving_force = torque * 3.0 / self.MASS / self.LENGTH ** 2
        theta_ddot = gravitational_force + driving_force + noise[1:2]

        theta_dot = theta_dot + self.DT * theta_ddot
        theta = theta + self.DT * theta_dot

        theta = torch.atan2(torch.sin(theta), torch.cos(theta))
        return torch.cat([theta, theta_dot])

    @jaxtyped(typechecker=beartype)
    def render(self) -> list[Tensor]:
        """
        Render all environments.

        Returns:
            A list of rendered frames for each environment.
        """
        state = self.state.detach().cpu()
        last_torque = self.last_torque.detach().cpu() if self.last_torque is not None else None

        frames = []
        for i in range(self.num_envs):
            img = Image.new("RGB", (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), "white")
            draw = ImageDraw.Draw(img, "RGBA")

            bound = 2.2
            scale = self.SCREEN_WIDTH / (bound * 2)
            offset = self.SCREEN_WIDTH // 2

            pole_length = self.LENGTH * scale
            pole_width = 0.1 * scale
            theta = state[i, 0]

            rotation_angle = theta + torch.pi
            cos_a, sin_a = torch.cos(rotation_angle), torch.sin(rotation_angle)

            l, r, t, b = -pole_width / 2, pole_width / 2, 0, pole_length
            coords = torch.tensor([(l, t), (r, t), (r, b), (l, b)])

            rotated_coords = coords @ torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
            transformed_coords = rotated_coords + torch.tensor([offset, offset])

            draw.polygon(transformed_coords.flatten().tolist(), fill=(204, 77, 77))

            axle_radius = 0.05 * scale
            draw.ellipse(
                (offset - axle_radius, offset - axle_radius, offset + axle_radius, offset + axle_radius),
                fill="black"
            )

            if last_torque is not None:
                torque_scale = torch.abs(last_torque[i]) / self.TORQUE_MAG
                img_size = int(scale * torque_scale * 0.8)
                if img_size > 10:
                    scaled_img = self.CLOCKWISE_IMG.resize((img_size, img_size), Image.Resampling.LANCZOS)
                    if last_torque[i] > 0:
                        scaled_img = scaled_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    paste_pos = (offset - img_size // 2, offset - img_size // 2)
                    img.paste(scaled_img, paste_pos, scaled_img)

            frames.append((to_tensor(img) * 255).to(torch.uint8))

        return frames
