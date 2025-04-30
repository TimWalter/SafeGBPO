import os
from typing import Literal, Optional, Any

import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor

from tasks.envs.quadrotor import QuadrotorEnv
from tasks.interfaces.rci_task import RCITask


class BalanceQuadrotorTask(QuadrotorEnv, RCITask):
    """
    ## The reward describes a stabilization task where deviations in position from
        reset state are punished.
    - Deviations from the horizontal starting position (Default: -1.0(x - x_0)^2)
    - Deviations from the vertical starting position (Default: -1.0(z - z_0)^2)
    - Roll angle (Default: -0.1roll^2)
    - Horizontal velocity (Default: -0.1x_dot^2)
    - Vertical velocity (Default: -0.1z_dot^2)
    - Roll velocity (Default: -0.1roll_dot^2)
    - Total thrust (Default: -0.002 * action[0]^2)
    - Desired roll angle (Default: -0.001 * action[1]^2)

    ## Starting State

    The starting state is sampled uniformly from the RCI set.

    ## Episode Truncation

    The episode truncates at 1000 time steps.
    """

    dist_penalty: float = 2.5
    roll_penalty: float = 0.1
    x_dot_penalty: float = 0.1
    z_dot_penalty: float = 0.1
    roll_dot_penalty: float = 0.1
    thrust_penalty: float = 0.002
    roll_angle_penalty: float = 0.001

    def __init__(self,
                 rci_size: int = 4,
                 device: Literal["cpu", "cuda"] = "cpu",
                 num_envs: int = 1,
                 stochastic: bool = True,
                 max_episode_steps: int = 1000,
                 render_mode: Optional[str] = None
                 ):
        path = os.path.join(os.path.dirname(__file__), "..", "assets", "quadrotor_")
        RCITask.__init__(self, device, num_envs, rci_size, path)
        QuadrotorEnv.__init__(self, device, num_envs, stochastic, render_mode,
                              max_episode_steps)

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
        self.state = self.rci.sample()
        self.goal_pos = self.state[:, 0:2].clone()

        self.steps = torch.zeros_like(self.steps)

        self.scheduled_reset = torch.zeros_like(self.scheduled_reset)

        if self.render_mode == "human":
            self.render()

        if seed is not None:
            torch.set_rng_state(rng_state)

        return self.observation, {}

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
        thrust, roll_angle = action.split(1, dim=1)
        thrust = thrust * self.thrust_mag + self.gravity
        roll_angle = roll_angle * self.roll_angle_mag
        x, z, roll, x_dot, z_dot, roll_dot = self.state.split(1, dim=1)
        return (- self.dist_penalty * torch.sqrt(
            (x - self.goal_pos[:, 0:1]) ** 2 +
            (z - self.goal_pos[:, 1:2]) ** 2)
                - self.roll_penalty * roll ** 2
                - self.x_dot_penalty * x_dot ** 2
                - self.z_dot_penalty * z_dot ** 2
                - self.roll_dot_penalty * roll_dot ** 2
                - self.thrust_penalty * thrust ** 2
                - self.roll_angle_penalty * roll_angle ** 2).squeeze(dim=1)
