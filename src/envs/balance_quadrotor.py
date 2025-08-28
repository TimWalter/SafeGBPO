from pathlib import Path
from typing import Optional, Any

import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor

from envs.simulators.quadrotor import QuadrotorEnv
from envs.interfaces.rci_env import RCIEnv


class BalanceQuadrotorEnv(QuadrotorEnv, RCIEnv):
    """
    The quadrotor starts in a random position and the goal is to
    apply thrust and roll angle to stabilize it in a horizontal position.

    ## Rewards
    Since the goal is to keep the quadrotor close to the reset state,
    the reward punishes for:
    - Deviations from the horizontal starting position
    - Deviations from the vertical starting position
    - Roll angle
    - Horizontal velocity
    - Vertical velocity
    - Roll velocity
    - Total thrust
    - Desired roll angle

    ## Starting State
    The starting state is sampled uniformly from the RCI set.

    ## Safety
    The safety constraints prevent escalation of the position and velocity, such that balancing always remains
    possible within the finite time horizon of the problem. We induce a safe action set from a
    robust control invariant (RCI) state set.
    """
    DIST_PENALTY: float = 2.5
    ROLL_PENALTY: float = 0.1
    X_DOT_PENALTY: float = 0.1
    Z_DOT_PENALTY: float = 0.1
    ROLL_DOT_PENALTY: float = 0.1
    THRUST_PENALTY: float = 0.002
    ROLL_ANGLE_PENALTY: float = 0.001

    def __init__(self, num_envs: int, num_steps: int):
        """
        Initialize the BalanceQuadrotor environment.

        Args:
            num_envs: Number of environments to vectorize.
            num_steps: Number of steps till the episode ends.
        """
        path = Path(__file__).parent.parent / "assets" / "quadrotor_"
        RCIEnv.__init__(self, num_envs, path)
        QuadrotorEnv.__init__(self, num_envs, num_steps)

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
        self.state = self.rci.sample()
        self.goal = self.state[:, 0:2].clone()

        return self.observation, {}

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
        x, z, roll, x_dot, z_dot, roll_dot = self.state.split(1, dim=1)
        thrust, roll_angle = action.split(1, dim=1)
        thrust = thrust * self.THRUST_MAG + self.GRAVITY
        roll_angle = roll_angle * self.ROLL_ANGLE_MAG
        dist = torch.sqrt((x - self.goal[:, 0:1]) ** 2 + (z - self.goal[:, 1:2]) ** 2)
        return (- self.DIST_PENALTY * dist
                - self.ROLL_PENALTY * roll ** 2
                - self.X_DOT_PENALTY * x_dot ** 2
                - self.Z_DOT_PENALTY * z_dot ** 2
                - self.ROLL_DOT_PENALTY * roll_dot ** 2
                - self.THRUST_PENALTY * thrust ** 2
                - self.ROLL_ANGLE_PENALTY * roll_angle ** 2).squeeze(dim=1)
