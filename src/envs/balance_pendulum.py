from pathlib import Path
from typing import Optional, Any

from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor

from envs.simulators.pendulum import PendulumEnv
from envs.interfaces.rci_env import RCIEnv


class BalancePendulumEnv(PendulumEnv, RCIEnv):
    """
    The pendulum starts in a random position and the goal is to
    apply torque on the free end to swing it into an upright position, with its center
    of gravity right above the fixed point.

    ## Rewards
    Since the goal is to keep the pole stable and upright for as long as
    possible, the reward is punishing for:
    - Tilting the pole away from the center
    - Tilting the pole too fast
    - Applying torque

    ## Starting State
    The starting state is sampled uniformly from the RCI set.

    ## Safety
    We define the safety constraints as the part of the state space from which the controller
    can maintain balance, effectively limiting the velocity and angle close to the upright position. We induce a
    safe action set from a robust control invariant (RCI) state set.
    """
    ANGLE_PENALTY: float = 1.0
    VELOCITY_PENALTY: float = 0.1
    ACTION_PENALTY: float = 0.001

    @jaxtyped(typechecker=beartype)
    def __init__(self, num_envs: int, num_steps: int):
        """
        Initialize the BalancePendulum environment.

        Args:
            num_envs: Number of environments to vectorize.
            num_steps: Number of steps till the episode ends.
        """
        path = Path(__file__).parent.parent / "assets" / "pendulum_"
        RCIEnv.__init__(self, num_envs, path)
        PendulumEnv.__init__(self, num_envs, num_steps)

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
        self.state= self.rci.sample()

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
        theta, theta_dot = self.state.split(1, dim=1)
        return (-self.ANGLE_PENALTY * theta ** 2
                - self.VELOCITY_PENALTY * theta_dot ** 2
                - self.ACTION_PENALTY * action ** 2).squeeze(dim=1)
