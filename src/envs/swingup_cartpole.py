from typing import Optional, Any

import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor

import src.sets as sets
from envs.simulators.cartpole import CartPoleEnv
from envs.interfaces.safe_state_env import SafeStateEnv


class SwingUpCartPoleEnv(CartPoleEnv, SafeStateEnv):
    """
    The pendulum is placed downright initially and has to be swung up before balancing starts.

    ## Reward
    The reward is a negative quadratic penalty on the cart position, cart velocity, pole angle,
    pole angular velocity, and action.

    ## Starting State
    The starting state is uniformly sampled from the states set, with the exception that the pole angle is set to pi.

    ## Safety
    The safe state set is the feasible state set, to maintain bounds.
    """
    CART_POSITION_PENALTY: float = 0.05
    CART_VELOCITY_PENALTY: float = 0.1
    POLE_ANGLE_PENALTY: float = 1.0
    POLE_VELOCITY_PENALTY: float = 0.1
    CART_ACTION_PENALTY: float = 0.0

    @jaxtyped(typechecker=beartype)
    def __init__(self, num_envs: int, num_steps: int):
        """
        Initialize the SwingUpCartPole environment.

        Args:
            num_envs: Number of environments to vectorize.
            num_steps: Number of steps till the episode ends.
        """
        SafeStateEnv.__init__(self, 4)
        CartPoleEnv.__init__(self, num_envs, num_steps)

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

        self.state = (torch.rand(self.num_envs, 4) * 0.1 - 0.05)
        self.state[:, 1] = torch.pi
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
        x, theta, x_dot, theta_dot = self.state.split(1, dim=1)
        return (- self.CART_POSITION_PENALTY * x ** 2
                - self.CART_VELOCITY_PENALTY * x_dot ** 2
                - self.POLE_ANGLE_PENALTY * theta ** 2
                - self.POLE_VELOCITY_PENALTY * theta_dot ** 2
                - self.CART_ACTION_PENALTY * action ** 2).squeeze(dim=1)

    @jaxtyped(typechecker=beartype)
    def safe_state_set(self) -> sets.Zonotope:
        """
        Get the safe state set for the current state.

        Returns:
            A convex set representing the safe action set.

        Note:
            Cache the result if it is expensive to compute
        """
        return self.state_set
