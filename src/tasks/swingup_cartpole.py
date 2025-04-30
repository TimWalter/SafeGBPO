from typing import Literal, Optional, Any

import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor

import src.sets as sets
from tasks.envs.cartpole import CartPoleEnv
from tasks.interfaces.safe_action_task import SafeActionTask


class SwingUpCartPoleTask(CartPoleEnv, SafeActionTask):
    """
    ## Description
    The pendulum is placed downright initially and has to be
    swung up before balancing starts.

    ## Starting State
    The pole angle is in the range `(-π, -π+0.05)` or `(π-0.05, π)`.

    ## Episode Truncation

    The episode truncates at 240 time steps.
    """

    cart_position_penalty: float = 0.05
    cart_velocity_penalty: float = 0.1
    pole_angle_penalty: float = 1.0
    pole_velocity_penalty: float = 0.1
    cart_action_penalty: float = 0.0

    def __init__(self,
                 device: Literal["cpu", "cuda"] = "cpu",
                 num_envs: int = 1,
                 stochastic: bool = False,
                 max_episode_steps: int = 240,
                 render_mode: Optional[str] = None
                 ):
        SafeActionTask.__init__(self, device, 1)
        CartPoleEnv.__init__(self, device, num_envs, stochastic, render_mode,
                             max_episode_steps)

    @jaxtyped(typechecker=beartype)
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> tuple[
        Float[Tensor, "{self.num_envs} {self.observation_space.shape[1]}"],
        dict[str, Any]]:
        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)

        self.state = (self._rand(4) * 0.1 - 0.05)
        self.state[:, 2:3] = torch.pi - self._rand(1) * 0.1 + 0.05
        self.state[:, 2:3] = torch.atan2(torch.sin(self.state[:, 2:3]),
                                         torch.cos(self.state[:, 2:3]))

        self.steps = torch.zeros_like(self.steps)

        self.scheduled_reset = torch.zeros_like(self.scheduled_reset)

        if self.render_mode == "human":
            self.render()

        if seed is not None:
            torch.set_rng_state(rng_state)

        return self.observation, {}

    @jaxtyped(typechecker=beartype)
    def reward(self, action: Tensor) -> Tensor:
        x, theta, x_dot, theta_dot = self.state.split(1, dim=1)
        return (- self.cart_position_penalty * x ** 2
                - self.cart_velocity_penalty * x_dot ** 2
                - self.pole_angle_penalty * theta ** 2
                - self.pole_velocity_penalty * theta_dot ** 2
                - self.cart_action_penalty * action ** 2).squeeze(dim=1)

    @jaxtyped(typechecker=beartype)
    def safe_action_set(self) -> sets.Zonotope:
        """
        Get the safe state set for the current state.

        Returns:
            A convex set representing the safe action set.

        Note:
            Cache the result if it is expensive to compute
        """
        return sets.Zonotope(
            torch.zeros((self.num_envs, self.action_space.shape[1]),
                        device=self.device, dtype=torch.float64),
            0.9 * torch.ones((self.num_envs, self.action_space.shape[1], 1),
                             device=self.device, dtype=torch.float64)
        )
