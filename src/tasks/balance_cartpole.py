import os
from typing import Literal, Optional, Any

import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor

from tasks.envs.cartpole import CartPoleEnv
from tasks.interfaces.rci_task import RCITask


class BalanceCartPoleTask(CartPoleEnv, RCITask):
    """
    ## Description

    The pendulum is placed upright on the cart and the goal is
    to balance the pole by applying forces in the left and right direction on the cart.

    ## Rewards Since the goal is to keep the pole stable & upright for as long as
    possible, the reward is punishing for:
    - Moving the cart away from the center (Default: -0.05*x^2)
    - Moving the cart too fast (Default: -0.1*x_dot^2)
    - Tilting the pole away from the center (Default: -1.0*theta^2)
    - Tilting the pole too fast (Default: -0.1*theta_dot^2)
    - Applying force to the cart (Default: -0.0*action^2)

    ## Starting State

    The starting state is sampled uniformly from the RCI set.

    ## Episode Truncation

    The episode truncates at 240 time steps.
    """

    cart_position_penalty: float = 0.05
    cart_velocity_penalty: float = 0.1
    pole_angle_penalty: float = 1.0
    pole_velocity_penalty: float = 0.1
    cart_action_penalty: float = 0.0

    def __init__(self,
                 rci_size: int = 4,
                 device: Literal["cpu", "cuda"] = "cpu",
                 num_envs: int = 1,
                 stochastic: bool = False,
                 max_episode_steps: int = 240,
                 render_mode: Optional[str] = None
                 ):
        path = os.path.join(os.path.dirname(__file__), "..", "assets", "cartpole_")
        RCITask.__init__(self, device, num_envs, rci_size, path)
        CartPoleEnv.__init__(self, device, num_envs, False, render_mode, max_episode_steps)

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
        self.state= self.rci.sample()

        self.steps= torch.zeros_like(self.steps)

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
        x, theta, x_dot, theta_dot = self.state.split(1, dim=1)
        return (- self.cart_position_penalty * x ** 2
                - self.cart_velocity_penalty * x_dot ** 2
                - self.pole_angle_penalty * theta ** 2
                - self.pole_velocity_penalty * theta_dot ** 2
                - self.cart_action_penalty * action ** 2).squeeze(dim=1)
