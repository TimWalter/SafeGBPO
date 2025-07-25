import os
from typing import Literal, Optional, Any

import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor

from tasks.envs.pendulum import PendulumEnv
from tasks.interfaces.rci_task import RCITask


class BalancePendulumTask(PendulumEnv, RCITask):
    """
    ## Description

    The inverted pendulum swing up problem is based on the classic problem in control
    theory. The pendulum starts in a random position and the goal is to
    apply torque on the free end to swing it into an upright position, with its center
    of gravity right above the fixed point.

    ## Rewards Since the goal is to keep the pole stable & upright for as long as
    possible, the reward is punishing for:
    - Tilting the pole away from the center (Default: -1.0*theta^2)
    - Tilting the pole too fast (Default: -0.1*theta_dot^2)
    - Applying torque (Default: -0.001*action^2)

    where `theta` is the pendulum's angle normalized between *[-pi, pi]*
    (with 0 being in the upright position). Based on the above equation, the minimum
    reward that can be obtained is *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> +
    0.001 * 2<sup>2</sup>) = -16.2736044*, while the maximum reward is zero
    (pendulum is upright with zero velocity and no torque applied).

    ## Starting State

    The starting state is sampled uniformly from the RCI set.

    ## Episode Truncation

    The episode truncates at 240 time steps.

    """
    angle_penalty: float = 1.0
    velocity_penalty: float = 0.1
    action_penalty: float = 0.001

    def __init__(self,
                 rci_size: int = 4,
                 device: Literal["cpu", "cuda"] = "cpu",
                 num_envs: int = 1,
                 stochastic: bool = True,
                 max_episode_steps: int = 240,
                 render_mode: Optional[str] = None
                 ):
        path = os.path.join(os.path.dirname(__file__), "..", "assets", "pendulum_")
        RCITask.__init__(self, device, num_envs, rci_size, path)
        PendulumEnv.__init__(self, device, num_envs, stochastic, render_mode, max_episode_steps)

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
        theta, theta_dot = self.state.split(1, dim=1)
        return (-self.angle_penalty * theta ** 2
                - self.velocity_penalty * theta_dot ** 2
                - self.action_penalty * action ** 2).squeeze(dim=1)
