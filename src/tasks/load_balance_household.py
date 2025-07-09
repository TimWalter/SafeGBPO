import os
from typing import Literal, Optional, Any

import torch
import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor

from tasks.envs.household import HouseholdEnv
from tasks.interfaces.safe_state_task import SafeStateTask
from sets.zonotope import Zonotope
from tasks.wrapper.boundary_projection import BoundaryProjectionWrapper
from tqdm import tqdm


class LoadBalanceHouseholdTask(HouseholdEnv, SafeStateTask):
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
                 device: Literal["cpu", "cuda"] = "cpu",
                 num_envs: int = 1,
                 stochastic: bool = True,
                 max_episode_steps: int = 240,
                 render_mode: Optional[str] = None
                 ):
        SafeStateTask.__init__(self, device, num_state_gens=3)
        HouseholdEnv.__init__(self, device, num_envs, stochastic, render_mode, max_episode_steps)

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
        state = np.tile(np.array([self.soc_init, self.t_in_init, self.t_ret_init]), self.num_envs).reshape(self.num_envs, -1)
        self.state= torch.tensor(state, dtype=torch.float64, device=self.device)

        self.steps = torch.zeros_like(self.steps)

        self.scheduled_reset = torch.zeros_like(self.scheduled_reset)

        if self.render_mode == "human":
            self.render()

        if seed is not None:
            torch.set_rng_state(rng_state)

        return self.observation, {}

    @jaxtyped(typechecker=beartype)
    def safe_state_set(self):
        soc, t_in, t_ret = self.state.split(1, dim=1)
        t_in = t_in.flatten()
        t_ret = t_ret.flatten()
        # extract state bounds
        soc_l = self.state_bounds[:, 0, 0]
        soc_u = self.state_bounds[:, 0, 1]
        t_in_l = self.state_bounds[:, 1, 0]
        t_in_u = self.state_bounds[:, 1, 1]
        t_ret_l = self.state_bounds[:, 2, 0]
        t_ret_u = self.state_bounds[:, 2, 1]
        # define constants
        c_a = 1 - (self.h_fh + self.h_out) * self.dt/(self.h_out * self.tau)
        c_b = self.h_fh * self.dt / (self.h_out * self.tau)
        c_c = self.h_fh / self.c_w_fh * self.dt
        # get data
        current_step = self.steps.detach().cpu().numpy()[0]
        t_out = self.get_data(self.heatpump_data, current_step, 1, "outside_temp")
        t_out_next = self.get_data(self.heatpump_data, current_step + 1, 1, "outside_temp")
        cop = self.get_data(self.heatpump_data, current_step, 1, "COP")
        # compute intervals
        s_soc_l = soc_l
        s_soc_u = soc_u
        s_t_ret_l = (t_ret_l - t_in * c_c - self.p_hp_min * cop * self.dt / self.c_w_fh) / (1 - c_c)
        s_t_ret_u = (t_ret_u - t_in * c_c - self.p_hp_max * cop * self.dt / self.c_w_fh) / (1 - c_c)
        s_t_in_l = (
            t_in_l 
            - (c_b * cop * self.dt * self.p_hp_min) / self.c_w_fh
            - t_ret * (c_a * c_b + c_b * (1 - c_c)) 
            - t_out * c_a * self.dt / self.tau 
            - t_out_next * self.dt / self.tau
            ) / (c_a**2 + c_b * c_c)
        s_t_in_u = (
            t_in_u 
            - (c_b * cop * self.dt * self.p_hp_max) / self.c_w_fh
            - t_ret * (c_a * c_b + c_b * (1 - c_c)) 
            - t_out * c_a * self.dt / self.tau 
            - t_out_next * self.dt / self.tau
        ) / (c_a**2 + c_b * c_c)

        # check that these bounds are smaller than the feasible bounds
        s_t_in_l = torch.maximum(s_t_in_l, t_in_l)
        s_t_in_u = torch.minimum(s_t_in_u, t_in_u)
        s_t_ret_l = torch.maximum(s_t_ret_l, t_ret_l)
        s_t_ret_u = torch.minimum(s_t_ret_u, t_ret_u)
        # compute center and generators
        soc_length = (s_soc_u - s_soc_l) / 2
        t_in_length = (s_t_in_u - s_t_in_l) / 2
        t_ret_length = (s_t_ret_u - s_t_ret_l) / 2
        s_soc_center = s_soc_l + soc_length
        s_t_ret_center = s_t_ret_l + t_ret_length
        s_t_in_center = s_t_in_l + t_in_length
        
        center = torch.cat([s_soc_center.reshape(1,-1), s_t_in_center.reshape(1,-1), s_t_ret_center.reshape(1,-1)]).transpose(1,0)
        generator = torch.diag_embed(
            torch.cat([soc_length.reshape(1,-1), t_in_length.reshape(1,-1), t_ret_length.reshape(1,-1)]).transpose(1,0)
            )
        safe_state_set = Zonotope(center, generator)
        return safe_state_set

    
    @jaxtyped(typechecker=beartype)
    def reward(self,
               action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]) \
            -> Float[Tensor, "{self.num_envs}"]:
        soc, t_in, t_ret = self.state.split(1, dim=1)
        p_ess, p_hp = action.split(1, dim=1)
        current_step = self.steps.detach().cpu().numpy()[0]
        p_load = self.get_data(self.load_data, current_step, 1, "p")
        p_pv = self.get_data(self.pv_data, current_step, 1, 'p')
        buying_price = self.get_data(self.buying_price, current_step, 1, 'price')
        p_total = p_ess + p_hp + p_load + p_pv
        electricity_cost = torch.where(
            p_total >= 0,
            p_total * self.dt * buying_price,
            p_total * self.dt * self.selling_price
        )
        comfort_cost = (t_in - self.t_in_setpoint)**2 * self.cost_coefficient_hp
        return -(electricity_cost.flatten() + comfort_cost.flatten())



if __name__=="__main__": 
    env = LoadBalanceHouseholdTask(num_envs=2)
    num_steps = 1000000
    safe_env = BoundaryProjectionWrapper(
        env, 
        lin_state = [5.0, 21.0, 25.0], 
        lin_action=[0.0, 0.0], 
        lin_noise=[0.0, 0.0, 0.0]
        )
    state, _ = safe_env.reset()
    for _ in tqdm(range(num_steps)):
        action = torch.tensor(safe_env.action_space.sample())
        safe_action = safe_env.actions(action)
        new_state, reward, terminated, truncated, _ = env.step(action)
        if terminated.any() or truncated.any():
            state, _ = safe_env.reset()
        else:
            state = new_state