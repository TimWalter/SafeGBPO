import os
from typing import Optional, Literal

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from gymnasium import spaces
from jaxtyping import jaxtyped, Float, Bool
from torch import Tensor

import src.sets as sets
from src.tasks.envs.interfaces.torch_vector_env import TorchVectorEnv


class HouseholdEnv(TorchVectorEnv):
    """
    ## Description

    The system consists of a pendulum attached at one end to a fixed point, and
    he other end being free.

    The diagram below specifies the coordinate system used for the implementation of the
    pendulum's dynamic equations.

    ![Pendulum Coordinate System](/src/assets/pendulum.png)

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Space

    The action is a `Tensor` with shape `(num_envs, 1)` which can take values `[-1,
    1]` indicating the torque and direction applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -1.0 | 1.0 |

    ## Observation Space

    The observation is a `Tensor` with shape `(num_envs, 3)` representing the x-y
    coordinates of the pendulum's free end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | sin(theta)       | -1.0 | 1.0 |
    | 1   | cos(theta)       | -1.0 | 1.0 |
    | 2   | Angular Velocity | -inf | inf |

    ## Arguments

    stochastic: bool = True (whether to introduce stochastic dampening)
    max_episode_steps: int = 200 (maximum number of steps in an episode)
    device: Literal["cpu", "cuda"] = "cpu" (device to use for torch tensors)
    render_mode: Optional[str] = None (render mode for the environment)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    # Constants

    dt: float = 1.0
    p_hp_max: float = 5.0  # kW
    p_hp_min: float = 0.0  # kW
    p_ess_max: float = 2.0  # kW
    p_ess_min: float = -2.0  # kW
    t_in_init: float = 20.0  # °C
    t_in_setpoint: float = 21
    t_ret_init: float = 25.0  # °C
    soc_init: float = 5  # kWh
    h_fh: float = 1.1  # kW/K
    h_out: float = 0.26  # kW/K
    tau: float = 240  # h
    c_w_fh: float = 1.1625  # kWh/k
    cost_coefficient_hp: float = 0.1
    selling_price: float = 0.08  # €/kWh 
    start_time: int = 7800  # index of data source that corresponds to our desired start time (7800=November 21st)



    # Dim==state_shape[1] is required for the linearisation, such that the requirements
    # for a full dimensional box are given.
    noise: sets.Box = sets.Box(torch.zeros(1, 3), torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]))

    screen_width = 500
    screen_height = 500

    def __init__(self,
                 device: Literal["cpu", "cuda"] = "cpu",
                 num_envs: int = 1,
                 stochastic: bool = False,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 24,  # always train over one day
                 ):
        high = np.array([10.0, 24.0, 100], dtype=np.float64)
        low = np.array([0.0, 18.0, 10], dtype=np.float64)
        action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
        observation_space = spaces.Box(low, high, dtype=np.float64)
        super().__init__(device, num_envs, observation_space, action_space,
                         [
                             [low[0], high[0]], # State bounds here are the same as observation bounds
                             [low[1], high[1]],
                             [low[2], high[2]]
                         ],
                         stochastic, render_mode)

        self.max_episode_steps = max_episode_steps
        path = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
        self.load_data = pd.read_csv(path + "/ICLR_load.csv", index_col=0)
        self.pv_data = pd.read_csv(path + "/ICLR_pv.csv", index_col=0)
        self.heatpump_data = pd.read_csv(path + "/DE_Temperature_and_COP2016.csv", sep=";", index_col=0)
        time_of_use_prices = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2])
        time_of_use_prices = np.tile(time_of_use_prices, 366)  # use the same price profile for every day of the year
        self.buying_price = pd.DataFrame(time_of_use_prices, columns=["price"])
        self.buying_price.insert(0, "t", self.pv_data.index)
        self.buying_price.set_index(["t"])

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[
        Tensor, "{self.num_envs} {self.observation_space.shape[1]}"]:
        return self.state

    @jaxtyped(typechecker=beartype)
    def dynamics(self,
                 action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]) \
            -> Float[Tensor, "{self.num_envs} {self.state.shape[1]}"]:
        """
        Do not alter the state yet only return the next state given the action"""
        rescaled_action = self.rescale_actions(action)
        p_ess, p_hp = rescaled_action.split(1, dim=1)
        soc, t_in, t_ret = self.state.split(1, dim=1)
        t_out = self.get_data(self.heatpump_data, 1, "outside_temp")
        cop = self.get_data(self.heatpump_data, 1, "COP")
        new_soc = soc + p_ess * self.dt
        new_t_in = t_in + self.dt * (
            -(self.h_fh + self.h_out)/(self.h_out * self.tau) * t_in 
            + self.h_fh/(self.h_out * self.tau) * t_ret 
            + t_out/self.tau
            )
        new_t_ret = t_ret + self.dt * (
            self.h_fh/self.c_w_fh * t_in 
            - self.h_fh/self.c_w_fh * t_ret
            + p_hp * cop/self.c_w_fh
        )
        return torch.cat([new_soc, new_t_in, new_t_ret], dim=1)
    
    @jaxtyped(typechecker=beartype)
    def rescale_actions(self, action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]) \
            -> Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]:
        """
        Rescale the normalized actions from [-1, 1] to their original ranges.
        
        Args:
            action: Normalized actions in range [-1, 1]
            
        Returns:
            Rescaled actions in their original ranges:
            - action[:,0]: ESS power in [p_ess_min, p_ess_max]
            - action[:,1]: HP power in [p_hp_min, p_hp_max]
        """
        p_ess = 0.5 * (action[:, 0:1] + 1.0) * (self.p_ess_max - self.p_ess_min) + self.p_ess_min
        p_hp = 0.5 * (action[:, 1:2] + 1.0) * (self.p_hp_max - self.p_hp_min) + self.p_hp_min
        return torch.cat([p_ess, p_hp], dim=1)

    @jaxtyped(typechecker=beartype)
    def get_data(
        self,
        data_source: pd.DataFrame,
        n_timesteps: int,
        key: str
    ) -> Float[Tensor, "{n_timesteps}"]:
        current_step = self.steps.detach().cpu().numpy()[0]
        start = self.start_time + current_step
        end = self.start_time + current_step + n_timesteps
        data = data_source[key][start:end].to_numpy()
        return torch.tensor(data, dtype=torch.float64, device=self.device)

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
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)

    @jaxtyped(typechecker=beartype)
    def episode_ending(self) -> tuple[
        Bool[Tensor, "{self.num_envs}"],
        Bool[Tensor, "{self.num_envs}"],
    ]:
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        truncated = self.steps >= self.max_episode_steps
        return terminated, truncated

    # @jaxtyped(typechecker=beartype)
    # def linear_dynamics(self,
    #                     lin_state: Float[Tensor, "{self.state.shape[1]}"],
    #                     lin_action: Float[Tensor, "{self.action_space.shape[1]}"],
    #                     lin_noise: Float[Tensor, "{self.noise.center.shape[1]}"]
    #                     ) -> tuple[
    #     Float[Tensor, "{self.state.shape[1]}"],
    #     Float[Tensor, "{self.state.shape[1]} {self.state.shape[1]}"],
    #     Float[Tensor, "{self.state.shape[1]} {self.action_space.shape[1]}"],
    #     Float[Tensor, "{self.state.shape[1]} {self.noise.center.shape[1]}"]
    # ]:
    #     """
    #     Compute the linearised dynamics around the given state, action and noise by
    #     computing a first order taylor series of the update.

    #     Args:
    #         lin_state: Linearisation point for the state.
    #         lin_action: Linearisation point for the action.
    #         lin_noise: Linearisation point for the noise.

    #     Returns:
    #         constant_matrix: The constant matrix in the linear dynamics.
    #         state_matrix: The state matrix in the linear dynamics.
    #         action_matrix: The action matrix in the linear dynamics.
    #         noise_matrix: The noise matrix in the linear dynamics.
    #     """

    #     constant_mat = torch.tensor([
    #         lin_state[0] + self.dt * (lin_state[1] + self.dt * theta_ddot),
    #         lin_state[1] + self.dt * theta_ddot
    #     ], dtype=torch.float64, device=self.device)

    #     state_mat = torch.eye(2, dtype=torch.float64, device=self.device)

    #     action_mat = torch.zeros((2, 1), dtype=torch.float64, device=self.device)

    #     noise_mat = torch.zeros((self.state.shape[1], self.noise.center.shape[1]),
    #                             dtype=torch.float64, device=self.device)

    #     return constant_mat, state_mat, action_mat, noise_mat

    # @jaxtyped(typechecker=beartype)
    # def reachable_set(self) -> sets.Zonotope:
    #     """
    #     Compute the one step reachable set.

    #     Returns:
    #         The one step reachable set.
    #     """
    #     ## @Hannah same as action_mat in the linear dynamics just already vectorised
    #     center = self.dynamics(self.action_set.center)

    #     d_driving_force_d_action = 3.0 / self.mass / self.length ** 2 * self.torque_mag
    #     d_theta_ddot_d_action = d_driving_force_d_action

    #     action_mat = torch.zeros((*self.state.shape, self.action_space.shape[1]),
    #                              dtype=torch.float64, device=self.device)
    #     action_mat[:, 1, 0] += self.dt * d_theta_ddot_d_action
    #     action_mat[:, 0, 0] += self.dt ** 2 * d_theta_ddot_d_action

    #     generator = torch.bmm(action_mat, self.action_set.generator)
    #     return sets.Zonotope(center, torch.cat([generator], dim=2))

    def draw(self):
        # Feel free to not have visualisation
        pass
