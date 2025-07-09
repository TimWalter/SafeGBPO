import os
from typing import Optional, Literal, Any

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
    n_forecasts: int = 4  # how many time steps to look into the future


    # Dim==state_shape[1] is required for the linearisation, such that the requirements
    # for a full dimensional box are given.
    noise: sets.Box = sets.Box(torch.tensor([[0.0, 10.8985, 0.0]]), torch.tensor([[[0.0, 0.0, 0.0], [20.1985, 0.0, 0.0], [0.0, 0.0, 0.0]]]))

    screen_width = 500
    screen_height = 500

    def __init__(self,
                 device: Literal["cpu", "cuda"] = "cpu",
                 num_envs: int = 1,
                 stochastic: bool = False,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 24,  # always train over one day
                 ):
        # load data sources
        path = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
        self.load_data = pd.read_csv(path + "/ICLR_load.csv", index_col=0)
        self.pv_data = pd.read_csv(path + "/ICLR_pv.csv", index_col=0)
        self.heatpump_data = pd.read_csv(path + "/DE_Temperature_and_COP2016.csv", sep=";", index_col=0)
        time_of_use_prices = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2])
        time_of_use_prices = np.tile(time_of_use_prices, 366)  # use the same price profile for every day of the year
        self.buying_price = pd.DataFrame(time_of_use_prices, columns=["price"])
        self.buying_price.insert(0, "t", self.pv_data.index)
        self.buying_price.set_index(["t"])

        eps = 0.1
        high_load = np.tile(self.load_data["p"].max() + eps, self.n_forecasts)
        low_load = np.tile(self.load_data["p"].min() - eps, self.n_forecasts)
        high_pv = np.tile(self.pv_data["p"].max() + eps, self.n_forecasts)
        low_pv = np.tile(self.pv_data["p"].min() - eps, self.n_forecasts)
        high_out_temp = np.tile(self.heatpump_data["outside_temp"].max() + eps, self.n_forecasts)
        low_out_temp = np.tile(self.heatpump_data["outside_temp"].min() - eps, self.n_forecasts)
        high_cop = np.tile(self.heatpump_data["COP"].max() + eps, self.n_forecasts)
        low_cop = np.tile(self.heatpump_data["COP"].min() - eps, self.n_forecasts)
        high_price = np.tile(self.buying_price["price"].max() + eps, self.n_forecasts)
        low_price = np.tile(self.buying_price["price"].min() - eps, self.n_forecasts)
        high_state = np.array([10.0, 24.0, 100], dtype=np.float64)
        low_state = np.array([0.0, 18.0, 10], dtype=np.float64)
        
        high = np.concatenate([high_state, high_load, high_pv, high_out_temp, high_cop, high_price])
        low = np.concatenate([low_state, low_load, low_pv, low_out_temp, low_cop, low_price])
        action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
        observation_space = spaces.Box(low, high, dtype=np.float64)

        super().__init__(device, num_envs, observation_space, action_space,
                         [
                             [low[0], high[0]], # bounds for state of charge, indoor temp, return temp
                             [low[1], high[1]],
                             [low[2], high[2]]
                         ],
                         stochastic, render_mode)

        self.max_episode_steps = max_episode_steps

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[
        Tensor, "{self.num_envs} {self.observation_space.shape[1]}"]:
        current_step = self.steps.detach().cpu().numpy()[0]
        load_forecast = self.get_data(self.load_data, current_step, self.n_forecasts, 'p').tile((self.num_envs,)).reshape((self.num_envs, self.n_forecasts))
        pv_forecast = self.get_data(self.pv_data, current_step, self.n_forecasts, 'p').tile((self.num_envs,)).reshape((self.num_envs, self.n_forecasts))
        out_temp_forecast = self.get_data(self.heatpump_data, current_step, self.n_forecasts, 'outside_temp').tile((self.num_envs,)).reshape((self.num_envs, self.n_forecasts))
        cop_forecast = 3.01#self.get_data(self.heatpump_data, current_step, self.n_forecasts, "COP").tile((self.num_envs,)).reshape((self.num_envs, self.n_forecasts))
        price_forecast = self.get_data(self.buying_price, current_step, self.n_forecasts, "price").tile((self.num_envs,)).reshape((self.num_envs, self.n_forecasts))
        
        return torch.cat([self.state, load_forecast, pv_forecast, out_temp_forecast, cop_forecast, price_forecast], dim=1)

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
        state = np.tile(np.array([self.soc_init, self.t_in_init, self.t_ret_init]),
                        self.num_envs).reshape(self.num_envs, -1)
        self.state = torch.tensor(state, dtype=torch.float64, device=self.device)

        self.steps = torch.zeros_like(self.steps)

        self.scheduled_reset = torch.zeros_like(self.scheduled_reset)

        if self.render_mode == "human":
            self.render()

        if seed is not None:
            torch.set_rng_state(rng_state)

        return self.observation, {}

    @jaxtyped(typechecker=beartype)
    def dynamics(self,
                 action: Float[Tensor, "{self.num_envs} {self.action_space.shape[1]}"]) \
            -> Float[Tensor, "{self.num_envs} {self.state.shape[1]}"]:
        """
        Do not alter the state yet only return the next state given the action"""
        p_ess, p_hp = action.split(1, dim=1)
        ess_w = 0.5 * (self.p_ess_max - self.p_ess_min)
        ess_b = 0.5 * (self.p_ess_max - self.p_ess_min) + self.p_ess_min
        hp_w = 0.5 * (self.p_hp_max - self.p_hp_min)
        hp_b = 0.5 * (self.p_hp_max - self.p_hp_min) + self.p_hp_min
        p_ess = p_ess * ess_w + ess_b
        p_hp = p_hp * hp_w + hp_b

        soc, t_in, t_ret = self.state.split(1, dim=1)

        current_step = self.steps.detach().cpu().numpy()[0]
        t_out = self.get_data(self.heatpump_data, current_step, 1, "outside_temp")
        cop = 3.01#self.get_data(self.heatpump_data, current_step, 1, "COP")

        c_0 = (self.h_fh + self.h_out)/(self.h_out * self.tau)
        c_1 = self.h_fh/(self.h_out * self.tau)
        c_2 = self.h_fh/self.c_w_fh
        c_3 = t_out/self.tau
        c_4 = cop/self.c_w_fh

        soc = soc + p_ess * self.dt
        new_t_in = t_in + self.dt * (
            -c_0 * t_in
            + c_1 * t_ret
            + c_3
            )
        new_t_ret = t_ret + self.dt * (
            c_2 * t_in
            - c_2 * t_ret
            + c_4 * p_hp
        )
        return torch.cat([soc, new_t_in, new_t_ret], dim=1)
    
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
        current_step: np.int32,
        n_timesteps: int,
        key: str
    ) -> Float[Tensor, "{n_timesteps}"]:
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

    @jaxtyped(typechecker=beartype)
    def linear_dynamics(self,
                        lin_state: Float[Tensor, "{self.state.shape[1]}"],
                        lin_action: Float[Tensor, "{self.action_space.shape[1]}"],
                        lin_noise: Float[Tensor, "{self.noise.center.shape[1]}"]
                        ) -> tuple[
        Float[Tensor, "{self.state.shape[1]}"],
        Float[Tensor, "{self.state.shape[1]} {self.state.shape[1]}"],
        Float[Tensor, "{self.state.shape[1]} {self.action_space.shape[1]}"],
        Float[Tensor, "{self.state.shape[1]} {self.noise.center.shape[1]}"]
    ]:
        """
        Compute the linearised dynamics around the given state, action and noise by
        computing a first order taylor series of the update.

        Args:
            lin_state: Linearisation point for the state.
            lin_action: Linearisation point for the action.
            lin_noise: Linearisation point for the noise.

        Returns:
            constant_matrix: The constant matrix in the linear dynamics.
            state_matrix: The state matrix in the linear dynamics.
            action_matrix: The action matrix in the linear dynamics.
            noise_matrix: The noise matrix in the linear dynamics.
        """
        ess_w = 0.5 * (self.p_ess_max - self.p_ess_min)
        ess_b = 0.5 * (self.p_ess_max - self.p_ess_min) + self.p_ess_min
        hp_w = 0.5 * (self.p_hp_max - self.p_hp_min)
        hp_b = 0.5 * (self.p_hp_max - self.p_hp_min) + self.p_hp_min
        p_ess = lin_action[0] * ess_w + ess_b
        p_hp = lin_action[1] * hp_w + hp_b

        # observe
        current_step = self.steps.detach().cpu().numpy()[0]
        #t_out = self.get_data(self.heatpump_data, current_step, 1, "outside_temp")
        cop = 3.01 #self.get_data(self.heatpump_data, current_step, 1, "COP")

        # define constants
        c_0 = (self.h_fh + self.h_out) / (self.h_out * self.tau)
        c_1 = self.h_fh / (self.h_out * self.tau)
        c_2 = self.h_fh / self.c_w_fh
        c_3 = lin_noise[1] / self.tau
        c_4 = cop / self.c_w_fh

        constant_mat = torch.tensor([
            lin_state[0] + self.dt * p_ess,
            lin_state[1] + self.dt * (-c_0*lin_state[1] + c_1*lin_state[2] + c_3),
            lin_state[2] + self.dt * (c_2*lin_state[1] - c_2*lin_state[2] + c_4*p_hp)
        ], dtype=torch.float64, device=self.device)

        state_mat = torch.eye(3, dtype=torch.float64, device=self.device)
        state_mat[1, 1] += self.dt * (-c_0)
        state_mat[2, 1] += self.dt * c_2
        state_mat[1, 2] += self.dt * c_1
        state_mat[2, 2] += self.dt * (-c_2)

        action_mat = torch.zeros((3, 2), dtype=torch.float64, device=self.device)
        action_mat[0, 0] += self.dt * ess_w
        action_mat[2, 1] += self.dt * c_4 * hp_w

        noise_mat = torch.zeros((self.state.shape[1], self.noise.center.shape[1]),
                                dtype=torch.float64, device=self.device)
        noise_mat[1, 1] += self.dt / self.tau

        return constant_mat, state_mat, action_mat, noise_mat

    @jaxtyped(typechecker=beartype)
    def reachable_set(self) -> sets.Zonotope:
        """
        Compute the one step reachable set.

        Returns:
            The one step reachable set.
        """
        center = self.dynamics(self.action_set.center)

        action_mat = torch.zeros((*self.state.shape, self.action_space.shape[1]),
                                 dtype=torch.float64, device=self.device)

        current_step = self.steps.detach().cpu().numpy()[0]
        cop = self.get_data(self.heatpump_data, current_step, 1, "COP")

        ess_w = 0.5 * (self.p_ess_max - self.p_ess_min)
        hp_w = 0.5 * (self.p_hp_max - self.p_hp_min)
        c_4 = cop / self.c_w_fh

        action_mat[:, 0, 0] += self.dt * ess_w
        action_mat[:, 2, 1] += self.dt * c_4[0] * hp_w
        generator = torch.bmm(action_mat, self.action_set.generator)
        return sets.Zonotope(center, torch.cat([generator], dim=2))

    def draw(self):
        # Feel free to not have visualisation
        pass
