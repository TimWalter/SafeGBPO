from pathlib import Path

import torch
from torch import Tensor
from beartype import beartype
from PIL import Image
from jaxtyping import jaxtyped, Float, Bool
from torchvision.transforms.functional import to_tensor

import src.sets as sets
from envs.simulators.interfaces.simulator import Simulator


class HouseholdEnv(Simulator):
    """
    The environment simulates a household with a battery and a heat pump.

    ## Action Set
    | Num | Action                | Min  | Max |
    |-----|-----------------------|------|-----|
    | 0   | Battery Charge Power  | -1.0 | 1.0 |
    | 1   | Heat Pump Power       | -1.0 | 1.0 |

    ## State Set
    | Num | State              | Min  | Max   |
    |-----|--------------------|------|-------|
    | 0   | State of Charge    | 0.0  | 10.0  |
    | 1   | Indoor Temperature | 18.0 | 24.0  |
    | 2   | Return Temperature | 10.0 | 100.0 |

    ## Observation Set
    | Num  | Observation                  | Min  | Max   |
    |------|------------------------------|------|-------|
    | 0    | State of Charge              | 0.0  | 10.0  |
    | 1    | Indoor Temperature           | 18.0 | 24.0  |
    | 2    | Return Temperature           | 10.0 | 100.0 |
    | 3-7  | Load Forecast                | 0.0  | 1.0   |
    | 8-12 | PV Forecast                  | 0.0  | 1.0   |
    | 13-17| Outside Temperature Forecast | -10.0| 40.0  |
    | 18-22| Buying Price Forecast        | 0.0 | 1.0   |
    """
    DT: float = 1.0
    BATTERY_CHARGE_POWER_MAG: float = 2.0
    HEAT_PUMP_POWER_MAG: float = 2.5
    HEAT_PUMP_POWER_OFFSET: float = 2.5
    H_FH: float = 1.1
    H_OUT: float = 0.26
    TAU: float = 240
    C_W_FH: float = 1.1625
    COP: float = 3.01
    C_0: float = (H_FH + H_OUT) / (H_OUT * TAU)
    C_1: float = H_FH / (H_OUT * TAU)
    C_2: float = H_FH / C_W_FH
    C_4: float = COP / C_W_FH


    START_TIME: int = 7800
    ASSET_PATH: Path = Path(__file__).parent.parent.parent / "assets"
    LOAD_DATA: Tensor = torch.load(ASSET_PATH / "load_data.pt", weights_only=True).to(torch.get_default_device())
    PV_DATA: Tensor = torch.load(ASSET_PATH / "pv_data.pt", weights_only=True).to(torch.get_default_device())
    T_OUT_DATA: Tensor = torch.load(ASSET_PATH / "outside_temperature_data.pt", weights_only=True).to(
        torch.get_default_device())
    BUYING_DATA: Tensor = torch.tensor(
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5,
         0.5, 0.2, 0.2]).repeat(366)

    SCREEN_WIDTH: int = 0
    SCREEN_HEIGHT: int = 0

    @jaxtyped(typechecker=beartype)
    def __init__(self, num_envs: int, num_steps: int):
        """
        Initialize the Pendulum environment.

        Args:
            num_envs: Number of environments to vectorize.
            num_steps: Number of steps till the episode ends.
        """
        state_set = sets.AxisAlignedBox(
            torch.tensor([5.0, 21.0, 55.0]).unsqueeze(0).repeat(num_envs, 1),
            torch.diag_embed(torch.tensor([5.0, 3.0, 45.0]).repeat(num_envs, 1))
        )
        noise_set = sets.AxisAlignedBox(
            torch.tensor([0.0, 15.0, 0.0]).unsqueeze(0).repeat(num_envs, 1),
            torch.diag_embed(torch.tensor([0.0, 25.0, 0.0]).repeat(num_envs, 1))
        )
        observation_set = sets.AxisAlignedBox(
            torch.tensor([5.0, 21.0, 55.0] + [0.5] * 10 + [15.0] * 5 + [0.5] * 5).unsqueeze(0).repeat(num_envs, 1),
            torch.diag_embed(torch.tensor([5.0, 3.0, 45.0] + [0.5] * 10 + [25.0] * 5 + [0.5] * 5).repeat(num_envs, 1))
        )
        super().__init__(2, state_set, noise_set, observation_set, num_envs)

        self.num_steps = num_steps

        self.LOAD_DATA = self.LOAD_DATA[self.START_TIME:].unsqueeze(0).repeat(num_envs, 1)
        self.PV_DATA = self.PV_DATA[self.START_TIME:].unsqueeze(0).repeat(num_envs, 1)
        self.T_OUT_DATA = self.T_OUT_DATA[self.START_TIME:].unsqueeze(0).repeat(num_envs, 1)
        self.BUYING_DATA = self.BUYING_DATA[self.START_TIME:].unsqueeze(0).repeat(num_envs, 1)

    @jaxtyped(typechecker=beartype)
    @property
    def observation(self) -> Float[Tensor, "{self.num_envs} {self.obs_dim}"]:
        """
        Get the current observation of the environment.

        Returns:
            The current observation of the environment as a batch of observations.
        """
        return torch.cat([self.state,
                          self.LOAD_DATA[:, self.steps:self.steps + 5],
                          self.PV_DATA[:, self.steps:self.steps + 5],
                          self.T_OUT_DATA[:, self.steps:self.steps + 5],
                          self.BUYING_DATA[:, self.steps:self.steps + 5]], dim=1)

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
        return torch.zeros(self.num_envs)

    @jaxtyped(typechecker=beartype)
    def episode_ending(self) -> tuple[
        Bool[Tensor, "{self.num_envs}"],
        Bool[Tensor, "{self.num_envs}"],
    ]:
        """
        Check if the episode is ending for each environment.

        Returns:
            terminated: Whether the episode is terminated for each environment.
            truncated: Whether the episode is truncated for each environment.

        Notes:
            Termination
            refers to the episode ending after reaching a terminal state
            that is defined as part of the environment definition.
            Examples are - task success, task failure, robot falling down etc.
            Notably, this also includes episodes ending in finite-horizon environments
            due to a time-limit inherent to the environment. Note that to preserve
            Markov property, a representation of the remaining time must be present in
            the agent’s observation in finite-horizon environments

            Truncation
            refers to the episode ending after an externally defined condition (that is
            outside the scope of the Markov Decision Process). This could be a
            time-limit, a robot going out of bounds etc. An infinite-horizon environment
            is an obvious example of where this is needed. We cannot wait forever for
            the episode to complete, so we set a practical time-limit after which we
            forcibly halt the episode. The last state in this case is not a terminal
            state since it has a non-zero transition probability of moving to another
            state as per the Markov Decision Process that defines the RL problem. This
            is also different from time-limits in finite horizon environments as the
            agent in this case has no idea about this time-limit.
        """
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = (self.steps >= self.num_steps) * torch.ones(self.num_envs, dtype=torch.bool)
        return terminated, truncated

    @jaxtyped(typechecker=beartype)
    def execute_action(self, action: Float[Tensor, "{self.num_envs} {self.action_dim}"]):
        """
        Execute the action in the environment by updating the state.

        Args:
            action: Action to execute in the environment.
        """
        noise = self.noise_set.sample()
        noise[:, 1] = self.T_OUT_DATA[:, self.steps]
        self.state = self.dynamics(self.state, action, noise)

    @jaxtyped(typechecker=beartype)
    def unbatched_dynamics(self,
                           state: Float[Tensor, "{self.state_dim}"],
                           action: Float[Tensor, "{self.action_dim}"],
                           noise: Float[Tensor, "{self.state_dim}"]) \
            -> Float[Tensor, "{self.state_dim}"]:
        """
        Unbatched dynamics function that computes the next state given the current state,
        action, and noise. We batch this function automatically using torch.vmap for vectorized execution.

        Args:
            state: Current state of the environment.
            action: Action to execute in the environment.
            noise: Noise sample to perturb the dynamics.

        Returns:
            Next state.
        """
        soc, t_in, t_ret = state.split(1)
        battery_charge_power, heat_pump_power = action.split(1)
        battery_charge_power = battery_charge_power * self.BATTERY_CHARGE_POWER_MAG
        heat_pump_power = heat_pump_power * self.HEAT_PUMP_POWER_MAG

        t_out = noise[1:2]
        c_3 = t_out / self.TAU

        soc_dot = battery_charge_power
        t_in_dot = -self.C_0 * t_in + self.C_1 * t_ret + c_3
        t_ret_dot = self.C_2 * t_in - self.C_2 * t_ret + self.C_4 * heat_pump_power

        soc = soc + self.DT * soc_dot
        t_in = t_in + self.DT * t_in_dot
        t_ret = t_ret + self.DT * t_ret_dot

        return torch.cat([soc, t_in, t_ret])

    @jaxtyped(typechecker=beartype)
    def render(self) -> list[Tensor]:
        """
        Render all environments.

        Returns:
            A list of rendered frames for each environment.
        """
        frames = []
        for i in range(self.num_envs):
            img = Image.new("RGB", (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), "white")
            frames.append((to_tensor(img) * 255).to(torch.uint8))

        return frames
