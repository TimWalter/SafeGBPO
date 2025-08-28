import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor

from envs.simulators.household import HouseholdEnv
from envs.interfaces.safe_state_env import SafeStateEnv



class ManageHouseholdEnv(HouseholdEnv, SafeStateEnv):
    """
    The household environment starts in a random state and its goal is to
    maintain a comfortable indoor temperature while minimising electricity costs.

    ## Rewards
    Since the goal is to keep the indoor temperature close to the setpoint,
    the reward punishes for:
    - Deviations from the setpoint temperature
    - Electricity costs

    ## Starting State
    The starting state is sampled uniformly from the state set.

    ## Safety
    We define the safety constraints as the part of the state space from which the controller can maintain
    feasibility within one time step.
    """
    SELLING_PRICE: float = 0.08
    T_IN_SETPOINT: float = 21
    COST_COEFFICIENT_HP: float = 100

    @jaxtyped(typechecker=beartype)
    def __init__(self, num_envs: int, num_steps: int):
        """
        Initialize the ManageHousehold environment.

        Args:
            num_envs: Number of environments to vectorize.
            num_steps: Number of steps till the episode ends.
        """
        SafeStateEnv.__init__(self, num_state_gens=3)
        HouseholdEnv.__init__(self, num_envs, num_steps)

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
        soc, t_in, t_ret = self.state.split(1, dim=1)
        battery_charge_power, heat_pump_power = action.split(1, dim=1)

        p_total = (battery_charge_power + heat_pump_power
                   + self.LOAD_DATA[:, self.steps:self.steps + 1]
                   + self.PV_DATA[:, self.steps:self.steps + 1])
        electricity_cost = torch.where(
            p_total >= 0,
            p_total * self.DT * self.BUYING_DATA[:, self.steps:self.steps + 1],
            p_total * self.DT * self.SELLING_PRICE
        )
        comfort_cost = (t_in - self.T_IN_SETPOINT) ** 2 * self.COST_COEFFICIENT_HP
        return (-electricity_cost - comfort_cost).squeeze(dim=1)

    @jaxtyped(typechecker=beartype)
    def safe_state_set(self):
        """
        Get the safe state set for the current state.
m
        Returns:
            A convex set representing the safe state set.
        """
        return self.state_set
