from dataclasses import dataclass


@dataclass
class EnvConfig:
    num_envs: int
    num_steps: int

    @property
    def name(self) -> str:
        return self.__class__.__name__[:-6]


@dataclass
class BalancePendulumConfig(EnvConfig):
    num_envs: int = 8
    num_steps: int = 3000


@dataclass
class BalanceCartPoleConfig(EnvConfig):
    num_envs: int = 16
    num_steps: int = 240


@dataclass
class SwingUpCartPoleConfig(EnvConfig):
    num_envs: int = 32
    num_steps: int = 240


@dataclass
class BalanceQuadrotorConfig(EnvConfig):
    num_envs: int = 16
    num_steps: int = 240


@dataclass
class NavigateQuadrotorConfig(EnvConfig):
    num_envs: int = 64
    num_steps: int = 400
    num_obstacles: int = 1
    min_radius: float = 2.0
    max_radius: float = 3.0
    draw_safe_state_set: bool = False


@dataclass
class NavigateSeekerConfig(EnvConfig):
    num_envs: int = 24
    num_steps: int = 400
    num_obstacles: int = 1
    min_radius: float = 2.0
    max_radius: float = 4.0
    draw_safe_action_set: bool = True
    safe_action_polytope: bool =  False

@dataclass
class ManageHouseholdConfig(EnvConfig):
    num_envs: int = 8
    num_steps: int = 24 * 30
