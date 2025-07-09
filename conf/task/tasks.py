from conf.task.envs import *
from conf.task.wrapper import OrthogonalRayMapConfig


@dataclass
class RCITaskConfig:
    rci_size: int = MISSING


@dataclass
class BalancePendulumConfig(RCITaskConfig, PendulumConfig):
    name: str = "BalancePendulumTask"
    rci_size: int = 4
    max_episode_steps: int = 240
    wrappers: list = field(default_factory=lambda: [
        #BoundaryProjectionConfig(
        #    lin_state=[0.0, 0.0],
        #    lin_action=[0.0],
        #    lin_noise=[0.0, 0.0]
        #)
    ])




@dataclass
class BalanceCartPoleConfig(RCITaskConfig, CartPoleConfig):
    name: str = "BalanceCartPoleTask"
    rci_size: int = 4
    max_episode_steps: int = 240
    wrappers: list = field(default_factory=lambda: [
    ])


@dataclass
class BalanceQuadrotorConfig(RCITaskConfig, QuadrotorConfig):
    name: str = "BalanceQuadrotorTask"
    rci_size: int = 4
    max_episode_steps: int = 240
    wrappers: list = field(default_factory=lambda: [
    ])


@dataclass
class SwingUpCartPoleConfig(CartPoleConfig):
    name: str = "SwingUpCartPoleTask"
    max_episode_steps: int = 240
    num_envs: int = 32
    wrappers: list = field(default_factory=lambda: [
        OrthogonalRayMapConfig(
            lin_state=[0.0, 0.0, 0.0, 0.0],
            lin_action=[0.0],
            lin_noise=[0.0, 0.0, 0.0, 0.0],
            linear_projection=False
        ),
    ])


@dataclass
class NavigateQuadrotorConfig(QuadrotorConfig):
    name: str = "NavigateQuadrotorTask"
    max_episode_steps: int = 400
    num_envs: int = 64
    num_obstacles: int = 1
    domain_width: int = 16
    domain_height: int = 16
    min_radius: float = 2.0
    max_radius: float = 3.0
    draw_safe_state_set: bool = False
    wrappers: list = field(
        default_factory=lambda: [
            BoundaryProjectionConfig(
                lin_state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                lin_action=[0.0, 0.0],
                lin_noise=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        ]
    )

@dataclass
class NavigateSeekerConfig(SeekerConfig):
    name: str = "NavigateSeekerTask"
    max_episode_steps: int = 400
    num_envs: int = 64
    num_obstacles: int = 1
    domain_width: int = 16
    domain_height: int = 16
    min_radius: float = 2.0
    max_radius: float = 4.0
    draw_safe_state_set: bool = False
    wrappers: list = field(default_factory=lambda: [
        BoundaryProjectionConfig(
            lin_state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            lin_action=[0.0, 0.0],
            lin_noise=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
    ])

@dataclass
class LoadBalanceHouseholdConfig(HouseholdConfig):
    name: str = "LoadBalanceHouseholdTask"
    max_episode_steps: int = 1000
    num_envs: int = 8
    wrappers: list = field(default_factory=lambda: [
        BoundaryProjectionConfig(
            lin_state=[5.0, 21.0, 55.0],
            lin_action=[0.0, 0.0],
            lin_noise=[0.0, 10.8985, 0.0],
        ),
    ])