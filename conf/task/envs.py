from dataclasses import dataclass, field

from omegaconf import MISSING

from conf.task.wrapper import (
    BoundaryProjectionConfig,
    ZonotopeRayMapConfig,
    SafetyWrapperConfig
)


@dataclass
class TorchVectorConfig:
    name: str = MISSING
    device: str = "cuda"
    num_envs: int = MISSING
    stochastic: bool = True
    render_mode: str = "rgb_array"
    wrappers: list = MISSING


@dataclass
class PendulumConfig(TorchVectorConfig):
    name: str = "PendulumEnv"
    num_envs: int = 8
    wrappers: list = field(default_factory=lambda: [
        BoundaryProjectionConfig(
            lin_state=[0.0, 0.0],
            lin_action=[0.0],
            lin_noise=[0.0, 0.0],
        ),
    ])


@dataclass
class CartPoleConfig(TorchVectorConfig):
    name: str = "CartPoleEnv"
    num_envs: int = 16
    stochastic: bool = False
    wrappers: list = field(default_factory=lambda: [
        BoundaryProjectionConfig(
            lin_state=[0.0, 0.0, 0.0, 0.0],
            lin_action=[0.0],
            lin_noise=[0.0, 0.0, 0.0, 0.0],
        ),
    ])


@dataclass
class QuadrotorConfig(TorchVectorConfig):
    name: str = "QuadrotorEnv"
    num_envs: int = 16
    wrappers: list = field(default_factory=lambda: [
        BoundaryProjectionConfig(
            lin_state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            lin_action=[0.0, 0.0],
            lin_noise=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ])

@dataclass
class SeekerConfig(TorchVectorConfig):
    name: str = "SeekerEnv"
    num_envs: int = 8
    wrappers: list = field(default_factory=lambda: [
        BoundaryProjectionConfig(
            lin_state=[0.0, 0.0],
            lin_action=[0.0, 0.0],
            lin_noise=[0.0, 0.0],
        ),
    ])

@dataclass
class HouseholdConfig(TorchVectorConfig):
    name: str = "HouseholdEnv"
    num_envs: int = 8
    wrappers: list = field(default_factory=lambda: [
        BoundaryProjectionConfig(
            lin_state=[5.0, 21.0, 55.0],
            lin_action=[0.0, 0.0],
            lin_noise=[0.0, 10.8985, 0.0],
        ),
    ])