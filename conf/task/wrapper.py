from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class SafetyWrapperConfig:
    name: str = MISSING
    lin_state: list = MISSING
    lin_action: list = MISSING
    lin_noise: list = MISSING

@dataclass
class BoundaryProjectionConfig(SafetyWrapperConfig):
    name: str = "BoundaryProjectionWrapper"

@dataclass
class OrthogonalRayMapConfig(SafetyWrapperConfig):
    name: str = "OrthogonalRayMapWrapper"
    linear_projection: bool = True

@dataclass
class ZonotopeRayMapConfig(SafetyWrapperConfig):
    name: str = "ZonotopeRayMapWrapper"
    linear_projection: bool = True
    num_generators: int = MISSING
    reuse_safe_set: bool = True
    passthrough: bool = False

