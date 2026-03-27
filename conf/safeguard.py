from dataclasses import dataclass


@dataclass
class SafeguardConfig:
    regularisation_coefficient: float

    @property
    def name(self) -> str:
        return self.__class__.__name__[:-6]


@dataclass
class BoundaryProjectionConfig(SafeguardConfig):
    regularisation_coefficient: float = 0.1


@dataclass
class RayMaskConfig(SafeguardConfig):
    regularisation_coefficient: float = 0.1
    linear_projection: bool = True
    zonotopic_approximation: bool = True
    passthrough: bool = False


@dataclass
class FSNetConfig(SafeguardConfig):
    regularisation_coefficient: float = 0.05
    penalty_threshold: float = 1e-3
    penalty_coefficient: float = 0.1
    differentiable_steps: int = 4
    total_steps: int = 8


@dataclass
class PinetConfig(SafeguardConfig):
    regularisation_coefficient: float = 0.1
    n_iter: int = 10
    n_iter_bwd: int = 10
