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
    regularisation_coefficient: float = 0.1
    penalty_addition_threshold: float = 1e-3
    residual_coefficient: float = 0.1
    
    # FSNet general solver config parameters
    lbfgs_history_size: int = 8
    max_iter: int = 8
    max_diff_iter: int = 4

    # fsnet lbfgs torch opt solver config parameters
    lbfgs_torch_learning_rate: float = 1.0
    gradient_clipping_max_norm: float = 2.0 # gradient clipping for stable training

    # fsnet lbfgs original solver config parameters
    convergence_value_tolerance: float = 1e-6
    convergence_gradient_tolerance: float = 1e-6
    objective_scale: float = 1.0
    line_search_armijo_c: float = 1e-4   
    line_search_rho: float = 0.5
    line_search_max_iter: int = 10
    verbose: bool = False
    store_trajectory: bool = False

@dataclass
class PinetConfig(SafeguardConfig):
    regularisation_coefficient: float = 0.1
    n_iter_admm: int = 10
    n_iter_bwd: int = 10
    fpi: bool = False
    store_trajectory: bool = False