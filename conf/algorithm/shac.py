from dataclasses import dataclass, field


@dataclass
class SHACConfig:
    name: str = "SHAC"
    len_trajectories: int = 60
    gamma: float = 0.99
    polyak_target: float = 0.312
    td_weight: float = 0.99
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [64, 64],
        "activation_fn": "nn.ELU()",
        "layer_norm": True
    })
    policy_learning_rate_schedule: str = "constant"
    policy_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 0.00125,
        "betas": [0.7, 0.95]
    })
    vf_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [64, 64]
    })
    vf_learning_rate_schedule: str = "linear"
    vf_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 0.0021,
        "betas": [0.9, 0.999],
        "layer_norm": True
    })
    vf_num_fits: int = 36
    vf_fit_num_batches: int = 4
    clip_grad: bool = True
    max_grad_norm: float = 2.1
    regularisation_coefficient: float = 0.1
    adaptive_regularisation: bool = False
