from dataclasses import dataclass, field


@dataclass
class SHACConfig:
    name: str = "SHAC"
    len_trajectories: int = 64
    gamma: float = 0.95
    polyak_target: float = 0.4
    td_weight: float = 0.95
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [128, 128],
        "activation_fn": "nn.ELU()",
        "layer_norm": True
    })
    policy_learning_rate_schedule: str = "constant"
    policy_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 0.0004,
        "betas": [0.7, 0.95]
    })
    vf_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [128, 128, 128]
    })
    vf_learning_rate_schedule: str = "linear"
    vf_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 0.002,
        "betas": [0.7, 0.95]
    })
    vf_num_fits: int = 50
    vf_fit_num_batches: int = 1
    clip_grad: bool = True
    max_grad_norm: float = 1.0
    regularisation_coefficient: float = 0.0
    adaptive_regularisation: bool = False
