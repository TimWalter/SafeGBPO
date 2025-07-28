from dataclasses import dataclass, field


@dataclass
class SHACConfig:
    name: str = "SHAC"
    len_trajectories: int = 550
    gamma: float = 0.9
    polyak_target: float = 0.7
    td_weight: float = 0.93
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [512, 512, 512],
        "activation_fn": "nn.ELU()",
        "layer_norm": True
    })
    policy_learning_rate_schedule: str = "constant"
    policy_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 1e-6,
        "betas": [0.9, 0.999]
    })
    vf_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [512, 512, 512]
    })
    vf_learning_rate_schedule: str = "linear"
    vf_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 2e-6,
        "betas": [0.9, 0.999]
    })
    vf_num_fits: int = 50
    vf_fit_num_batches: int = 100
    clip_grad: bool = True
    max_grad_norm: float = 1.5
    regularisation_coefficient: float = 0.0
    adaptive_regularisation: bool = False
