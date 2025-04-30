from dataclasses import dataclass, field


@dataclass
class SACConfig:
    name: str = "SAC"
    buffer_size: int = 60_000
    gamma: float = 0.99
    polyak_target: float = 0.673
    batch_size: int = 64
    learning_starts: int = 64
    policy_frequency: int = 6
    target_frequency: int = 7
    alpha: float = 0.2
    autotune: bool = True
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [173, 173, 173, 173],
        "activation_fn": "nn.ReLU()",
        "layer_norm": False
    })
    policy_learning_rate_schedule: str = "linear"
    policy_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 2e-3,
        "betas": [0.9, 0.999]
    })
    vf_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [135, 135, 135, 135],
        "activation_fn": "nn.ReLU()",
        "layer_norm": False
    })
    vf_learning_rate_schedule: str = "constant"
    vf_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 2e-3,
        "betas": [0.7, 0.95]
    })
