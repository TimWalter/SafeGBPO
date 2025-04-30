from dataclasses import dataclass, field


@dataclass
class SACConfig:
    name: str = "SAC"
    device: str = "cuda"
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    polyak_target: float = 0.995
    batch_size: int = 256
    learning_starts: int = 120
    policy_frequency: int = 2
    target_frequency: int = 1
    alpha: float = 0.1
    autotune: bool = True
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [256, 256],
        "activation_fn": "nn.ReLU()",
        "layer_norm": False
    })
    policy_learning_rate_schedule: str = "constant"
    policy_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 3e-4,
        "betas": [0.7, 0.95]
    })
    vf_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [256, 256],
        "activation_fn": "nn.ReLU()",
        "layer_norm": False
    })
    vf_learning_rate_schedule: str = "linear"
    vf_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 1e-3,
        "betas": [0.7, 0.95]
    })
