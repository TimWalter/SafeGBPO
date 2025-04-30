from dataclasses import dataclass, field


@dataclass
class SACConfig:
    name: str = "SAC"
    buffer_size: int = 45_000
    gamma: float = 0.99
    polyak_target: float = 0.323
    batch_size: int = 256
    learning_starts: int = 5000
    policy_frequency: int = 10
    target_frequency: int = 2
    alpha: float = 0.47
    autotune: bool = True
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [359, 359, 359, 359],
        "activation_fn": "nn.ReLU()",
        "layer_norm": False
    })
    policy_learning_rate_schedule: str = "linear"
    policy_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 1e-3,
        "betas": [0.7, 0.95]
    })
    vf_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [977, 977, 977],
        "activation_fn": "nn.ELU()",
        "layer_norm": False
    })
    vf_learning_rate_schedule: str = "linear"
    vf_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 5e-4,
        "betas": [0.9, 0.999]
    })
