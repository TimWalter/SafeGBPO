from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    name: str = "PPO"
    len_trajectories: int = 1400
    gamma: float = 0.99
    gae_lambda: float = 0.94
    num_batches: int = 30
    num_fits: int = 30
    norm_adv: bool = True
    clip_coef: float = 0.29
    clip_value_loss: bool = True
    ent_coef: float = 0.06
    max_grad_norm: float = 2.45
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [128, 128, 128],
        "activation_fn": "nn.ReLU()",
        "layer_norm": False
    })
    policy_learning_rate_schedule: str = "constant"
    policy_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 1e-4,
        "betas": [0.9, 0.999]
    })
    vf_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [128, 128, 128, 128],
        "activation_fn": "nn.ReLU()",
        "layer_norm": True
    })
    vf_learning_rate_schedule: str = "linear"
    vf_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 3e-3,
        "betas": [0.7, 0.95]
    })
