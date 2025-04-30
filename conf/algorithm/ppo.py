from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    name: str = "PPO"
    device: str = "cuda"
    len_trajectories: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_batches: int = 32
    num_fits: int = 40
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    ent_coef: float = 0.0
    max_grad_norm: float = 2.5
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [64, 64],
        "activation_fn": "nn.ReLu()",
        "layer_norm": False
    })
    policy_learning_rate_schedule: str = "constant"
    policy_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 3e-4,
        "betas": [0.7, 0.95]
    })
    vf_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [64, 64],
        "activation_fn": "nn.ReLu()",
        "layer_norm": False
    })
    vf_learning_rate_schedule: str = "linear"
    vf_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 3e-4,
        "betas": [0.7, 0.95]
    })
