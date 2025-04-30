from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    name: str = "PPO"
    len_trajectories: int = 1700
    gamma: float = 0.95
    gae_lambda: float = 0.94
    num_batches: int = 45
    num_fits: int = 32
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    ent_coef: float = 0.09
    max_grad_norm: float = 1.45
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [256, 256, 256, 256],
        "activation_fn": "nn.Tanh()",
        "layer_norm": False
    })
    policy_learning_rate_schedule: str = "constant"
    policy_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 3e-4,
        "betas": [0.9, 0.999]
    })
    vf_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [200, 200],
        "activation_fn": "nn.ELU()",
        "layer_norm": True
    })
    vf_learning_rate_schedule: str = "linear"
    vf_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 2e-3,
        "betas": [0.7, 0.95]
    })
