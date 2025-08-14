from conf.learning_algorithms import SACConfig

config = SACConfig(
    buffer_size=300_000,
    polyak_target=0.3,
    learning_starts=6000,
    policy_frequency=1,
    target_frequency=1,
    policy_kwargs={"net_arch": [838, 838]},
    policy_optim_kwargs={"lr": 1e-5},
    vf_kwargs={"net_arch": [937, 937, 937]},
    vf_optim_kwargs={"lr": 2.5e-4},
)
