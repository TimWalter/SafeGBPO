from conf.learning_algorithms import SACConfig

config = SACConfig(
    buffer_size=60_000,
    polyak_target=0.673,
    learning_starts=64,
    policy_frequency=6,
    target_frequency=7,
    policy_kwargs={"net_arch": [173, 173, 173, 173]},
    policy_optim_kwargs={"lr": 2e-3},
    vf_kwargs={"net_arch": [135, 135, 135, 135]},
    vf_optim_kwargs={"lr": 2e-3},
)
