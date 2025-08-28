from conf.learning_algorithms import SACConfig

config = SACConfig(
    buffer_size=45_000,
    polyak_target=0.323,
    learning_starts=5000,
    policy_frequency=10,
    target_frequency=2,
    policy_kwargs={"net_arch": [359, 359, 359, 359]},
    policy_optim_kwargs={"lr": 1e-3},
    vf_kwargs={"net_arch": [977, 977, 977]},
    vf_optim_kwargs={"lr": 5e-4},
)
