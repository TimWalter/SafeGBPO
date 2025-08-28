from conf.learning_algorithms import SHACConfig

config = SHACConfig(
    len_trajectories=64,
    polyak_target=0.4,
    td_weight=0.95,
    vf_num_fits=50,
    vf_fit_num_batches=1,
    policy_kwargs={"net_arch": [128, 128]},
    policy_optim_kwargs={"lr": 0.0004},
    vf_kwargs={"net_arch": [128, 128, 128]},
    vf_optim_kwargs={"lr": 0.002},
)
