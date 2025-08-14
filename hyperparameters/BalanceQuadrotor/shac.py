from conf.learning_algorithms import SHACConfig

config = SHACConfig(
    len_trajectories=43,
    polyak_target=0.22,
    td_weight=0.93,
    vf_num_fits=24,
    vf_fit_num_batches=1,
    policy_kwargs={"net_arch": [128, 128]},
    policy_optim_kwargs={"lr": 0.002},
    vf_kwargs={"net_arch": [120, 120, 120]},
    vf_optim_kwargs={"lr": 0.004},
)
