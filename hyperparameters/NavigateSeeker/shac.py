from conf.learning_algorithms import SHACConfig

config = SHACConfig(
    len_trajectories=256,
    polyak_target=0.63,
    td_weight=0.91,
    vf_num_fits=8,
    vf_fit_num_batches=167,
    policy_kwargs={"net_arch": [436, 436, 436]},
    policy_optim_kwargs={"lr": 0.0004},
    vf_kwargs={"net_arch": [418, 418, 418, 418]},
    vf_optim_kwargs={"lr": 0.0001},
)
