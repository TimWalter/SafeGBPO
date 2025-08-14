from conf.learning_algorithms import SHACConfig

config = SHACConfig(
    len_trajectories=550,
    polyak_target=0.7,
    td_weight=0.93,
    vf_num_fits=50,
    vf_fit_num_batches=100,
    policy_kwargs={"net_arch": [512, 512, 512]},
    policy_optim_kwargs={"lr": 1e-6},
    vf_kwargs={"net_arch": [512, 512, 512]},
    vf_optim_kwargs={"lr": 2e-6},
)
