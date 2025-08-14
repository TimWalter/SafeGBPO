from conf.learning_algorithms import PPOConfig

config = PPOConfig(
    len_trajectories=1400,
    clip_coef=0.29,
    ent_coef=0.06,
    num_batches=30,
    num_fits=30,
    policy_kwargs={"net_arch": [128, 128, 128]},
    policy_optim_kwargs={"lr": 1e-4},
    vf_kwargs={"net_arch": [128, 128, 128, 128]},
    vf_optim_kwargs={"lr": 3e-3},
)
