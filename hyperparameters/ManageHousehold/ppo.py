from conf.learning_algorithms import PPOConfig

config = PPOConfig(
    len_trajectories=24*30*4,
    clip_coef=0.32,
    ent_coef=0.065,
    num_batches = 30,
    num_fits = 30,
    policy_kwargs={"net_arch": [256, 256, 256]},
    policy_optim_kwargs={"lr": 5e-5},
    vf_kwargs={"net_arch": [256, 256, 256, 256]},
    vf_optim_kwargs={"lr": 6e-4},
)
