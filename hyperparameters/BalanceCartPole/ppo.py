from conf.learning_algorithms import PPOConfig

config = PPOConfig(
    len_trajectories=1700,
    clip_coef=0.2,
    ent_coef=0.09,
    num_batches = 45,
    num_fits = 32,
    policy_kwargs={"net_arch": [256, 256, 256, 256]},
    policy_optim_kwargs={"lr": 3e-4},
    vf_kwargs={"net_arch": [200, 200]},
    vf_optim_kwargs={"lr": 2e-3},
)
