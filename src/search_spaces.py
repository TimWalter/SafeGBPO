from omegaconf import DictConfig
from optuna import Trial


def shac_modifications(trial: Trial, cfg: DictConfig) -> None:
    """
    Hyperparameter search space for SHAC algorithm.

    Args:
        trial (Trial): Optuna trial object.
        cfg (DictConfig): OmegaConf DictConfig object.
    """
    cfg.algorithm.len_trajectories = trial.suggest_int("len_trajectories",
                                                       cfg.env.max_episode_steps // 20,
                                                       cfg.env.max_episode_steps)
    cfg.algorithm.polyak_target = trial.suggest_float("polyak_target",
                                                      0.1,
                                                      1.0)
    cfg.algorithm.td_weight = trial.suggest_float("td_weight", 0.9, 0.99)

    policy_depth = trial.suggest_int("policy_depth", 2, 4)
    policy_width = trial.suggest_int("policy_width", 32, 2048)
    cfg.algorithm.policy_kwargs.net_arch = [policy_width] * policy_depth
    cfg.algorithm.policy_optim_kwargs.lr = trial.suggest_float("policy_lr",
                                                               1e-6,
                                                               1e-2,
                                                               log=True)

    vf_depth = trial.suggest_int("vf_depth", 2, 4)
    vf_width = trial.suggest_int("vf_width", 32, 1024)
    cfg.algorithm.vf_kwargs.net_arch = [vf_width] * vf_depth
    cfg.algorithm.vf_optim_kwargs.lr = trial.suggest_float("vf_lr",
                                                           1e-6,
                                                           1e-2,
                                                           log=True)

    cfg.algorithm.vf_num_fits = trial.suggest_int("vf_num_fits", 8, 64)
    cfg.algorithm.vf_fit_num_batches = trial.suggest_int("vf_fit_num_batches",
                                                         1,
                                                         cfg.algorithm.len_trajectories)

    cfg.algorithm.max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.5)


def sac_modifications(trial: Trial, cfg: DictConfig) -> None:
    """
    Hyperparameter search space for SAC algorithm.

    Args:
        trial (Trial): Optuna trial object.
        cfg (DictConfig): OmegaConf DictConfig object.
    """

    cfg.algorithm.buffer_size = trial.suggest_int("buffer_size",
                                                  cfg.env.max_episode_steps * cfg.env.num_envs,
                                                  cfg.samples)
    cfg.algorithm.polyak_target = trial.suggest_float("polyak_target",
                                                      0.1,
                                                      1.0)
    cfg.algorithm.learning_starts = trial.suggest_int("learning_starts",
                                                      0,
                                                      0.1 * cfg.samples)
    cfg.algorithm.policy_frequency = trial.suggest_int("policy_frequency",
                                                       1,
                                                       10)
    cfg.algorithm.target_frequency = trial.suggest_int("target_frequency",
                                                       1,
                                                       10)

    policy_depth = trial.suggest_int("policy_depth", 2, 4)
    policy_width = trial.suggest_int("policy_width", 32, 1024)
    cfg.algorithm.policy_kwargs.net_arch = [policy_width] * policy_depth
    cfg.algorithm.policy_optim_kwargs.lr = trial.suggest_float("policy_lr",
                                                               1e-6,
                                                               1e-2,
                                                               log=True)

    vf_depth = trial.suggest_int("vf_depth", 2, 4)
    vf_width = trial.suggest_int("vf_width", 32, 1024)
    cfg.algorithm.vf_kwargs.net_arch = [vf_width] * vf_depth
    cfg.algorithm.vf_optim_kwargs.lr = trial.suggest_float("vf_lr",
                                                           1e-6,
                                                           1e-2,
                                                           log=True)


def ppo_modifications(trial: Trial, cfg: DictConfig) -> None:
    """
    Hyperparameter search space for PPO algorithm.

    Args:
        trial (Trial): Optuna trial object.
        cfg (DictConfig): OmegaConf DictConfig object.
    """
    len_trajectories_in_eps = trial.suggest_int("len_trajectories_in_eps", 1, 5)
    cfg.algorithm.len_trajectories = cfg.env.max_episode_steps * len_trajectories_in_eps

    cfg.algorithm.clip_coef = trial.suggest_float("clip_coef", 0.2, 0.5)

    cfg.algorithm.ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)

    policy_depth = trial.suggest_int("policy_depth", 2, 4)
    policy_width = trial.suggest_int("policy_width", 32, 1024)
    cfg.algorithm.policy_kwargs.net_arch = [policy_width] * policy_depth
    cfg.algorithm.policy_optim_kwargs.lr = trial.suggest_float("policy_lr",
                                                               1e-6,
                                                               1e-2,
                                                               log=True)

    vf_depth = trial.suggest_int("vf_depth", 2, 4)
    vf_width = trial.suggest_int("vf_width", 32, 1024)
    cfg.algorithm.vf_kwargs.net_arch = [vf_width] * vf_depth
    cfg.algorithm.vf_optim_kwargs.lr = trial.suggest_float("vf_lr",
                                                           1e-6,
                                                           1e-2,
                                                           log=True)


modifications = {
    "SHAC": shac_modifications,
    "SAC": sac_modifications,
    "PPO": ppo_modifications
}
