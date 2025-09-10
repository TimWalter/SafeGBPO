from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

from optuna import Trial

if TYPE_CHECKING:
    from conf.experiment import Experiment


@dataclass
class LearningAlgorithmConfig:
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [64, 64],
    })
    policy_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 1e-4,
    })
    vf_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [128, 128],
    })
    vf_optim_kwargs: dict = field(default_factory=lambda: {
        "lr": 1e-4,
    })

    @property
    def name(self) -> str:
        return self.__class__.__name__[:-6]

    def vary(self, trial: Trial, cfg: Experiment):
        policy_depth = trial.suggest_int("policy_depth", 2, 4)
        policy_width = trial.suggest_int("policy_width", 32, 1024)
        self.policy_kwargs.net_arch = [policy_width] * policy_depth
        self.policy_optim_kwargs.lr = trial.suggest_float("policy_lr", 1e-6, 1e-2, log=True)

        vf_depth = trial.suggest_int("vf_depth", 2, 4)
        vf_width = trial.suggest_int("vf_width", 32, 1024)
        self.vf_kwargs.net_arch = [vf_width] * vf_depth
        self.vf_optim_kwargs.lr = trial.suggest_float("vf_lr", 1e-6, 1e-2, log=True)


@dataclass
class PPOConfig(LearningAlgorithmConfig):
    len_trajectories: int = 1700
    clip_coef: float = 0.2
    ent_coef: float = 0.09
    num_batches: int = 32
    num_fits: int = 32

    def vary(self, trial: Trial, cfg: Experiment):
        super().vary(trial, cfg)
        len_trajectories_in_eps = trial.suggest_int("len_trajectories_in_eps", 1, 5)
        self.len_trajectories = cfg.env.num_steps * len_trajectories_in_eps
        self.clip_coef = trial.suggest_float("clip_coef", 0.2, 0.5)
        self.ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)


@dataclass
class SACConfig(LearningAlgorithmConfig):
    buffer_size: int = 60_000
    polyak_target: float = 0.673
    learning_starts: int = 64
    policy_frequency: int = 6
    target_frequency: int = 7

    def vary(self, trial: Trial, cfg: Experiment):
        super().vary(trial, cfg)
        self.buffer_size = trial.suggest_int("buffer_size", cfg.env.num_steps * cfg.env.num_envs, cfg.interactions)
        self.polyak_target = trial.suggest_float("polyak_target", 0.1, 1.0)
        self.learning_starts = trial.suggest_int("learning_starts", 0, 0.1 * cfg.interactions)
        self.policy_frequency = trial.suggest_int("policy_frequency", 1, 10)
        self.target_frequency = trial.suggest_int("target_frequency", 1, 10)


@dataclass
class SHACConfig(LearningAlgorithmConfig):
    len_trajectories: int = 64
    polyak_target: float = 0.4
    td_weight: float = 0.95
    vf_num_fits: int = 50
    vf_fit_num_batches: int = 1

    def vary(self, trial: Trial, cfg: Experiment):
        super().vary(trial, cfg)
        self.len_trajectories = trial.suggest_int("len_trajectories", cfg.env.num_steps // 20, cfg.env.num_steps)
        self.polyak_target = trial.suggest_float("polyak_target", 0.1, 1.0)
        self.td_weight = trial.suggest_float("td_weight", 0.9, 0.99)
        self.vf_num_fits = trial.suggest_int("vf_num_fits", 8, 64)
        self.vf_fit_num_batches = trial.suggest_int("vf_fit_num_batches", 1, self.len_trajectories)
