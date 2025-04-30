from dataclasses import field, dataclass
from typing import Any

from omegaconf import MISSING

from conf.task.envs import TorchVectorConfig
from conf.task.tasks import *

@dataclass
class WandbConfig:
    project: str = "Safe Differentiable Reinforcement Learning"
    monitor_gym: bool = True
    save_code: bool = False
    sync_tensorboard: bool = False
    tags: list[str] = field(default_factory=lambda: [])
    dir: str = "../artefacts"
    group: str = "$TEMP"

@dataclass
class TrainCallbackConfig: 
    eval_freq: int = 5_000
    log_eval: bool = False
    log_vf: bool = False
    fast_eval: bool = True

@dataclass
class Config:
    defaults: list[Any] = field(default_factory=lambda: [
        "_self_",
        {"algorithm": "SHAC"}
    ])

    algorithm: Any = MISSING
    env: TorchVectorConfig = field(default_factory=lambda:BalancePendulumConfig)

    artefact_dir: str = MISSING

    wandb: WandbConfig = field(default_factory=WandbConfig)
    samples: int = 100_000
    callback: TrainCallbackConfig = field(default_factory=TrainCallbackConfig)

    hydra: dict = field(default_factory=lambda: {
        "run": {"dir": "../artefacts"},
        "mode": "RunMode.RUN"
    })
