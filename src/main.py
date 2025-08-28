from pathlib import Path
from dataclasses import asdict
from typing import Optional

import torch
import wandb
import optuna

from logger import Logger
from utils import categorise_run, import_module, gather_custom_modules
from conf.experiment import Experiment

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def run_experiment(cfg: Experiment, trial: Optional[optuna.Trial] = None) -> float:
    if trial is not None:
        cfg.learning_algorithm.vary(trial, cfg)

    group, tags = categorise_run(cfg)

    run = wandb.init(project="Leveraging Analytical Gradients for Provably Safe Reinforcement Learning",
                     config=asdict(cfg),
                     monitor_gym=True,
                     group=group,
                     tags=tags)

    if trial is not None:
        run.name = f"trial/{trial.number}/{run.name}"
        run.config.update(trial.params)

    run.config["config"] = asdict(cfg)

    modules = gather_custom_modules(Path(__file__).parent / "envs", "Env")
    modules |= gather_custom_modules(Path(__file__).parent / "safeguards", "Safeguard")
    modules |= gather_custom_modules(Path(__file__).parent / "learning_algorithms", "LearningAlgorithm")

    env_class = import_module(modules, cfg.env.name + "Env")
    env = env_class(**asdict(cfg.env))
    cfg.env.num_envs = env.EVAL_ENVS
    eval_env = env_class(**asdict(cfg.env))

    if cfg.safeguard:
        safeguard_class = import_module(modules, cfg.safeguard.name + "Safeguard")
        env = safeguard_class(env, **asdict(cfg.safeguard))
        eval_env = safeguard_class(eval_env, **asdict(cfg.safeguard))

    agent = import_module(modules, cfg.learning_algorithm.name)(**vars(cfg.learning_algorithm), env=env)
    logger = Logger(agent, env, eval_env, run, trial, cfg.eval_freq, cfg.fast_eval)
    agent.learn(interactions=cfg.interactions, logger=logger)

    run.finish()

    return logger.best_reward


if __name__ == "__main__":
    from conf.envs import *
    from conf.safeguard import *
    from conf.learning_algorithms import *

    wandb.login(key="")

    experiment_queue = [
        Experiment(num_runs=1,
                   learning_algorithm=SHACConfig(),
                   env=BalanceQuadrotorConfig(),
                   safeguard=RayMaskConfig(zonotopic_approximation=False),
                   interactions=15_000,
                   eval_freq=5_000,
                   fast_eval=False),
    ]

    for i, experiment in enumerate(experiment_queue):
        if experiment.num_runs == 0:
            print("[STATUS] Running hyperparameter search")

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(),
                                        pruner=optuna.pruners.HyperbandPruner(),
                                        storage=f"sqlite:///hyperparameters/{experiment.env.name}/study.sqlite3",
                                        study_name=experiment.learning_algorithm.name)

            pre_valued_objective = lambda trial: run_experiment(experiment, trial)
            study.optimize(pre_valued_objective, n_trials=100, n_jobs=1)
            print(f"Best value: {study.best_value} (params: {study.best_params})")

        else:
            print(f"[STATUS] Running experiment [{i + 1}/{len(experiment_queue)}]")
            for j in range(experiment.num_runs):
                run_experiment(experiment)
