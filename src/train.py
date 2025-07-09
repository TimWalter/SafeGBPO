import os
import shutil

import hydra
import optuna
import tqdm
import wandb
import torch
from omegaconf import OmegaConf

from env_creator import EnvCreator
from hydra_registrator import HydraRegistrator
from search_spaces import modifications
from trainer import Trainer
from utils import import_module

tqdm.tqdm.monitor_interval = 0
os.environ["WANDB_SILENT"] = "false"


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def objective(trial: optuna.Trial | None, experiment) -> float:
    registrator = HydraRegistrator(search_path="../conf/")
    registrator.register_all()
    with hydra.initialize(version_base=None, config_path=None):
        cfg = hydra.compose(config_name="Config")

        load_hyperparameters(cfg, experiment[0], experiment[1], experiment[3])
        load_env_and_projection(cfg, experiment[0], experiment[2], experiment[4], experiment[5])

        if trial is not None:
            seed = 42
            set_seed(seed)
            modifications[cfg.algorithm.name](trial, cfg)

        cfg.wandb.group = ""
        cfg.wandb.tags = [cfg.env.name[:-4]]
        cfg.wandb.group += cfg.env.name[:-4] + "-"
        if "Navigate" in cfg.env.name:
            cfg.wandb.group += str(cfg.env.num_obstacles) + "-"
            cfg.wandb.tags += [f"#Obs{cfg.env.num_obstacles}"]
        if cfg.env.wrappers:
            if cfg.env.wrappers[0].name == "BoundaryProjectionWrapper":
                cfg.wandb.group += "BP-"
                cfg.wandb.tags += ["BoundaryProjection"]
            elif cfg.env.wrappers[0].name == "OrthogonalRayMapWrapper":
                cfg.wandb.group += "ORM-"
                cfg.wandb.tags += ["OrthogonalRayMap"]
                if cfg.env.wrappers[0].linear_projection:
                    cfg.wandb.group += "Lin-"
                    cfg.wandb.tags += ["Linear"]
                else:
                    cfg.wandb.group += "Tanh-"
                    cfg.wandb.tags += ["Tanh"]
            elif cfg.env.wrappers[0].name == "ZonotopeRayMapWrapper":
                cfg.wandb.group += "ZRM-"
                cfg.wandb.tags += ["ZonotopeRayMap"]
                if cfg.env.wrappers[0].linear_projection:
                    cfg.wandb.group += "Lin-"
                    cfg.wandb.tags += ["Linear"]
                else:
                    cfg.wandb.group += "Tanh-"
                    cfg.wandb.tags += ["Tanh"]
                if cfg.env.wrappers[0].passthrough:
                    cfg.wandb.group += "PT-"
                    cfg.wandb.tags += ["Passthrough"]
        else:
            cfg.wandb.group += "NP-"
            cfg.wandb.tags += ["NoProjection"]
        cfg.wandb.group += cfg.algorithm.name
        cfg.wandb.tags += [cfg.algorithm.name]
        if cfg.algorithm.name == "SHAC":
            if cfg.algorithm.adaptive_regularisation:
                cfg.wandb.group += "-AR"
                cfg.wandb.tags += ["AdaptiveRegularisation"]
            elif cfg.algorithm.regularisation_coefficient > 0:
                cfg.wandb.group += "-Reg"
                cfg.wandb.tags += ["Regularisation"]

        run = wandb.init(**cfg.wandb)

        if trial is not None:
            run.name = f"trial/{trial.number}/{run.name}"
            run.tags += (study.study_name,)
            run.config.update(trial.params)

        cfg.artefact_dir = f"../artefacts/{run.name}"
        os.makedirs(cfg.artefact_dir, exist_ok=True)
        if os.path.exists("../artefacts/.hydra"):
            shutil.move("../artefacts/.hydra", cfg.artefact_dir + "/.hydra")

        run.config["config"] = OmegaConf.to_container(cfg, resolve=True)

        creator = EnvCreator(**cfg.env)
        env = creator.create()
        creator.params["num_envs"] = 1
        if "draw_safe_state_set" in creator.params and "NoProjection" not in cfg.wandb.tags:
            creator.params["draw_safe_state_set"] = True
        eval_env = creator.create()

        trainer = Trainer(**cfg.algorithm, samples=cfg.samples, callback=cfg.callback,
                          env=env, eval_env=eval_env, env_name=cfg.env.name,
                          wrapper_names=[wrapper.name for wrapper in cfg.env.wrappers],
                          run=run, trial=trial)

        best_reward = trainer.train()

        run.finish()

        return best_reward


def load_hyperparameters(cfg, env, algo, reg):
    file_path = f"../hyperparameters/{env}/{algo.lower()}.py"
    class_name = f"{algo}Config"
    if os.path.exists(file_path):
        config_class = import_module({class_name: file_path}, class_name)
        algorithm_config = config_class()
        if reg and hasattr(algorithm_config, "regularisation_coefficient"):
            algorithm_config.regularisation_coefficient = 0.1
        cfg.algorithm = OmegaConf.structured(algorithm_config)
    else:
        raise FileNotFoundError(f"Can't find hyperparameters: \"{file_path}\"")


def load_env_and_projection(cfg, env, proj, tanh, passthrough):
    env_class_name = f"{env}Config"
    env_class = import_module({env_class_name: "../conf/task/tasks.py"}, env_class_name)
    env_config = env_class()
    cfg.env = OmegaConf.structured(env_config)
    cfg.samples = samples[cfg.env.name[:-4]]
    cfg.callback.eval_freq = eval_freq[cfg.env.name[:-4]]

    if proj == "":
        cfg.env.wrappers = []
    else:
        proj_class_name = f"{proj}Config"
        proj_class = import_module({proj_class_name: "../conf/task/wrapper.py"},
                                   proj_class_name)

        state_dim = state_dims[[key for key in state_dims.keys() if key in env][0]]
        action_dim = action_dims[[key for key in action_dims.keys() if key in env][0]]
        if env != "Household":
            proj_config = proj_class(
                lin_state=[0.0] * state_dim,
                lin_action=[0.0] * action_dim,
                lin_noise=[0.0] * state_dim
            )
        else:
            proj_config = proj_class(
                lin_state=[5.0, 21.0, 55.0],
                lin_action=[0.0, 0.0],
                lin_noise=[0.0, 10.8985, 0.0],
            )


        if hasattr(proj_config, "num_generators"):
            proj_config.num_generators = 2 * action_dim if action_dim != 1 else 1
        if hasattr(proj_config, "linear_projection") and tanh:
            proj_config.linear_projection = False
        if hasattr(proj_config, "passthrough") and passthrough:
            proj_config.passthrough = True
        cfg.env.wrappers = OmegaConf.structured([proj_config])

state_dims = {
    "Pendulum": 2,
    "CartPole": 4,
    "Quadrotor": 6,
    "Seeker": 2,
    "Household": 3,
}
action_dims = {
    "Pendulum": 1,
    "CartPole": 1,
    "Quadrotor": 2,
    "Seeker": 2,
    "Household": 2
}
samples = {
    "BalancePendulum": 100_000,
    "BalanceQuadrotor": 200_000,
    "NavigateSeeker": 10_000_000,
    "LoadBalanceHousehold": 100_000,
}

eval_freq = {
    "BalancePendulum": 2_500,
    "BalanceQuadrotor": 5_000,
    "NavigateSeeker": 50_000,
    "LoadBalanceHousehold": 2_500,
}

# FLAGS
# 0 = Hyperparameter search
experiment_queue = [
    [1, "LoadBalanceHousehold", "SHAC", "BoundaryProjection", False, False, False],
]

if __name__ == "__main__":
    wandb.login(key="")

    for experiment in experiment_queue:
        if experiment[0] == 0:
            status = "[STATUS] Running hyperparameter search: "
        else:
            status = f"[STATUS] Running experiment [{experiment[0]}]: "
        status += f"{experiment[1]}-{experiment[2]}-{experiment[3]}{"-Reg" if experiment[4] else ""}"
        status += f"{"-Tanh" if experiment[5] else ""}{"-PT" if experiment[6] else ""}"

        for i in range(experiment[0]):
            objective(None, experiment[1:])

        if experiment[0] == 0:
            sampler = optuna.samplers.TPESampler()
            pruner = optuna.pruners.HyperbandPruner()

            study = optuna.create_study(direction="maximize",
                                        sampler=sampler,
                                        pruner=pruner,
                                        storage="sqlite:///navigate_seeker_soft.sqlite3",
                                        study_name=experiment[2])

            pre_valued_objective = lambda trial: objective(trial, experiment[1:])
            study.optimize(pre_valued_objective, n_trials=100, n_jobs=1)
            print(f"Best value: {study.best_value} (params: {study.best_params})")
