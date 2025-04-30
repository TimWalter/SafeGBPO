from optuna import Trial
from wandb.sdk.wandb_run import Run

from src.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from src.callbacks.train_callback import TrainCallback
from tasks.envs.interfaces.torch_vector_env import TorchVectorEnv
from src.utils import import_module, gather_custom_modules


class Trainer:
    """
    Trainer class that creates the model and trains it.

    Attributes:
        run (Run): The wandb run object.
        trial (Trial): The optuna trial object.
        env (TorchVectorEnv): The training environment.
        eval_env (TorchVectorEnv): The evaluation environment.
        env_name (str): The name of the environment to use.
        wrapper_names (list[str]): The names of the wrappers to apply to the environment.
        samples (int): The number of samples to train on.
        callback (dict): The parameters for the TrainingCallback.
        name (str): The name of the algorithm to use. (Hast to match class name)
        hyperparameters (dict): The hyperparameters to pass to the algorithm.
        modules (dict): The modules to import the algorithms from.
        model (ActorCriticAlgorithm): The trained model.
    """

    def __init__(self,
                 run: Run,
                 trial: Trial | None,
                 env: TorchVectorEnv,
                 eval_env: TorchVectorEnv,
                 env_name: str,
                 wrapper_names: list[str],
                 samples: int,
                 callback: dict,
                 name: str,
                 **hyperparameters: dict):
        """

        Args:
            run (Run): The wandb run object.
            trial (Trial | None): The optuna trial object.
            env (TorchVectorEnv): The training environment.
            eval_env (TorchVectorEnv): The evaluation environment.
            env_name (str): The name of the environment to use.
            wrapper_names (list[str]): The names of the wrappers to apply to the environment.
            samples (int): The number of samples to train on.
            callback (dict): The parameters for the TrainingCallback.
            name (str): The name of the algorithm to use. (Hast to match class name)
            **hyperparameters (dict): The hyperparameters to pass to the algorithm
        """
        self.run = run
        self.trial = trial
        self.env = env
        self.eval_env = eval_env
        self.env_name = env_name
        self.wrapper_names = wrapper_names
        self.samples = samples
        self.callback = callback
        self.name = name
        self.hyperparameters = hyperparameters

        self.modules = gather_custom_modules("./algorithms", "ActorCriticAlgorithm")
        self.model = None


    def train(self) -> float:
        """
        Train the model and return the best reward.

        Returns:
            float: The best episodic reward achieved during training.
        """
        self.model = import_module(self.modules, self.name)(**self.hyperparameters,
                                                            env=self.env)
        callback = TrainCallback(model=self.model,
                                 algo_name= self.name,
                                 eval_env=self.eval_env,
                                 env_name=self.env_name,
                                 wrapper_names=self.wrapper_names,
                                 wandb_run=self.run,
                                 optuna_trial=self.trial,
                                 **self.callback)
        self.model.learn(samples=self.samples, callback=callback)

        return callback.best_reward
