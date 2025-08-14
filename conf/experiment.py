from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from utils import import_module
from conf.envs import EnvConfig
from conf.safeguard import SafeguardConfig
from conf.learning_algorithms import LearningAlgorithmConfig

@dataclass
class Experiment:
    num_runs: int  # 0 = Hyperparameter search
    learning_algorithm: LearningAlgorithmConfig
    env: EnvConfig
    safeguard: Optional[SafeguardConfig]
    interactions: int
    eval_freq: int
    fast_eval: bool

    def __post_init__(self):
        """
        Load the hyperparameters and set the regularisation coefficient if applicable.
        """
        file_path = Path(__file__).parent.parent / "hyperparameters" / self.env.name / f"{self.learning_algorithm.name.lower()}.py"
        self.learning_algorithm = import_module({"config": file_path}, "config")
        if hasattr(self.safeguard, "regularisation_coefficient"):
            self.learning_algorithm.regularisation_coefficient = self.safeguard.regularisation_coefficient
        else:
            self.learning_algorithm.regularisation_coefficient = 0.0
