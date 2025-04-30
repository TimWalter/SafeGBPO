from gymnasium.vector import VectorWrapper

from tasks.envs.interfaces.torch_vector_env import TorchVectorEnv
from src.utils import import_module, gather_custom_modules


class EnvCreator:
    """
    Class to create an environment with wrappers dynamically from a Hydra config.

    Attributes:
        name (str): Name of the environment to create. (has to match actual class)
        params (dict): Parameters to pass to the environment.
        wrappers (list): List of wrappers to apply to the environment.
        creation (TorchVectorEnv|VectorWrapper): The created environment.
        modules (dict): Modules to import from for wrappers. Has to include all imports
                        that are not custom and therefore either in ./envs for environments
                        or wrappers.
    """

    def __init__(self, name: str, wrappers: list, **params: dict):
        """
        Args:
            name (str): Name of the environment to create. (has to match actual class)
            wrappers (list): List of wrappers to apply to the environment.
            **params (dict): Parameters to pass to the environment.
        """
        self.name = name
        self.params = params
        self.wrappers = wrappers
        self.creation = None
        self.modules = {}
        self.modules |= gather_custom_modules("./tasks", "Task")
        self.modules |= gather_custom_modules("./tasks/wrapper", "Wrapper")

    def _wrap(self, creation: TorchVectorEnv, name: str, **params: dict) -> VectorWrapper:
        """
        Wrap the environment with a wrapper.

        Args:
            creation (TorchVectorEnv|VectorWrapper): The environment to wrap.
            name (str): Name of the wrapper class.
            **params (dict): Parameters to pass to the wrapper.

        Returns:
            VectorWrapper: The wrapped environment.
        """
        wrapper_class = import_module(self.modules, name)
        return wrapper_class(creation, **params)


    def create(self) -> TorchVectorEnv|VectorWrapper:
        """
        Create the environment with all wrappers.

        Returns:
            TorchVectorEnv|VectorWrapper: The created environment.
        """

        env_class = import_module(self.modules, self.name)
        self.creation = env_class(**self.params)
        for wrapper in self.wrappers:
            self.creation = self._wrap(self.creation, **wrapper)
        return self.creation
