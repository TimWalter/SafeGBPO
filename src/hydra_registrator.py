import importlib.util
import inspect
import os
from dataclasses import is_dataclass

from hydra.core.config_store import ConfigStore

from utils import find_python_files


class HydraRegistrator:
    """
    A class to register structured configurations with Hydra.
    """

    def __init__(self, search_path: str):
        """
        Initialize the HydraRegistrator.

        Args:
            search_path (str): The path to search for Python files to register.
        """
        self.search_path = search_path
        self.cs = ConfigStore.instance()

    def register_all(self, structured: bool = True) -> None:
        """
        Register all configurations in the search path.

        Args:
            structured (bool): Whether to register structured configurations.
        """
        if structured:
            python_files = find_python_files(self.search_path)
            for file_path in python_files:
                self.register_structured_config(file_path)
        else:
            raise NotImplementedError(
                "Only structured registration is supported for now.")

    def register_structured_config(self, file_path: str) -> None:
        """
        Dynamically import a Python file and register all dataclass objects,
        as structured configurations.

        Args:
            file_path (str): The path to the Python file to
        """
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        relative_path = os.path.relpath(file_path, self.search_path)
        group = os.path.dirname(relative_path).replace(os.sep, "/")

        for name, obj in inspect.getmembers(module):
            if is_dataclass(obj):
                if name == "Config":
                    register_name = "Config"
                else:
                    register_name = name.replace("Config", "")
                self.cs.store(name=register_name, node=obj,
                              group=group if group else None)
