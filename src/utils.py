from __future__ import annotations
from typing import TYPE_CHECKING
import os
import ast
from pathlib import Path
from typing import Callable, Any
from importlib.util import spec_from_file_location, module_from_spec

from conf.safeguard import RayMaskConfig


if TYPE_CHECKING:
    from conf.experiment import Experiment

def categorise_run(cfg: Experiment) -> tuple[str, list[str]]:
    """ Categorise the run based on the configuration.
    Args:
        cfg: The configuration of the experiment.
    Returns:
        A tuple containing the group name and a list of tags.
    """
    group = ""
    tags = []

    if cfg.safeguard:
        if cfg.safeguard.name == "BoundaryProjection":
            group += "BP"
            tags += ["BoundaryProjection"]
        elif isinstance(cfg.safeguard, RayMaskConfig):
            group += "RM"
            tags += ["RayMask"]

            if cfg.safeguard.zonotopic_approximation:
                group += "(Z)"
                tags += ["Zonotopic"]
            else:
                group += "(O)"
                tags += ["Orthogonal"]
            if cfg.safeguard.linear_projection:
                group += "(Lin)"
                tags += ["Linear"]
            else:
                group += "(Tanh)"
                tags += ["Hyperbolic"]
            if cfg.safeguard.passthrough:
                group += "(PT)"
                tags += ["Passthrough"]
        if cfg.safeguard.regularisation_coefficient > 0:
            group += "(Reg)"
            tags += ["Regularised"]
    else:
        tags += ["Unsafe"]

    group += "-" + cfg.learning_algorithm.name
    tags += [cfg.learning_algorithm.name]

    group += "-" + cfg.env.name
    tags += [cfg.env.name]
    if hasattr(cfg.env, 'num_obstacles'):
        group += f"(#Obs={str(cfg.env.num_obstacles)})"
        tags += [f"#Obs{cfg.env.num_obstacles}"]

    return group, tags


def import_module(modules: dict, name: str) -> Callable:
    """
    Import a class from a module by name.

    Args:
        modules: A list of modules to search in.
        name: The name of the module to import.

    Returns:
        The constructor of the class.
    """
    if name not in modules:
        raise ValueError(f"Module {name} is not recognized.")

    module_path = modules[name]

    spec = spec_from_file_location(name, module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def find_python_files(directory: Path) -> list[str]:
    """
    Find all Python files in a directory and its subdirectories.

    Args:
        directory: The directory to search in.

    Returns:
        A list of all Python files found in the directory.
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def is_subclass(base: Any, subclass: str) -> bool:
    """
    Check if a node base is a subclass of subclass.

    Args:
        base: The base node to check.
        subclass: The subclass to check for.

    Returns:
        True if base is a subclass of subclass, False otherwise.
    """
    if isinstance(base, ast.Name):
        return subclass in base.id
    elif isinstance(base, ast.Attribute):
        return subclass in base.attr
    return False


def gather_custom_modules(directory: Path, subclass: str = None) -> dict:
    """
    Gather all custom modules in a directory.

    Args:
        directory: The directory to search in.
        subclass: The subclass to search for.

    Returns:
        A dictionary of all custom modules found in the directory.
    """
    modules = {}
    python_files = find_python_files(directory)
    for file_path in python_files:
        with open(file_path, 'r') as file:
            tree = ast.parse(file.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if subclass is not None:
                    if any(is_subclass(base, subclass) for base in node.bases):
                        modules[node.name] = file_path
                else:
                    modules[node.name] = file_path
    return modules
