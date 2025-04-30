import ast
import importlib
import os
from typing import Callable, Any

from cvxpylayers.torch import CvxpyLayer


class PassthroughCvxpyLayer(CvxpyLayer):
    @staticmethod
    def backward(ctx, *dvars):
        return dvars


def find_python_files(directory: str) -> list[str]:
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

    if module_path.endswith('.py'):  # Custom
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, name)
    else:
        module = importlib.import_module(module_path)
        return getattr(module, name)


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


def gather_custom_modules(directory: str, subclass: str = None) -> dict:
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
