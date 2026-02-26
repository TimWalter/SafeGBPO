from __future__ import annotations
from typing import TYPE_CHECKING, Any
import os
import ast
import sys
import importlib
import importlib.util
from pathlib import Path

from beartype import beartype
from jaxtyping import jaxtyped

from conf.safeguard import RayMaskConfig

if TYPE_CHECKING:
    from conf.experiment import Experiment


@jaxtyped(typechecker=beartype)
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
        elif cfg.safeguard.name == "FSNet":
            group += "FSNet"
            tags += ["FSNet"]
        elif cfg.safeguard.name == "Pinet":
            group += "Pinet"
            tags += ["Pinet"]
        if cfg.safeguard.regularisation_coefficient > 0:
            group += "(Reg)"
            tags += ["Regularised"]
    else:
        tags += ["Unsafe"]

    group += "-" + cfg.learning_algorithm.name
    tags += [cfg.learning_algorithm.name]

    group += "-" + cfg.env.name
    tags += [cfg.env.name]
    if hasattr(cfg.env, "num_obstacles"):
        group += f"(#Obs={str(cfg.env.num_obstacles)})"
        tags += [f"#Obs{cfg.env.num_obstacles}"]

    return group, tags


@jaxtyped(typechecker=beartype)
def import_module(modules: dict[str, str | Path], name: str):
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
    target = modules[name]

    if isinstance(target, str):
        module = importlib.import_module(target)
        return getattr(module, name)

    if isinstance(target, Path):
        file_path = target
        if not file_path.exists():
            raise FileNotFoundError(f"Config path not found: {file_path}")

        safe_name = (
                "config__" + str(file_path.resolve()).replace(os.sep, "_").replace(":", "_")
        )
        if safe_name in sys.modules:
            module = sys.modules[safe_name]
        else:
            spec = importlib.util.spec_from_file_location(safe_name, str(file_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec for {file_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[safe_name] = module
            spec.loader.exec_module(module)

        return getattr(module, name)


@jaxtyped(typechecker=beartype)
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


@jaxtyped(typechecker=beartype)
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


@jaxtyped(typechecker=beartype)
def _find_import_root(start: Path) -> Path:
    """
    Support PEP 420 namespace packages (no `__init__.py`).
    Find the nearest ancestor that is on sys.path (e.g., your `src` dir).

    Args:
        start: The starting path to search from.

    Returns:
        The path of the nearest ancestor that is on sys.path.
    """
    cur = start.resolve()
    syspaths: set[Path] = set()
    for p in sys.path:
        if isinstance(p, str):
            try:
                syspaths.add(Path(p).resolve())
            except Exception:
                pass
    while True:
        if cur in syspaths:
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError(
        f"Cannot find an import root on sys.path for {start}. Ensure its ancestor (e.g., `src`) is on sys.path."
    )


@jaxtyped(typechecker=beartype)
def _module_name_from_path(file_path: Path, import_root: Path) -> str:
    """
    Build a fully qualified module name from file path, starting at the
    import root (directory present on sys.path).

    Args:
        file_path: The path of the file to convert.
        import_root: The root directory from which to build the module name.

    Returns:
        The module name.
    """
    rel = file_path.resolve().relative_to(import_root)
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts)


@jaxtyped(typechecker=beartype)
def gather_custom_modules(directory: Path, subclass: str | None = None) -> dict[str, str]:
    """
    Gather all custom modules in a directory.

    Args:
        directory: The directory to search in.
        subclass: The subclass to search for.

    Returns:
        All custom modules found in the directory.
    """
    modules: dict[str, str] = {}
    python_files = find_python_files(directory)
    if not python_files:
        return modules

    import_root = _find_import_root(directory)

    for file_path_str in python_files:
        file_path = Path(file_path_str)
        if file_path.name == "__init__.py":
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        module_name = _module_name_from_path(file_path, import_root)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if subclass is not None:
                    if any(is_subclass(base, subclass) for base in node.bases):
                        modules[node.name] = module_name
                else:
                    modules[node.name] = module_name
    return modules
