from pathlib import Path


#############################################
# Project root setup
#############################################

# This file centralizes project root inference.
#
# The project root is the root of one runnable project version.
# It is the directory from which relative runtime paths are resolved.


def infer_project_root(config_path: str | Path) -> Path:
    """
    Infer the project root from a config path.

    The project root is defined as the nearest parent directory containing
    both a src/ directory and a configs/ directory.
    """
    path = Path(config_path).expanduser().resolve()

    candidate = path if path.is_dir() else path.parent

    for parent in (candidate, *candidate.parents):
        if is_project_root(parent):
            return parent

    raise FileNotFoundError(
        "Could not infer project root from config path. "
        "Expected one parent directory to contain both src/ and configs/."
    )


def is_project_root(path: str | Path) -> bool:
    """
    Return True if path looks like the root of one runnable project version.
    """
    path = Path(path)

    return (
        (path / "src").is_dir()
        and (path / "configs").is_dir()
    )


def resolve_project_root(
    project_root: str | Path | None,
    config_path: str | Path | None = None,
) -> Path:
    """
    Resolve an explicit project root or infer it from the config path.
    """
    if project_root is not None:
        resolved_project_root = Path(project_root).expanduser().resolve()
        check_project_root(resolved_project_root)
        return resolved_project_root

    if config_path is None:
        raise ValueError(
            "project_root is None and no config_path was provided for inference."
        )

    return infer_project_root(config_path)


def check_project_root(project_root: str | Path) -> None:
    """
    Validate that a path is a project root.
    """
    project_root = Path(project_root)

    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    if not project_root.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {project_root}")

    if not is_project_root(project_root):
        raise ValueError(
            "Invalid project root. Expected a directory containing both "
            f"src/ and configs/: {project_root}"
        )