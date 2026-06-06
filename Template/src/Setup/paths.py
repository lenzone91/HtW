from pathlib import Path
import shutil


#############################################
# Path setup
#############################################

# This file centralizes runtime output path preparation.
#
# It does not load configs.
# It does not save models, metrics, or logs.
# It only resolves and creates directories used by the rest of the project.


def setup_paths(
    paths_config: dict | None = None,
    project_root: str | Path | None = None,
) -> dict:
    """
    Resolve runtime paths, create required directories, and return path context.
    """
    if paths_config is None:
        paths_config = {}

    project_root = resolve_path(project_root or paths_config.get("project_root", "."))

    run_root = resolve_optional_path(
        paths_config.get("run_root", None),
        default=project_root / "results",
        root=project_root,
    )

    experiment_name = paths_config.get("experiment_name", "default_experiment")
    run_name = paths_config.get("run_name", "default_run")
    sweep_name = paths_config.get("sweep_name", None)
    overwrite = paths_config.get("overwrite", False)

    experiment_dir = run_root / experiment_name

    sweep_dir = build_sweep_dir(
        experiment_dir=experiment_dir,
        sweep_name=sweep_name,
    )

    run_dir = build_run_dir(
        experiment_dir=experiment_dir,
        sweep_dir=sweep_dir,
        run_name=run_name,
    )

    prepare_run_dir(
        run_dir=run_dir,
        overwrite=overwrite,
    )

    path_context = build_path_context(
        project_root=project_root,
        run_root=run_root,
        experiment_dir=experiment_dir,
        sweep_dir=sweep_dir,
        run_dir=run_dir,
    )

    create_run_subdirs(path_context)
    check_writable_dirs(path_context)

    return stringify_paths(path_context)


#############################################
# Path resolution
#############################################

# Path resolution is kept explicit so every downstream subsystem receives
# absolute paths and does not need to infer project-relative locations.


def resolve_path(
    path: str | Path,
    root: str | Path | None = None,
) -> Path:
    """
    Resolve a filesystem path to an absolute Path object.

    Relative paths are resolved from root when root is provided.
    Absolute paths are preserved.
    """
    path = Path(path).expanduser()

    if path.is_absolute():
        return path.resolve()

    if root is None:
        return path.resolve()

    return (Path(root).expanduser().resolve() / path).resolve()


def resolve_optional_path(
    path: str | Path | None,
    default: Path,
    root: str | Path | None = None,
) -> Path:
    """
    Resolve an optional path, using a default when no path is provided.
    """
    if path is None:
        return default.resolve()

    return resolve_path(path, root=root)


def build_sweep_dir(
    experiment_dir: Path,
    sweep_name: str | None,
) -> Path | None:
    """
    Build the sweep directory if the current run belongs to a sweep.
    """
    if sweep_name is None:
        return None

    return experiment_dir / "sweeps" / sweep_name


def build_run_dir(
    experiment_dir: Path,
    sweep_dir: Path | None,
    run_name: str,
) -> Path:
    """
    Build the run directory.

    Single runs and sweep runs are separated explicitly.
    """
    if sweep_dir is None:
        return experiment_dir / "single_runs" / run_name

    return sweep_dir / run_name


def build_path_context(
    project_root: Path,
    run_root: Path,
    experiment_dir: Path,
    sweep_dir: Path | None,
    run_dir: Path,
) -> dict:
    """
    Build the standard runtime path context.
    """
    return {
        "project_root": project_root,
        "run_root": run_root,
        "experiment_dir": experiment_dir,
        "sweep_dir": sweep_dir,
        "run_dir": run_dir,
        "checkpoints_dir": run_dir / "checkpoints",
        "logs_dir": run_dir / "logs",
        "metrics_dir": run_dir / "metrics",
        "configs_dir": run_dir / "configs",
        "artifacts_dir": run_dir / "artifacts",
        "is_sweep_run": sweep_dir is not None,
    }


#############################################
# Directory creation
#############################################


def prepare_run_dir(
    run_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Create the run directory, optionally overwriting an existing one.
    """
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}")

        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)


def create_run_subdirs(path_context: dict) -> None:
    """
    Create all standard run subdirectories.
    """
    for path_name in get_directory_path_names():
        path_context[path_name].mkdir(parents=True, exist_ok=True)


def get_directory_path_names() -> tuple[str, ...]:
    """
    Return path context keys that must correspond to directories.
    """
    return (
        "run_dir",
        "checkpoints_dir",
        "logs_dir",
        "metrics_dir",
        "configs_dir",
        "artifacts_dir",
    )


#############################################
# Path checks
#############################################


def check_writable_dirs(path_context: dict) -> None:
    """
    Check that all runtime directories are writable.
    """
    for path_name in get_directory_path_names():
        check_writable_dir(path_context[path_name])


def check_writable_dir(path: Path) -> None:
    """
    Check that one directory is writable.
    """
    test_file = path / ".write_test"

    try:
        test_file.touch()
        test_file.unlink()

    except OSError as error:
        raise PermissionError(f"Directory is not writable: {path}") from error


#############################################
# Context formatting
#############################################


def stringify_paths(path_context: dict) -> dict:
    """
    Convert Path objects in the path context to strings.
    """
    return {
        path_name: stringify_path_value(path)
        for path_name, path in path_context.items()
    }


def stringify_path_value(value):
    """
    Convert Path values to strings while preserving non-path values.
    """
    if isinstance(value, Path):
        return str(value)

    return value