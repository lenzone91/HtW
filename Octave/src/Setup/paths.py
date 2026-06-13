from pathlib import Path
import shutil
import sys


EXISTING_RUN_DIR_POLICIES = {"fail", "overwrite", "ask"}


def setup_paths(
    paths_config: dict | None = None,
    config_path: str | Path | None = None,
) -> dict:
    paths_config = dict(paths_config or {})

    project_root = resolve_project_root(
        project_root=paths_config.get("project_root"),
        config_path=config_path,
    )
    run_root = resolve_path(
        paths_config.get("run_root", "Octave/runs"),
        root=project_root,
    )

    experiment_name = paths_config.get("experiment_name", "ac_video_jepa")
    run_name = paths_config.get("run_name", "default_run")
    existing_run_dir_policy = resolve_existing_run_dir_policy(paths_config)

    experiment_dir = run_root / experiment_name
    run_dir = experiment_dir / run_name

    prepare_run_dir(
        run_dir=run_dir,
        existing_run_dir_policy=existing_run_dir_policy,
    )

    path_context = {
        "project_root": project_root,
        "run_root": run_root,
        "experiment_dir": experiment_dir,
        "run_dir": run_dir,
        "checkpoints_dir": run_dir / "checkpoints",
        "logs_dir": run_dir / "logs",
        "metrics_dir": run_dir / "metrics",
        "configs_dir": run_dir / "configs",
        "artifacts_dir": run_dir / "artifacts",
    }

    create_run_subdirs(path_context)
    return stringify_paths(path_context)


def resolve_project_root(
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    if project_root is not None:
        return resolve_path(project_root)

    if config_path is not None:
        config_path = Path(config_path).expanduser().resolve()
        for parent in (config_path.parent, *config_path.parents):
            if (parent / "pyproject.toml").exists():
                return parent.resolve()

    return Path.cwd().resolve()


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    path = Path(path).expanduser()

    if path.is_absolute():
        return path.resolve()

    if root is None:
        return path.resolve()

    return (Path(root).expanduser().resolve() / path).resolve()


def resolve_existing_run_dir_policy(paths_config: dict) -> str:
    policy = paths_config.get("existing_run_dir_policy")

    if policy is None:
        policy = "overwrite" if paths_config.get("overwrite", False) else "fail"

    if policy not in EXISTING_RUN_DIR_POLICIES:
        raise ValueError(
            "Invalid existing_run_dir_policy: "
            f"{policy}. Expected one of {sorted(EXISTING_RUN_DIR_POLICIES)}."
        )

    return policy


def prepare_run_dir(
    run_dir: Path,
    existing_run_dir_policy: str,
) -> None:
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        return

    if existing_run_dir_policy == "fail":
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    if existing_run_dir_policy == "overwrite":
        shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return

    if existing_run_dir_policy == "ask":
        confirm_run_dir_deletion(run_dir)
        shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return

    raise ValueError(
        "Invalid existing_run_dir_policy: "
        f"{existing_run_dir_policy}. Expected one of "
        f"{sorted(EXISTING_RUN_DIR_POLICIES)}."
    )


def confirm_run_dir_deletion(run_dir: Path) -> None:
    if not sys.stdin.isatty():
        raise FileExistsError(
            "Run directory already exists and existing_run_dir_policy='ask' "
            "cannot prompt in a non-interactive environment: "
            f"{run_dir}"
        )

    answer = input(
        "Run directory already exists. Delete and recreate it? "
        f"{run_dir} [y/N]: "
    )

    if answer.strip().lower() not in {"y", "yes"}:
        raise FileExistsError(f"Run directory already exists: {run_dir}")


def create_run_subdirs(path_context: dict) -> None:
    for key in (
        "run_dir",
        "checkpoints_dir",
        "logs_dir",
        "metrics_dir",
        "configs_dir",
        "artifacts_dir",
    ):
        path_context[key].mkdir(parents=True, exist_ok=True)


def stringify_paths(path_context: dict) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in path_context.items()
    }
