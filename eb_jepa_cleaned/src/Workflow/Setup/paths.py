"""Run-directory and path resolution for the runtime context.

Setup owns run-directory creation (rather than Hydra's job/output-dir machinery),
so the existing-run-dir policy stays explicit and overridable by a launcher.
"""

import shutil
import sys
from pathlib import Path

EXISTING_RUN_DIR_POLICIES = {"fail", "overwrite", "ask"}
RUN_SUBDIRS = ("checkpoints_dir", "logs_dir", "configs_dir", "artifacts_dir")


def setup_paths(paths_config: dict | None = None) -> dict:
    """
    Resolve the run directory (and its subdirectories) and create them on disk
    according to the existing-run-dir policy. Returns a dict of string paths.
    """
    paths_config = dict(paths_config or {})

    project_root = _resolve_path(paths_config.get("project_root") or Path.cwd())
    run_root = _resolve_path(paths_config.get("run_root", "runs"), root=project_root)
    experiment_dir = run_root / paths_config.get("experiment_name", "ac_video_jepa")
    run_dir = experiment_dir / paths_config.get("run_name", "default_run")

    policy = paths_config.get("existing_run_dir_policy", "fail")
    _prepare_run_dir(run_dir, policy)

    paths = {
        "project_root": project_root,
        "run_root": run_root,
        "experiment_dir": experiment_dir,
        "run_dir": run_dir,
        "checkpoints_dir": run_dir / "checkpoints",
        "logs_dir": run_dir / "logs",
        "configs_dir": run_dir / "configs",
        "artifacts_dir": run_dir / "artifacts",
    }
    for key in RUN_SUBDIRS:
        paths[key].mkdir(parents=True, exist_ok=True)

    return {key: str(value) for key, value in paths.items()}


def _resolve_path(path, root: Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if root is None:
        return path.resolve()
    return (Path(root).expanduser().resolve() / path).resolve()


def _prepare_run_dir(run_dir: Path, policy: str) -> None:
    if policy not in EXISTING_RUN_DIR_POLICIES:
        raise ValueError(
            f"Invalid existing_run_dir_policy '{policy}'. "
            f"Expected one of {sorted(EXISTING_RUN_DIR_POLICIES)}."
        )

    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        return

    if policy == "fail":
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    if policy == "ask":
        _confirm_deletion(run_dir)

    shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _confirm_deletion(run_dir: Path) -> None:
    if not sys.stdin.isatty():
        raise FileExistsError(
            "Run directory exists and existing_run_dir_policy='ask' cannot prompt "
            f"in a non-interactive environment: {run_dir}"
        )
    answer = input(f"Run directory exists. Delete and recreate it? {run_dir} [y/N]: ")
    if answer.strip().lower() not in {"y", "yes"}:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
