"""
Tests for runtime path setup utilities.

This file validates path resolution, run directory creation, path context
construction, and serializable path formatting.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

from pathlib import Path

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.paths import (
    build_path_context,
    build_run_dir,
    build_sweep_dir,
    check_writable_dir,
    create_run_subdirs,
    prepare_run_dir,
    resolve_optional_path,
    resolve_path,
    setup_paths,
    stringify_paths,
)


#############################################
# Path resolution
#############################################


def test_resolve_path_returns_absolute_path(tmp_path: Path) -> None:
    resolved_path = resolve_path(tmp_path)

    assert resolved_path.is_absolute()


def test_resolve_optional_path_uses_default_when_none(tmp_path: Path) -> None:
    default_path = tmp_path / "default"

    resolved_path = resolve_optional_path(
        path=None,
        default=default_path,
    )

    assert resolved_path == default_path.resolve()


#############################################
# Run path construction
#############################################


def test_build_sweep_dir_returns_none_without_sweep_name(tmp_path: Path) -> None:
    sweep_dir = build_sweep_dir(
        experiment_dir=tmp_path / "experiment",
        sweep_name=None,
    )

    assert sweep_dir is None


def test_build_sweep_dir_builds_sweep_path(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"

    sweep_dir = build_sweep_dir(
        experiment_dir=experiment_dir,
        sweep_name="sweep_0",
    )

    assert sweep_dir == experiment_dir / "sweeps" / "sweep_0"


def test_build_run_dir_builds_single_run_path(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"

    run_dir = build_run_dir(
        experiment_dir=experiment_dir,
        sweep_dir=None,
        run_name="run_0",
    )

    assert run_dir == experiment_dir / "single_runs" / "run_0"


def test_build_run_dir_builds_sweep_run_path(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "experiment" / "sweeps" / "sweep_0"

    run_dir = build_run_dir(
        experiment_dir=tmp_path / "experiment",
        sweep_dir=sweep_dir,
        run_name="run_0",
    )

    assert run_dir == sweep_dir / "run_0"


#############################################
# Path context
#############################################


def test_build_path_context_contains_expected_keys(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_root = tmp_path / "results"
    experiment_dir = run_root / "experiment"
    run_dir = experiment_dir / "single_runs" / "run_0"

    path_context = build_path_context(
        project_root=project_root,
        run_root=run_root,
        experiment_dir=experiment_dir,
        sweep_dir=None,
        run_dir=run_dir,
    )

    assert set(path_context.keys()) == {
        "project_root",
        "run_root",
        "experiment_dir",
        "sweep_dir",
        "run_dir",
        "checkpoints_dir",
        "logs_dir",
        "metrics_dir",
        "configs_dir",
        "artifacts_dir",
        "is_sweep_run",
    }


#############################################
# Directory creation
#############################################


def test_prepare_run_dir_creates_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    prepare_run_dir(
        run_dir=run_dir,
        overwrite=False,
    )

    assert run_dir.is_dir()


def test_prepare_run_dir_rejects_existing_directory_without_overwrite(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileExistsError):
        prepare_run_dir(
            run_dir=run_dir,
            overwrite=False,
        )


def test_prepare_run_dir_overwrites_existing_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    old_file = run_dir / "old_file.txt"
    old_file.write_text("old content", encoding="utf-8")

    prepare_run_dir(
        run_dir=run_dir,
        overwrite=True,
    )

    assert run_dir.is_dir()
    assert not old_file.exists()


def test_create_run_subdirs_creates_expected_subdirectories(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    path_context = build_path_context(
        project_root=tmp_path,
        run_root=tmp_path / "results",
        experiment_dir=tmp_path / "results" / "experiment",
        sweep_dir=None,
        run_dir=run_dir,
    )

    create_run_subdirs(path_context)

    assert path_context["checkpoints_dir"].is_dir()
    assert path_context["logs_dir"].is_dir()
    assert path_context["metrics_dir"].is_dir()
    assert path_context["configs_dir"].is_dir()
    assert path_context["artifacts_dir"].is_dir()


def test_check_writable_dir_accepts_writable_directory(tmp_path: Path) -> None:
    check_writable_dir(tmp_path)


#############################################
# Context formatting
#############################################


def test_stringify_paths_converts_path_values_to_strings(tmp_path: Path) -> None:
    path_context = {
        "run_dir": tmp_path / "run",
        "is_sweep_run": False,
    }

    stringified_context = stringify_paths(path_context)

    assert stringified_context == {
        "run_dir": str(tmp_path / "run"),
        "is_sweep_run": False,
    }


#############################################
# Full path setup
#############################################


def test_setup_paths_creates_standard_run_structure(tmp_path: Path) -> None:
    context = setup_paths(
        paths_config={
            "idea_root": str(tmp_path),
            "run_root": str(tmp_path / "results"),
            "experiment_name": "experiment",
            "run_name": "run_0",
            "sweep_name": None,
            "overwrite": False,
        },
    )

    assert Path(context["run_dir"]).is_dir()
    assert Path(context["checkpoints_dir"]).is_dir()
    assert Path(context["logs_dir"]).is_dir()
    assert Path(context["metrics_dir"]).is_dir()
    assert Path(context["configs_dir"]).is_dir()
    assert Path(context["artifacts_dir"]).is_dir()
    assert context["is_sweep_run"] is False