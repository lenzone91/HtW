from pathlib import Path

import pytest

from Octave.src.Setup.paths import setup_paths


def test_setup_paths_resolves_and_creates_run_directories(tmp_path: Path) -> None:
    path_context = setup_paths(
        paths_config={
            "project_root": str(tmp_path),
            "run_root": "runs",
            "experiment_name": "experiment",
            "run_name": "run_0",
            "overwrite": False,
        }
    )

    assert Path(path_context["run_dir"]).exists()
    assert Path(path_context["logs_dir"]).exists()
    assert Path(path_context["checkpoints_dir"]).exists()


def test_setup_paths_rejects_existing_run_without_overwrite(tmp_path: Path) -> None:
    config = {
        "project_root": str(tmp_path),
        "run_root": "runs",
        "experiment_name": "experiment",
        "run_name": "run_0",
        "overwrite": False,
    }
    setup_paths(config)

    with pytest.raises(FileExistsError):
        setup_paths(config)


def test_setup_paths_overwrites_existing_run_when_enabled(tmp_path: Path) -> None:
    config = {
        "project_root": str(tmp_path),
        "run_root": "runs",
        "experiment_name": "experiment",
        "run_name": "run_0",
        "overwrite": True,
    }

    first_context = setup_paths(config)
    marker = Path(first_context["run_dir"]) / "marker.txt"
    marker.write_text("old", encoding="utf-8")

    second_context = setup_paths(config)

    assert Path(second_context["run_dir"]).exists()
    assert not marker.exists()
