from pathlib import Path

import pytest

from Octave.src.Setup.paths import (
    resolve_existing_run_dir_policy,
    setup_paths,
)


def test_setup_paths_resolves_and_creates_run_directories(tmp_path: Path) -> None:
    path_context = setup_paths(
        paths_config={
            "project_root": str(tmp_path),
            "run_root": "runs",
            "experiment_name": "experiment",
            "run_name": "run_0",
            "existing_run_dir_policy": "fail",
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
        "existing_run_dir_policy": "fail",
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
        "existing_run_dir_policy": "overwrite",
    }

    first_context = setup_paths(config)
    marker = Path(first_context["run_dir"]) / "marker.txt"
    marker.write_text("old", encoding="utf-8")

    second_context = setup_paths(config)

    assert Path(second_context["run_dir"]).exists()
    assert not marker.exists()


def test_setup_paths_keeps_legacy_overwrite_alias(tmp_path: Path) -> None:
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


def test_setup_paths_ask_policy_fails_when_non_interactive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class NonInteractiveStdin:
        @staticmethod
        def isatty() -> bool:
            return False

    config = {
        "project_root": str(tmp_path),
        "run_root": "runs",
        "experiment_name": "experiment",
        "run_name": "run_0",
        "existing_run_dir_policy": "ask",
    }
    setup_paths({**config, "existing_run_dir_policy": "fail"})
    monkeypatch.setattr("Octave.src.Setup.paths.sys.stdin", NonInteractiveStdin())

    with pytest.raises(FileExistsError, match="cannot prompt"):
        setup_paths(config)


def test_setup_paths_ask_policy_overwrites_when_confirmed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class InteractiveStdin:
        @staticmethod
        def isatty() -> bool:
            return True

    config = {
        "project_root": str(tmp_path),
        "run_root": "runs",
        "experiment_name": "experiment",
        "run_name": "run_0",
        "existing_run_dir_policy": "ask",
    }

    first_context = setup_paths({**config, "existing_run_dir_policy": "fail"})
    marker = Path(first_context["run_dir"]) / "marker.txt"
    marker.write_text("old", encoding="utf-8")

    monkeypatch.setattr("Octave.src.Setup.paths.sys.stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")

    second_context = setup_paths(config)

    assert Path(second_context["run_dir"]).exists()
    assert not marker.exists()


def test_resolve_existing_run_dir_policy_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Invalid existing_run_dir_policy"):
        resolve_existing_run_dir_policy({"existing_run_dir_policy": "unknown"})
