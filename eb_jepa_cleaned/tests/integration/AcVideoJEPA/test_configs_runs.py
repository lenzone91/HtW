"""
User run configs in the project-root Configs/ folder: a thin run config there
reuses the framework groups (pkg:// search path) and launches end-to-end, and
re-running an existing config triggers the existing-results "ask" policy.
"""

from pathlib import Path

import pytest

from src.AIML.Execution.launch import launch

CONFIGS_DIR = str(Path(__file__).resolve().parents[3] / "Configs")


def test_launch_user_config_from_configs_folder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    objects = launch(
        CONFIGS_DIR,
        config_name="toy_run",
        existing_run_dir_policy="overwrite",
    )
    assert objects["trainer"].state.finished
    # run_name from the config keys the results directory
    assert (tmp_path / "runs" / "ac_video_jepa" / "toy_run").is_dir()


def test_rerun_existing_results_triggers_ask(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # First run creates the results.
    launch(CONFIGS_DIR, config_name="toy_run", existing_run_dir_policy="overwrite")
    # Re-running the same config uses the config's policy (ask); non-interactive
    # -> it aborts rather than silently overwrite (the user is asked otherwise).
    with pytest.raises(FileExistsError):
        launch(CONFIGS_DIR, config_name="toy_run")
