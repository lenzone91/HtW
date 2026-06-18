"""
User run configs in project-root Configs/ as Project_2-style fragment folders:
`Configs/<run>/` (fragments + config.yaml entry, composed by Hydra) and the
merged `Configs/<run>.yaml` snapshot. Launching works from either form, and
re-running an existing config triggers the existing-results "ask" policy.

`monkeypatch.chdir(tmp_path)` makes the config's relative `run_root` ("runs")
resolve under a temp dir, so run directories don't pollute the repo.
"""

from pathlib import Path

import pytest

from src.AIML.Execution.launch import launch
from src.Workflow.Configs.run_config import resolve_run_config, save_composed_run

CONFIGS = Path(__file__).resolve().parents[3] / "Configs"
RUN_NAME = "toy_run_offline"
RUN_DIR = CONFIGS / RUN_NAME
RUN_SNAPSHOT = CONFIGS / f"{RUN_NAME}.yaml"

# Keep the launch tests fast: the offline kitchen-sink run trains 10 epochs by
# default; one is enough to exercise compose -> build -> fit dispatch.
QUICK = ["trainer.max_epochs=1"]


def test_resolve_folder_and_snapshot_agree():
    folder = resolve_run_config(str(RUN_DIR))
    snapshot = resolve_run_config(str(RUN_SNAPSHOT))
    assert folder["setup"]["paths"]["run_name"] == RUN_NAME
    assert set(folder) == set(snapshot)
    assert folder["module"].keys() == snapshot["module"].keys()


def test_save_composed_run_writes_sibling_snapshot(tmp_path):
    out = save_composed_run(str(RUN_DIR), output_path=tmp_path / f"{RUN_NAME}.yaml")
    assert out.is_file()
    assert resolve_run_config(str(out))["setup"]["paths"]["run_name"] == RUN_NAME


def test_launch_from_run_folder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    objects = launch(str(RUN_DIR), overrides=QUICK, existing_run_dir_policy="overwrite")
    assert objects["trainer"].state.finished
    assert (tmp_path / "runs" / "ac_video_jepa" / RUN_NAME).is_dir()


def test_launch_from_composed_snapshot(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    objects = launch(str(RUN_SNAPSHOT), overrides=QUICK, existing_run_dir_policy="overwrite")
    assert objects["trainer"].state.finished


def test_rerun_existing_results_triggers_ask(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    launch(str(RUN_DIR), overrides=QUICK, existing_run_dir_policy="overwrite")  # create results
    # Re-running uses the config's policy (ask); non-interactive -> aborts rather
    # than silently overwrite (the user is prompted otherwise).
    with pytest.raises(FileExistsError):
        launch(str(RUN_DIR), overrides=QUICK)
