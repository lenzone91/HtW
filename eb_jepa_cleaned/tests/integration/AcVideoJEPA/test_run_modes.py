"""
Run modes via AIML/Execution.launch on the AcVideoJEPA Hydra config:
train, validate, resume, mode-from-config, and unknown-mode rejection.

`monkeypatch.chdir(tmp_path)` makes the config's relative `run_root` ("runs")
resolve under a temp dir, so run directories don't pollute the repo.
"""

from pathlib import Path

import pytest

import src.AcVideoJEPA as acvideo_jepa
from src.AIML.Execution.launch import launch

CONFIG_DIR = str(Path(acvideo_jepa.__file__).parent / "configs")


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "ac_video_jepa" / "default_run"


def test_launch_train_writes_run_and_snapshots(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    objects = launch(CONFIG_DIR, mode="train", existing_run_dir_policy="overwrite")

    assert objects["trainer"].state.finished
    configs_dir = _run_dir(tmp_path) / "configs"
    assert (configs_dir / "config.yaml").is_file()
    assert (configs_dir / "runtime_context.yaml").is_file()


def test_launch_mode_defaults_from_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # No --mode: falls back to config["run"]["mode"] == "train".
    objects = launch(CONFIG_DIR, existing_run_dir_policy="overwrite")
    assert objects["trainer"].state.finished


def test_launch_validate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    objects = launch(CONFIG_DIR, mode="validate", existing_run_dir_policy="overwrite")
    assert set(objects) >= {"trainer", "module", "datamodule"}


def test_launch_resume_from_checkpoint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    trained = launch(CONFIG_DIR, mode="train", existing_run_dir_policy="overwrite")
    checkpoint = tmp_path / "step.ckpt"
    trained["trainer"].save_checkpoint(str(checkpoint))

    resumed = launch(
        CONFIG_DIR,
        mode="resume",
        existing_run_dir_policy="overwrite",
        checkpoint_path=str(checkpoint),
        overrides=["trainer.max_steps=2"],
    )
    assert resumed["trainer"].state.finished


def test_launch_resume_without_checkpoint_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        launch(CONFIG_DIR, mode="resume", existing_run_dir_policy="overwrite")


def test_launch_rejects_unknown_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        launch(CONFIG_DIR, mode="bogus", existing_run_dir_policy="overwrite")
