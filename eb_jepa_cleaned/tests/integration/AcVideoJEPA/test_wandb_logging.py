"""
wandb logging wired end-to-end: a launch with `loggers=wandb` (offline) trains
one step and writes the wandb run under the resolved run logs directory.
"""

from pathlib import Path

import src.AcVideoJEPA as acvideo_jepa
from src.AIML.Execution.launch import launch

CONFIG_DIR = str(Path(acvideo_jepa.__file__).parent / "configs")


def test_launch_train_with_wandb_offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    objects = launch(
        CONFIG_DIR,
        mode="train",
        existing_run_dir_policy="overwrite",
        overrides=["loggers=wandb"],
    )

    assert objects["trainer"].state.finished
    logs_dir = tmp_path / "runs" / "ac_video_jepa" / "default_run" / "logs"
    # The (offline) wandb logger writes under the resolved run logs directory.
    assert (logs_dir / "wandb").is_dir()
