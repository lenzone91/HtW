from pathlib import Path

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from Octave.src.Training.Checkpoints.configs import DEFAULT_LAST_CHECKPOINT_CONFIG
from Octave.src.Training.Checkpoints.checkpoints import (
    get_default_checkpoint_dir,
    resolve_checkpoint_dirpath,
)
from Octave.src.Training.Checkpoints.factory import (
    build_checkpoint_callbacks,
)


def make_runtime_context(tmp_path: Path) -> dict:
    return {
        "paths": {
            "run_dir": str(tmp_path / "run"),
            "checkpoints_dir": str(tmp_path / "run" / "checkpoints"),
        },
    }


def test_get_default_checkpoint_dir_returns_context_checkpoint_dir(tmp_path: Path) -> None:
    runtime_context = make_runtime_context(tmp_path)

    assert get_default_checkpoint_dir(runtime_context) == str(
        tmp_path / "run" / "checkpoints"
    )


def test_resolve_checkpoint_dirpath_uses_runtime_checkpoint_dir(tmp_path: Path) -> None:
    runtime_context = make_runtime_context(tmp_path)

    dirpath = resolve_checkpoint_dirpath(
        config={"dirpath": None},
        runtime_context=runtime_context,
    )

    assert dirpath == str(tmp_path / "run" / "checkpoints")


def test_resolve_checkpoint_dirpath_resolves_relative_dirpath(tmp_path: Path) -> None:
    runtime_context = make_runtime_context(tmp_path)

    dirpath = resolve_checkpoint_dirpath(
        config={"dirpath": "custom"},
        runtime_context=runtime_context,
    )

    assert dirpath == str((tmp_path / "run" / "custom").resolve())


def test_build_checkpoint_callbacks_builds_last_callback(tmp_path: Path) -> None:
    runtime_context = make_runtime_context(tmp_path)

    callbacks = build_checkpoint_callbacks(
        checkpoint_configs={
            "last": DEFAULT_LAST_CHECKPOINT_CONFIG,
        },
        runtime_context=runtime_context,
    )

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], ModelCheckpoint)
    assert callbacks[0].save_last is True


def test_build_checkpoint_callbacks_returns_empty_list_for_empty_config() -> None:
    assert build_checkpoint_callbacks({}) == []


def test_build_checkpoint_callbacks_rejects_unknown_checkpoint_type() -> None:
    with pytest.raises(RuntimeError, match="Unknown checkpoint"):
        build_checkpoint_callbacks(
            {
                "bad": {
                    "checkpoint_type": "unknown",
                },
            }
        )
