"""
Tests for checkpoint callback factory utilities.

This file validates checkpoint path resolution and checkpoint callback
construction.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

from copy import deepcopy
from pathlib import Path

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Checkpoints.configs import (
    DEFAULT_CHECKPOINT_CONFIGS,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Checkpoints.factory import (
    build_checkpoint_callbacks,
    get_default_checkpoint_dir,
    get_path_root,
    resolve_checkpoint_dirpath,
)


#############################################
# Helpers
#############################################


def make_runtime_context(tmp_path: Path) -> dict:
    return {
        "paths": {
            "run_dir": str(tmp_path / "run"),
            "checkpoints_dir": str(tmp_path / "run" / "checkpoints"),
        },
    }


#############################################
# Default checkpoint directory
#############################################


def test_get_default_checkpoint_dir_returns_none_without_runtime_context() -> None:
    assert get_default_checkpoint_dir(runtime_context=None) is None


def test_get_default_checkpoint_dir_returns_context_checkpoint_dir(
    tmp_path: Path,
) -> None:
    runtime_context = make_runtime_context(tmp_path)

    checkpoint_dir = get_default_checkpoint_dir(
        runtime_context=runtime_context,
    )

    assert checkpoint_dir == str(tmp_path / "run" / "checkpoints")


def test_get_default_checkpoint_dir_rejects_missing_checkpoint_dir() -> None:
    runtime_context = {
        "paths": {},
    }

    with pytest.raises(KeyError):
        get_default_checkpoint_dir(runtime_context=runtime_context)


#############################################
# Runtime path roots
#############################################


def test_get_path_root_returns_runtime_context_path(tmp_path: Path) -> None:
    runtime_context = make_runtime_context(tmp_path)

    root = get_path_root(
        runtime_context=runtime_context,
        root_key="run_dir",
    )

    assert root == (tmp_path / "run").resolve()


def test_get_path_root_rejects_missing_runtime_context() -> None:
    with pytest.raises(ValueError):
        get_path_root(
            runtime_context=None,
            root_key="run_dir",
        )


#############################################
# Checkpoint dirpath resolution
#############################################


def test_resolve_checkpoint_dirpath_uses_default_checkpoint_dir(
    tmp_path: Path,
) -> None:
    runtime_context = make_runtime_context(tmp_path)

    resolved_config = resolve_checkpoint_dirpath(
        config={"dirpath": None, "filename": "last"},
        runtime_context=runtime_context,
    )

    assert resolved_config == {
        "dirpath": str(tmp_path / "run" / "checkpoints"),
        "filename": "last",
    }


def test_resolve_checkpoint_dirpath_resolves_relative_dirpath(
    tmp_path: Path,
) -> None:
    runtime_context = make_runtime_context(tmp_path)

    resolved_config = resolve_checkpoint_dirpath(
        config={"dirpath": "custom_checkpoints", "filename": "last"},
        runtime_context=runtime_context,
    )

    assert resolved_config == {
        "dirpath": str((tmp_path / "run" / "custom_checkpoints").resolve()),
        "filename": "last",
    }


#############################################
# Callback construction
#############################################


def test_build_checkpoint_callbacks_builds_default_callbacks(
    tmp_path: Path,
) -> None:
    runtime_context = make_runtime_context(tmp_path)

    callbacks = build_checkpoint_callbacks(
        checkpoint_configs=deepcopy(DEFAULT_CHECKPOINT_CONFIGS),
        runtime_context=runtime_context,
    )

    assert len(callbacks) == 2
    assert all(isinstance(callback, ModelCheckpoint) for callback in callbacks)


def test_build_checkpoint_callbacks_rejects_unknown_checkpoint_type(
    tmp_path: Path,
) -> None:
    runtime_context = make_runtime_context(tmp_path)

    with pytest.raises(RuntimeError):
        build_checkpoint_callbacks(
            checkpoint_configs={
                "unknown_checkpoint": {
                    "checkpoint_type": "unknown_type",
                    "dirpath": None,
                    "filename": "checkpoint",
                },
            },
            runtime_context=runtime_context,
        )


def test_build_checkpoint_callbacks_rejects_missing_checkpoint_type(
    tmp_path: Path,
) -> None:
    runtime_context = make_runtime_context(tmp_path)

    with pytest.raises(RuntimeError):
        build_checkpoint_callbacks(
            checkpoint_configs={
                "missing_type_checkpoint": {
                    "dirpath": None,
                    "filename": "checkpoint",
                },
            },
            runtime_context=runtime_context,
        )