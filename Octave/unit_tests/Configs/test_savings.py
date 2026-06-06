"""
Pytest test module for Configs.savings.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

from pathlib import Path

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.conversion import load_config
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.savings import (
    save_config_snapshot,
    save_execution_snapshots,
    save_runtime_context_snapshot,
)


#############################################
# Test data
#############################################

CONFIG = {
    "setup": {
        "seed": 123,
    },
}

RUNTIME_CONTEXT = {
    "paths": {
        "run_dir": "runs/toy",
    },
}


#############################################
# Snapshot saving tests
#############################################

def test_save_config_snapshot(tmp_path: Path) -> None:
    """
    Test that the execution config snapshot is written and reloadable.
    """
    save_config_snapshot(
        config=CONFIG,
        snapshot_dir=tmp_path,
    )

    saved_config = load_config(tmp_path / "config.yaml")

    assert saved_config == CONFIG


def test_save_runtime_context_snapshot(tmp_path: Path) -> None:
    """
    Test that the runtime context snapshot is written and reloadable.
    """
    save_runtime_context_snapshot(
        runtime_context=RUNTIME_CONTEXT,
        snapshot_dir=tmp_path,
    )

    saved_runtime_context = load_config(tmp_path / "runtime_context.yaml")

    assert saved_runtime_context == RUNTIME_CONTEXT


def test_save_execution_snapshots(tmp_path: Path) -> None:
    """
    Test that both execution snapshots are written and reloadable.
    """
    save_execution_snapshots(
        config=CONFIG,
        runtime_context=RUNTIME_CONTEXT,
        snapshot_dir=tmp_path,
    )

    saved_config = load_config(tmp_path / "config.yaml")
    saved_runtime_context = load_config(tmp_path / "runtime_context.yaml")

    assert saved_config == CONFIG
    assert saved_runtime_context == RUNTIME_CONTEXT


def test_save_execution_snapshots_with_string_snapshot_dir(tmp_path: Path) -> None:
    """
    Test that snapshot_dir can be provided as a string path.
    """
    snapshot_dir = str(tmp_path)

    save_execution_snapshots(
        config=CONFIG,
        runtime_context=RUNTIME_CONTEXT,
        snapshot_dir=snapshot_dir,
    )

    saved_config = load_config(tmp_path / "config.yaml")
    saved_runtime_context = load_config(tmp_path / "runtime_context.yaml")

    assert saved_config == CONFIG
    assert saved_runtime_context == RUNTIME_CONTEXT