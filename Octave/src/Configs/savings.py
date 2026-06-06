from pathlib import Path

from .conversion import save_config


#############################################
# Config saving
#############################################

# This file provides low-level saving utilities for reproducibility snapshots.
#
# It does not:
#   - decide when snapshots are saved;
#   - decide whether training/evaluation should continue;
#   - build runtime paths;
#   - modify configs;
#   - build objects.
#
# Execution/ decides when and where these functions are called.


def save_config_snapshot(
    config: dict,
    snapshot_dir: str | Path,
    filename: str = "config.yaml",
) -> None:
    """
    Save the final config used for an execution.
    """
    save_config(
        config=config,
        path=Path(snapshot_dir) / filename,
    )


def save_runtime_context_snapshot(
    runtime_context: dict,
    snapshot_dir: str | Path,
    filename: str = "runtime_context.yaml",
) -> None:
    """
    Save the runtime context used for an execution.
    """
    save_config(
        config=runtime_context,
        path=Path(snapshot_dir) / filename,
    )


def save_execution_snapshots(
    config: dict,
    runtime_context: dict,
    snapshot_dir: str | Path,
    config_filename: str = "config.yaml",
    runtime_context_filename: str = "runtime_context.yaml",
) -> None:
    """
    Save all reproducibility snapshots for an execution.
    """
    save_config_snapshot(
        config=config,
        snapshot_dir=snapshot_dir,
        filename=config_filename,
    )

    save_runtime_context_snapshot(
        runtime_context=runtime_context,
        snapshot_dir=snapshot_dir,
        filename=runtime_context_filename,
    )