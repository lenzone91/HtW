"""
Tests for logger factory utilities.

This file validates logger path resolution and named logger construction.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

from pathlib import Path

import pytest
from lightning.pytorch.loggers import CSVLogger

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Loggers.factory import (
    build_loggers,
    resolve_logger_path,
    resolve_logger_paths,
)


#############################################
# Logger path resolution
#############################################


def test_resolve_logger_path_keeps_absolute_path(tmp_path: Path) -> None:
    absolute_path = tmp_path / "logs"

    resolved_path = resolve_logger_path(
        path=str(absolute_path),
        runtime_context=None,
    )

    assert resolved_path == str(absolute_path.resolve())


def test_resolve_logger_path_keeps_relative_path_without_runtime_context() -> None:
    resolved_path = resolve_logger_path(
        path="logs",
        runtime_context=None,
    )

    assert resolved_path == "logs"


def test_resolve_logger_path_resolves_relative_path_from_runtime_context(
    tmp_path: Path,
) -> None:
    runtime_context = {
        "paths": {
            "run_dir": str(tmp_path / "run"),
        },
    }

    resolved_path = resolve_logger_path(
        path="logs",
        runtime_context=runtime_context,
        relative_to="run_dir",
    )

    assert resolved_path == str((tmp_path / "run" / "logs").resolve())


def test_resolve_logger_path_rejects_unknown_runtime_context_root(
    tmp_path: Path,
) -> None:
    runtime_context = {
        "paths": {
            "run_dir": str(tmp_path / "run"),
        },
    }

    with pytest.raises(KeyError):
        resolve_logger_path(
            path="logs",
            runtime_context=runtime_context,
            relative_to="missing_root",
        )


def test_resolve_logger_paths_resolves_save_dir_and_wandb_dir(
    tmp_path: Path,
) -> None:
    runtime_context = {
        "paths": {
            "run_dir": str(tmp_path / "run"),
        },
    }

    config = {
        "save_dir": "logs",
        "dir": "wandb",
        "name": "dummy_logger",
    }

    resolved_config = resolve_logger_paths(
        config=config,
        runtime_context=runtime_context,
    )

    assert resolved_config == {
        "save_dir": str((tmp_path / "run" / "logs").resolve()),
        "dir": str((tmp_path / "run" / "wandb").resolve()),
        "name": "dummy_logger",
    }


#############################################
# Logger construction
#############################################


def test_build_loggers_returns_false_for_empty_config() -> None:
    loggers = build_loggers(
        logger_configs={},
        runtime_context=None,
    )

    assert loggers is False


def test_build_loggers_builds_named_csv_logger(tmp_path: Path) -> None:
    runtime_context = {
        "paths": {
            "run_dir": str(tmp_path / "run"),
        },
    }

    loggers = build_loggers(
        logger_configs={
            "csv": {
                "save_dir": "logs",
                "name": "csv_reports",
                "version": None,
                "prefix": "",
                "flush_logs_every_n_steps": 100,
            },
        },
        runtime_context=runtime_context,
    )

    assert isinstance(loggers, list)
    assert len(loggers) == 1
    assert isinstance(loggers[0], CSVLogger)


def test_build_loggers_rejects_unknown_logger_name() -> None:
    with pytest.raises(RuntimeError):
        build_loggers(
            logger_configs={
                "unknown_logger": {},
            },
            runtime_context=None,
        )