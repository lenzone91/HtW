from pathlib import Path

import pytest
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from Octave.src.Loggers import factory as logger_factory
from Octave.src.Loggers.factory import (
    build_loggers,
    get_wandb_watch_config,
    resolve_logger_path,
    watch_module_with_wandb_loggers,
)


def test_build_loggers_returns_false_for_empty_config() -> None:
    assert build_loggers(logger_configs={}) is False


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
    )

    assert resolved_path == str((tmp_path / "run" / "logs").resolve())


def test_build_loggers_builds_csv_logger(tmp_path: Path) -> None:
    runtime_context = {
        "paths": {
            "run_dir": str(tmp_path / "run"),
        },
    }

    loggers = build_loggers(
        logger_configs={
            "csv": {
                "save_dir": "logs",
            },
        },
        runtime_context=runtime_context,
    )

    assert len(loggers) == 1
    assert isinstance(loggers[0], CSVLogger)


def test_build_loggers_builds_offline_wandb_logger(tmp_path: Path) -> None:
    runtime_context = {
        "paths": {
            "run_dir": str(tmp_path / "run"),
        },
    }

    loggers = build_loggers(
        logger_configs={
            "wandb": {
                "project": "htw-test",
                "name": "offline-test",
                "offline": True,
                "save_dir": "logs",
                "dir": "wandb",
            },
        },
        runtime_context=runtime_context,
    )

    assert len(loggers) == 1
    assert isinstance(loggers[0], WandbLogger)


def test_get_wandb_watch_config_merges_defaults() -> None:
    watch_config = get_wandb_watch_config(
        {
            "wandb": {
                "watch": {
                    "enabled": True,
                    "log_freq": 5,
                },
            },
        }
    )

    assert watch_config == {
        "enabled": True,
        "log": "all",
        "log_freq": 5,
        "log_graph": True,
    }


def test_watch_module_with_wandb_loggers_delegates_to_wandb_logger(monkeypatch) -> None:
    class FakeWandbLogger:
        def __init__(self) -> None:
            self.watch_calls = []

        def watch(self, **kwargs) -> None:
            self.watch_calls.append(kwargs)

    monkeypatch.setattr(logger_factory, "WandbLogger", FakeWandbLogger)

    module = object()
    logger = FakeWandbLogger()

    watch_module_with_wandb_loggers(
        module=module,
        loggers=[logger],
        logger_configs={
            "wandb": {
                "watch": {
                    "enabled": True,
                    "log": "all",
                    "log_freq": 7,
                    "log_graph": False,
                },
            },
        },
    )

    assert logger.watch_calls == [
        {
            "model": module,
            "log": "all",
            "log_freq": 7,
            "log_graph": False,
        }
    ]


def test_build_loggers_rejects_unknown_logger() -> None:
    with pytest.raises(KeyError, match="Unknown logger"):
        build_loggers(logger_configs={"unknown": {}})


def test_build_loggers_rejects_unknown_wandb_watch_key() -> None:
    with pytest.raises(KeyError, match="Unknown wandb watch config keys"):
        build_loggers(
            logger_configs={
                "wandb": {
                    "watch": {
                        "unknown": 1,
                    },
                },
            }
        )
