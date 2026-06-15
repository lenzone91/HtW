from pathlib import Path

import pytest
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from Octave.src.Loggers import factory as logger_factory
from Octave.src.Loggers import wandb_metrics
from Octave.src.Loggers.factory import (
    build_logger_callbacks,
    build_loggers,
    get_wandb_metrics_config,
    get_wandb_watch_config,
    watch_module_with_wandb_loggers,
)
from Octave.src.Loggers.loggers import resolve_logger_path
from Octave.src.Loggers.wandb_metrics import (
    WandbScalarMetricsCallback,
    collect_prefixed_scalar_metrics,
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


def test_get_wandb_metrics_config_merges_defaults() -> None:
    metrics_config = get_wandb_metrics_config(
        {
            "wandb": {
                "metrics": {
                    "enabled": True,
                    "direct_log": False,
                    "metric_patterns": ["train/*"],
                },
            },
        }
    )

    assert metrics_config == {
        "enabled": True,
        "define_metrics": True,
        "direct_log": False,
        "step_metric": "trainer/global_step",
        "metric_patterns": ["train/*"],
        "require_wandb_logger": True,
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


def test_build_logger_callbacks_builds_wandb_metrics_callback() -> None:
    callbacks = build_logger_callbacks(
        {
            "wandb": {
                "metrics": {
                    "enabled": True,
                    "direct_log": True,
                },
            },
        }
    )

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], WandbScalarMetricsCallback)


def test_wandb_metrics_callback_defines_metric_patterns(monkeypatch) -> None:
    class FakeWandbLogger:
        def __init__(self) -> None:
            self.experiment = FakeExperiment()

    class FakeExperiment:
        def __init__(self) -> None:
            self.defined_metrics = []

        def define_metric(self, *args, **kwargs) -> None:
            self.defined_metrics.append((args, kwargs))

    class FakeTrainer:
        def __init__(self) -> None:
            self.loggers = [FakeWandbLogger()]

    monkeypatch.setattr(wandb_metrics, "WandbLogger", FakeWandbLogger)

    callback = WandbScalarMetricsCallback(
        define_metrics=True,
        direct_log=False,
        metric_patterns=["train/*"],
    )
    trainer = FakeTrainer()

    callback.on_fit_start(trainer=trainer, pl_module=None)

    assert trainer.loggers[0].experiment.defined_metrics == [
        (("trainer/global_step",), {}),
        (("train/*",), {"step_metric": "trainer/global_step"}),
    ]


def test_wandb_metrics_callback_logs_scalar_metrics(monkeypatch) -> None:
    class FakeWandbLogger:
        def __init__(self) -> None:
            self.experiment = FakeExperiment()

    class FakeExperiment:
        def __init__(self) -> None:
            self.logged_metrics = []

        def log(self, metrics: dict) -> None:
            self.logged_metrics.append(metrics)

    class FakeTrainer:
        def __init__(self) -> None:
            self.global_step = 3
            self.loggers = [FakeWandbLogger()]
            self.logged_metrics = {
                "train/loss": torch.tensor(1.5),
                "train/vector": torch.tensor([1.0, 2.0]),
                "val/loss": torch.tensor(2.5),
            }
            self.callback_metrics = {
                "train/pred_loss": 0.25,
            }

    monkeypatch.setattr(wandb_metrics, "WandbLogger", FakeWandbLogger)

    callback = WandbScalarMetricsCallback(
        define_metrics=False,
        direct_log=True,
    )
    trainer = FakeTrainer()

    callback.log_scalar_metrics(trainer=trainer, prefixes=("train/",))

    assert trainer.loggers[0].experiment.logged_metrics == [
        {
            "train/loss": 1.5,
            "train/pred_loss": 0.25,
            "trainer/global_step": 3,
        }
    ]


def test_collect_prefixed_scalar_metrics_ignores_non_scalars() -> None:
    class FakeTrainer:
        logged_metrics = {
            "train/loss": torch.tensor(1.0),
            "train/vector": torch.tensor([1.0, 2.0]),
            "other/loss": torch.tensor(3.0),
        }
        callback_metrics = {}

    metrics = collect_prefixed_scalar_metrics(
        trainer=FakeTrainer(),
        prefixes=("train/",),
    )

    assert metrics == {"train/loss": 1.0}


def test_build_loggers_rejects_unknown_logger() -> None:
    with pytest.raises(RuntimeError, match="Unknown logger"):
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


def test_build_loggers_rejects_unknown_wandb_metrics_key() -> None:
    with pytest.raises(KeyError, match="Unknown wandb metrics config keys"):
        build_loggers(
            logger_configs={
                "wandb": {
                    "metrics": {
                        "unknown": 1,
                    },
                },
            }
        )
