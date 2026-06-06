"""
Tests for execution factory utilities.

This file validates Execution-level object assembly.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.

The monkeypatch fixture is used to temporarily replace subsystem builders and
Lightning Trainer construction. This isolates Execution/ orchestration from
DataModules/, Modules/, Loggers/, Checkpoints/, EarlyStoppings/, Loading/, and
Lightning internals.
"""

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Execution import (
    factory as execution_factory,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Execution.factory import (
    build_evaluation_objects,
    build_evaluation_trainer,
    build_trainer,
    build_training_callbacks,
    build_training_objects,
    build_training_trainer,
)


#############################################
# Dummy objects
#############################################


class DummyTrainer:
    def __init__(self, logger=None, callbacks=None, **trainer_config) -> None:
        self.logger = logger
        self.callbacks = callbacks
        self.trainer_config = trainer_config


class DummyDataModule:
    pass


class DummyModule:
    pass


class LoadedDummyModule:
    pass


#############################################
# Callback construction
#############################################


def test_build_training_callbacks_gathers_checkpoint_and_early_stopping_callbacks(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        execution_factory,
        "build_checkpoint_callbacks",
        lambda checkpoint_configs, runtime_context: ["checkpoint_callback"],
    )
    monkeypatch.setattr(
        execution_factory,
        "build_early_stopping_callbacks",
        lambda early_stopping_configs, runtime_context: ["early_stopping_callback"],
    )

    callbacks = build_training_callbacks(
        config={
            "checkpoints": {},
            "early_stoppings": {},
        },
        runtime_context={},
    )

    assert callbacks == ["checkpoint_callback", "early_stopping_callback"]


#############################################
# Trainer construction
#############################################


def test_build_trainer_forwards_logger_callbacks_and_config(monkeypatch) -> None:
    monkeypatch.setattr(execution_factory, "Trainer", DummyTrainer)

    trainer = build_trainer(
        trainer_config={"max_epochs": 3},
        loggers=["logger"],
        callbacks=["callback"],
    )

    assert isinstance(trainer, DummyTrainer)
    assert trainer.logger == ["logger"]
    assert trainer.callbacks == ["callback"]
    assert trainer.trainer_config == {"max_epochs": 3}


def test_build_training_trainer_builds_loggers_callbacks_and_trainer(monkeypatch) -> None:
    monkeypatch.setattr(execution_factory, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        execution_factory,
        "build_loggers",
        lambda logger_configs, runtime_context: ["logger"],
    )
    monkeypatch.setattr(
        execution_factory,
        "build_training_callbacks",
        lambda config, runtime_context: ["callback"],
    )

    trainer = build_training_trainer(
        config={
            "loggers": {},
            "checkpoints": {},
            "early_stoppings": {},
            "trainer": {"max_epochs": 2},
        },
        runtime_context={},
    )

    assert isinstance(trainer, DummyTrainer)
    assert trainer.logger == ["logger"]
    assert trainer.callbacks == ["callback"]
    assert trainer.trainer_config == {"max_epochs": 2}


def test_build_evaluation_trainer_builds_logger_without_callbacks(monkeypatch) -> None:
    monkeypatch.setattr(execution_factory, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        execution_factory,
        "build_loggers",
        lambda logger_configs, runtime_context: ["logger"],
    )

    trainer = build_evaluation_trainer(
        config={
            "loggers": {},
            "trainer": {"enable_checkpointing": False},
        },
        runtime_context={},
    )

    assert isinstance(trainer, DummyTrainer)
    assert trainer.logger == ["logger"]
    assert trainer.callbacks == []
    assert trainer.trainer_config == {"enable_checkpointing": False}


#############################################
# Execution object bundles
#############################################


def test_build_training_objects_builds_execution_bundle(monkeypatch) -> None:
    datamodule = DummyDataModule()
    module = DummyModule()
    loaded_module = LoadedDummyModule()
    trainer = DummyTrainer()

    monkeypatch.setattr(
        execution_factory,
        "build_datamodule",
        lambda datamodule_configs, runtime_context: datamodule,
    )
    monkeypatch.setattr(
        execution_factory,
        "build_lightning_module",
        lambda lightning_module_configs, runtime_context, loading_config=None: module,
    )
    monkeypatch.setattr(
        execution_factory,
        "load_module_if_needed",
        lambda module, loading_config, runtime_context: loaded_module,
    )
    monkeypatch.setattr(
        execution_factory,
        "build_training_trainer",
        lambda config, runtime_context: trainer,
    )

    objects = build_training_objects(
        config={
            "datamodule": {},
            "module": {},
            "loading": {"module": {}},
        },
        runtime_context={},
    )

    assert objects == {
        "trainer": trainer,
        "module": loaded_module,
        "datamodule": datamodule,
    }


def test_build_evaluation_objects_builds_execution_bundle(monkeypatch) -> None:
    datamodule = DummyDataModule()
    module = DummyModule()
    loaded_module = LoadedDummyModule()
    trainer = DummyTrainer()

    monkeypatch.setattr(
        execution_factory,
        "build_datamodule",
        lambda datamodule_configs, runtime_context: datamodule,
    )
    monkeypatch.setattr(
        execution_factory,
        "build_lightning_module",
        lambda lightning_module_configs, runtime_context, loading_config=None: module,
    )
    monkeypatch.setattr(
        execution_factory,
        "load_module_if_needed",
        lambda module, loading_config, runtime_context: loaded_module,
    )
    monkeypatch.setattr(
        execution_factory,
        "build_evaluation_trainer",
        lambda config, runtime_context: trainer,
    )

    objects = build_evaluation_objects(
        config={
            "datamodule": {},
            "module": {},
            "loading": {"module": {}},
        },
        runtime_context={},
    )

    assert objects == {
        "trainer": trainer,
        "module": loaded_module,
        "datamodule": datamodule,
    }