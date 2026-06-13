from Octave.src.Execution import factory as execution_factory
from Octave.src.Execution.factory import (
    build_trainer,
    build_training_objects,
    build_validation_objects,
)


class DummyTrainer:
    def __init__(self, logger=None, callbacks=None, **kwargs) -> None:
        self.logger = logger
        self.callbacks = callbacks
        self.kwargs = kwargs


class DummyDataModule:
    pass


class DummyModule:
    pass


def test_build_trainer_uses_execution_level_loggers(monkeypatch) -> None:
    monkeypatch.setattr(execution_factory, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        execution_factory,
        "build_loggers",
        lambda logger_configs, runtime_context: ["logger"],
    )
    monkeypatch.setattr(
        execution_factory,
        "build_checkpoint_callbacks",
        lambda checkpoint_configs, runtime_context: ["callback"],
    )

    trainer = build_trainer(
        trainer_config={"max_epochs": 1},
        logger_configs={"csv": {}},
        checkpoint_configs={"last": {}},
        runtime_context={},
    )

    assert isinstance(trainer, DummyTrainer)
    assert trainer.logger == ["logger"]
    assert trainer.callbacks == ["callback"]
    assert trainer.kwargs == {"max_epochs": 1}


def test_build_training_objects_delegates_to_subsystem_factories(monkeypatch) -> None:
    datamodule = DummyDataModule()
    module = DummyModule()
    trainer = DummyTrainer()

    monkeypatch.setattr(
        execution_factory,
        "build_ac_video_jepa_datamodule",
        lambda config, runtime_context: datamodule,
    )
    monkeypatch.setattr(
        execution_factory,
        "build_ac_video_jepa_module",
        lambda config, runtime_context: module,
    )
    monkeypatch.setattr(
        execution_factory,
        "build_trainer",
        lambda trainer_config, logger_configs, checkpoint_configs, runtime_context: trainer,
    )
    watch_calls = []
    monkeypatch.setattr(
        execution_factory,
        "watch_module_with_wandb_loggers",
        lambda module, loggers, logger_configs: watch_calls.append(
            (module, loggers, logger_configs)
        ),
    )

    objects = build_training_objects(
        config={
            "datamodule": {},
            "module": {},
            "trainer": {},
            "loggers": {},
            "checkpoints": {},
        },
        runtime_context={},
    )

    assert objects == {
        "trainer": trainer,
        "module": module,
        "datamodule": datamodule,
    }
    assert watch_calls == [(module, trainer.logger, {})]


def test_build_validation_objects_loads_module_if_configured(monkeypatch) -> None:
    datamodule = DummyDataModule()
    module = DummyModule()
    loaded_module = DummyModule()
    trainer = DummyTrainer()

    monkeypatch.setattr(
        execution_factory,
        "build_ac_video_jepa_datamodule",
        lambda config, runtime_context: datamodule,
    )
    monkeypatch.setattr(
        execution_factory,
        "build_ac_video_jepa_module",
        lambda config, runtime_context: module,
    )
    monkeypatch.setattr(
        execution_factory,
        "build_trainer",
        lambda trainer_config, logger_configs, checkpoint_configs, runtime_context: trainer,
    )
    monkeypatch.setattr(
        execution_factory,
        "load_module_if_needed",
        lambda module, loading_config, runtime_context: loaded_module,
    )

    objects = build_validation_objects(
        config={
            "datamodule": {},
            "module": {},
            "trainer": {},
            "loggers": {},
            "checkpoints": {},
            "loading": {"module": {"enabled": True}},
        },
        runtime_context={},
    )

    assert objects == {
        "trainer": trainer,
        "module": loaded_module,
        "datamodule": datamodule,
    }
