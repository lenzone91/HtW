import pytest

from Octave.src.Execution import train as execution_train
from Octave.src.Execution import validate as execution_validate
from Octave.src.Execution.train import run_training
from Octave.src.Execution.validate import run_validation


class DummyTrainer:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.fit_called = False
        self.validate_called = False

    def fit(self, model, datamodule, ckpt_path=None) -> None:
        self.fit_called = True
        self.ckpt_path = ckpt_path
        if self.should_fail:
            raise RuntimeError("fit failed")

    def validate(self, model, datamodule):
        self.validate_called = True
        return [{"val/loss": 1.0}]


class DummyModule:
    pass


class DummyDataModule:
    pass


def make_runtime_context() -> dict:
    return {
        "paths": {
            "run_dir": "dummy_run_dir",
        },
    }


def make_config() -> dict:
    return {
        "runtime_context": make_runtime_context(),
        "datamodule": {},
        "module": {},
        "trainer": {},
        "loggers": {},
        "resume": {"enabled": False, "checkpoint_path": None},
    }


def test_run_training_calls_fit_and_marks_finished(monkeypatch) -> None:
    trainer = DummyTrainer()

    monkeypatch.setattr(
        execution_train,
        "build_training_objects",
        lambda config, runtime_context: {
            "trainer": trainer,
            "module": DummyModule(),
            "datamodule": DummyDataModule(),
        },
    )
    monkeypatch.setattr(
        execution_train,
        "save_execution_report",
        lambda execution_report, runtime_context, filename: None,
    )
    monkeypatch.setattr(execution_train, "close_external_services", lambda: None)

    report = run_training(make_config())

    assert report["status"] == "finished"
    assert trainer.fit_called is True
    assert trainer.ckpt_path is None


def test_run_training_passes_resume_checkpoint_path(monkeypatch) -> None:
    trainer = DummyTrainer()
    config = make_config()
    config["resume"] = {
        "enabled": True,
        "checkpoint_path": "checkpoint.ckpt",
    }

    monkeypatch.setattr(
        execution_train,
        "build_training_objects",
        lambda config, runtime_context: {
            "trainer": trainer,
            "module": DummyModule(),
            "datamodule": DummyDataModule(),
        },
    )
    monkeypatch.setattr(
        execution_train,
        "save_execution_report",
        lambda execution_report, runtime_context, filename: None,
    )
    monkeypatch.setattr(execution_train, "close_external_services", lambda: None)

    run_training(config)

    assert trainer.ckpt_path == "checkpoint.ckpt"


def test_run_training_rejects_resume_without_checkpoint_path(monkeypatch) -> None:
    config = make_config()
    config["resume"] = {
        "enabled": True,
        "checkpoint_path": None,
    }

    monkeypatch.setattr(
        execution_train,
        "build_training_objects",
        lambda config, runtime_context: {
            "trainer": DummyTrainer(),
            "module": DummyModule(),
            "datamodule": DummyDataModule(),
        },
    )
    monkeypatch.setattr(
        execution_train,
        "save_execution_report",
        lambda execution_report, runtime_context, filename: None,
    )
    monkeypatch.setattr(execution_train, "close_external_services", lambda: None)

    with pytest.raises(ValueError, match="resume is enabled"):
        run_training(config)


def test_run_training_marks_failed_and_reraises(monkeypatch) -> None:
    trainer = DummyTrainer(should_fail=True)
    saved_reports = []

    monkeypatch.setattr(
        execution_train,
        "build_training_objects",
        lambda config, runtime_context: {
            "trainer": trainer,
            "module": DummyModule(),
            "datamodule": DummyDataModule(),
        },
    )
    monkeypatch.setattr(
        execution_train,
        "save_execution_report",
        lambda execution_report, runtime_context, filename: saved_reports.append(
            execution_report.copy()
        ),
    )
    monkeypatch.setattr(execution_train, "close_external_services", lambda: None)

    with pytest.raises(RuntimeError, match="fit failed"):
        run_training(make_config())

    assert saved_reports[-1]["status"] == "failed"


def test_run_validation_calls_validate_and_stores_outputs(monkeypatch) -> None:
    trainer = DummyTrainer()

    monkeypatch.setattr(
        execution_validate,
        "build_validation_objects",
        lambda config, runtime_context: {
            "trainer": trainer,
            "module": DummyModule(),
            "datamodule": DummyDataModule(),
        },
    )
    monkeypatch.setattr(
        execution_validate,
        "save_execution_report",
        lambda execution_report, runtime_context, filename: None,
    )
    monkeypatch.setattr(execution_validate, "close_external_services", lambda: None)

    report = run_validation(make_config())

    assert report["status"] == "finished"
    assert report["outputs"] == [{"val/loss": 1.0}]
    assert trainer.validate_called is True
