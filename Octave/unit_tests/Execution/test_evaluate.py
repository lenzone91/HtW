"""
Tests for evaluation execution utilities.

This file validates evaluation-step checks, Lightning evaluation dispatch, and
high-level evaluation orchestration.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.

The monkeypatch fixture is used to temporarily replace setup, object builders,
snapshot saving, report saving, and cleanup. This isolates evaluate.py
orchestration from Setup/, factories, Lightning internals, filesystem snapshots,
and external services.
"""

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Execution import (
    evaluate as execution_evaluate,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Execution.evaluate import (
    check_evaluation_restoration_config,
    check_evaluation_step,
    run_evaluation,
    run_lightning_evaluation,
)


#############################################
# Dummy objects
#############################################


class DummyTrainer:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.validate_called = False
        self.test_called = False
        self.validate_kwargs = None
        self.test_kwargs = None

    def validate(self, model, datamodule):
        self.validate_called = True
        self.validate_kwargs = {
            "model": model,
            "datamodule": datamodule,
        }

        if self.should_fail:
            raise RuntimeError("dummy evaluation failure")

        return [{"val_metric": 1.0}]

    def test(self, model, datamodule):
        self.test_called = True
        self.test_kwargs = {
            "model": model,
            "datamodule": datamodule,
        }

        if self.should_fail:
            raise RuntimeError("dummy evaluation failure")

        return [{"test_metric": 1.0}]


class DummyModule:
    pass


class DummyDataModule:
    pass


#############################################
# Helpers
#############################################


def make_runtime_context() -> dict:
    return {
        "paths": {
            "run_dir": "dummy_run_dir",
        },
    }


def make_evaluation_config() -> dict:
    return {
        "setup": {},
        "datamodule": {},
        "module": {},
        "loading": {"module": {"enabled": False}},
        "resume": {"enabled": False, "checkpoint_path": None},
    }


#############################################
# Evaluation-step checks
#############################################


def test_check_evaluation_step_accepts_validate_and_test() -> None:
    check_evaluation_step("validate")
    check_evaluation_step("test")


def test_check_evaluation_step_rejects_invalid_step() -> None:
    with pytest.raises(ValueError):
        check_evaluation_step("predict")


#############################################
# Restoration config checks
#############################################


def test_check_evaluation_restoration_config_accepts_resume_disabled() -> None:
    check_evaluation_restoration_config(
        config={
            "resume": {"enabled": False, "checkpoint_path": None},
        },
    )


def test_check_evaluation_restoration_config_rejects_resume_enabled() -> None:
    with pytest.raises(ValueError):
        check_evaluation_restoration_config(
            config={
                "resume": {"enabled": True, "checkpoint_path": "checkpoint.ckpt"},
            },
        )


#############################################
# Lightning evaluation dispatch
#############################################


def test_run_lightning_evaluation_dispatches_validate() -> None:
    trainer = DummyTrainer()
    module = DummyModule()
    datamodule = DummyDataModule()

    outputs = run_lightning_evaluation(
        trainer=trainer,
        module=module,
        datamodule=datamodule,
        evaluation_step="validate",
    )

    assert trainer.validate_called is True
    assert trainer.test_called is False
    assert trainer.validate_kwargs == {
        "model": module,
        "datamodule": datamodule,
    }
    assert outputs == [{"val_metric": 1.0}]


def test_run_lightning_evaluation_dispatches_test() -> None:
    trainer = DummyTrainer()
    module = DummyModule()
    datamodule = DummyDataModule()

    outputs = run_lightning_evaluation(
        trainer=trainer,
        module=module,
        datamodule=datamodule,
        evaluation_step="test",
    )

    assert trainer.test_called is True
    assert trainer.validate_called is False
    assert trainer.test_kwargs == {
        "model": module,
        "datamodule": datamodule,
    }
    assert outputs == [{"test_metric": 1.0}]


def test_run_lightning_evaluation_rejects_unknown_step() -> None:
    with pytest.raises(ValueError):
        run_lightning_evaluation(
            trainer=DummyTrainer(),
            module=DummyModule(),
            datamodule=DummyDataModule(),
            evaluation_step="predict",
        )


#############################################
# Evaluation orchestration
#############################################


def test_run_evaluation_marks_report_finished_and_stores_outputs(monkeypatch) -> None:
    trainer = DummyTrainer()
    module = DummyModule()
    datamodule = DummyDataModule()
    saved_reports = []
    cleanup_called = {"value": False}

    monkeypatch.setattr(
        execution_evaluate,
        "setup_runtime",
        lambda setup_config, config_path=None: make_runtime_context(),
    )
    monkeypatch.setattr(
        execution_evaluate,
        "save_execution_snapshots",
        lambda config, runtime_context, snapshot_dir: None,
    )
    monkeypatch.setattr(
        execution_evaluate,
        "build_evaluation_objects",
        lambda config, runtime_context: {
            "trainer": trainer,
            "module": module,
            "datamodule": datamodule,
        },
    )
    monkeypatch.setattr(
        execution_evaluate,
        "save_execution_report",
        lambda execution_report, runtime_context, filename: saved_reports.append(
            execution_report.copy()
        ),
    )

    def fake_close_external_services() -> None:
        cleanup_called["value"] = True

    monkeypatch.setattr(
        execution_evaluate,
        "close_external_services",
        fake_close_external_services,
    )

    report = run_evaluation(
        config=make_evaluation_config(),
        evaluation_step="test",
    )

    assert report["status"] == "finished"
    assert report["outputs"] == [{"test_metric": 1.0}]
    assert trainer.test_called is True
    assert saved_reports[-1]["status"] == "finished"
    assert saved_reports[-1]["outputs"] == [{"test_metric": 1.0}]
    assert cleanup_called["value"] is True


def test_run_evaluation_marks_report_failed_and_reraises_error(monkeypatch) -> None:
    trainer = DummyTrainer(should_fail=True)
    saved_reports = []
    cleanup_called = {"value": False}

    monkeypatch.setattr(
        execution_evaluate,
        "setup_runtime",
        lambda setup_config, config_path=None: make_runtime_context(),
    )
    monkeypatch.setattr(
        execution_evaluate,
        "save_execution_snapshots",
        lambda config, runtime_context, snapshot_dir: None,
    )
    monkeypatch.setattr(
        execution_evaluate,
        "build_evaluation_objects",
        lambda config, runtime_context: {
            "trainer": trainer,
            "module": DummyModule(),
            "datamodule": DummyDataModule(),
        },
    )
    monkeypatch.setattr(
        execution_evaluate,
        "save_execution_report",
        lambda execution_report, runtime_context, filename: saved_reports.append(
            execution_report.copy()
        ),
    )

    def fake_close_external_services() -> None:
        cleanup_called["value"] = True

    monkeypatch.setattr(
        execution_evaluate,
        "close_external_services",
        fake_close_external_services,
    )

    with pytest.raises(RuntimeError):
        run_evaluation(
            config=make_evaluation_config(),
            evaluation_step="test",
        )

    assert saved_reports[-1]["status"] == "failed"
    assert "dummy evaluation failure" in saved_reports[-1]["error"]
    assert cleanup_called["value"] is True