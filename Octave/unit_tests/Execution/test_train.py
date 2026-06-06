"""
Tests for training execution utilities.

This file validates resume handling, restoration-config checks, and high-level
training orchestration.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.

The monkeypatch fixture is used to temporarily replace setup, object builders,
snapshot saving, report saving, and cleanup. This isolates train.py orchestration
from Setup/, factories, Lightning internals, filesystem snapshots, and external
services.
"""

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Execution import (
    train as execution_train,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Execution.train import (
    check_training_restoration_config,
    get_resume_checkpoint_path,
    run_training,
)


#############################################
# Dummy objects
#############################################


class DummyTrainer:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.fit_called = False
        self.fit_kwargs = None

    def fit(self, model, datamodule, ckpt_path=None) -> None:
        self.fit_called = True
        self.fit_kwargs = {
            "model": model,
            "datamodule": datamodule,
            "ckpt_path": ckpt_path,
        }

        if self.should_fail:
            raise RuntimeError("dummy training failure")


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


def make_training_config() -> dict:
    return {
        "setup": {},
        "datamodule": {},
        "module": {},
        "loading": {"module": {"enabled": False}},
        "resume": {"enabled": False, "checkpoint_path": None},
    }


#############################################
# Resume checkpoint path
#############################################


def test_get_resume_checkpoint_path_returns_none_without_config() -> None:
    assert get_resume_checkpoint_path(None) is None


def test_get_resume_checkpoint_path_returns_none_when_disabled() -> None:
    checkpoint_path = get_resume_checkpoint_path(
        resume_config={
            "enabled": False,
            "checkpoint_path": "checkpoint.ckpt",
        },
    )

    assert checkpoint_path is None


def test_get_resume_checkpoint_path_rejects_missing_checkpoint_when_enabled() -> None:
    with pytest.raises(ValueError):
        get_resume_checkpoint_path(
            resume_config={
                "enabled": True,
                "checkpoint_path": None,
            },
        )


def test_get_resume_checkpoint_path_returns_checkpoint_when_enabled() -> None:
    checkpoint_path = get_resume_checkpoint_path(
        resume_config={
            "enabled": True,
            "checkpoint_path": "checkpoint.ckpt",
        },
    )

    assert checkpoint_path == "checkpoint.ckpt"


#############################################
# Restoration config checks
#############################################


def test_check_training_restoration_config_accepts_non_ambiguous_config() -> None:
    check_training_restoration_config(
        config={
            "resume": {"enabled": True, "checkpoint_path": "checkpoint.ckpt"},
            "loading": {"module": {"enabled": False}},
        },
    )

    check_training_restoration_config(
        config={
            "resume": {"enabled": False, "checkpoint_path": None},
            "loading": {"module": {"enabled": True}},
        },
    )


def test_check_training_restoration_config_rejects_resume_and_module_loading() -> None:
    with pytest.raises(ValueError):
        check_training_restoration_config(
            config={
                "resume": {"enabled": True, "checkpoint_path": "checkpoint.ckpt"},
                "loading": {"module": {"enabled": True}},
            },
        )


#############################################
# Training orchestration
#############################################


def test_run_training_marks_report_finished_and_calls_fit(monkeypatch) -> None:
    trainer = DummyTrainer()
    module = DummyModule()
    datamodule = DummyDataModule()
    saved_reports = []
    cleanup_called = {"value": False}

    monkeypatch.setattr(
        execution_train,
        "setup_runtime",
        lambda setup_config, config_path=None: make_runtime_context(),
    )
    monkeypatch.setattr(
        execution_train,
        "save_execution_snapshots",
        lambda config, runtime_context, snapshot_dir: None,
    )
    monkeypatch.setattr(
        execution_train,
        "build_training_objects",
        lambda config, runtime_context: {
            "trainer": trainer,
            "module": module,
            "datamodule": datamodule,
        },
    )
    monkeypatch.setattr(
        execution_train,
        "save_execution_report",
        lambda execution_report, runtime_context, filename: saved_reports.append(
            execution_report.copy()
        ),
    )

    def fake_close_external_services() -> None:
        cleanup_called["value"] = True

    monkeypatch.setattr(
        execution_train,
        "close_external_services",
        fake_close_external_services,
    )

    config = make_training_config()
    report = run_training(config)

    assert report["status"] == "finished"
    assert trainer.fit_called is True
    assert trainer.fit_kwargs == {
        "model": module,
        "datamodule": datamodule,
        "ckpt_path": None,
    }
    assert saved_reports[-1]["status"] == "finished"
    assert cleanup_called["value"] is True


def test_run_training_marks_report_failed_and_reraises_error(monkeypatch) -> None:
    trainer = DummyTrainer(should_fail=True)
    saved_reports = []
    cleanup_called = {"value": False}

    monkeypatch.setattr(
        execution_train,
        "setup_runtime",
        lambda setup_config, config_path=None: make_runtime_context(),
    )
    monkeypatch.setattr(
        execution_train,
        "save_execution_snapshots",
        lambda config, runtime_context, snapshot_dir: None,
    )
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

    def fake_close_external_services() -> None:
        cleanup_called["value"] = True

    monkeypatch.setattr(
        execution_train,
        "close_external_services",
        fake_close_external_services,
    )

    with pytest.raises(RuntimeError):
        run_training(make_training_config())

    assert saved_reports[-1]["status"] == "failed"
    assert "dummy training failure" in saved_reports[-1]["error"]
    assert cleanup_called["value"] is True