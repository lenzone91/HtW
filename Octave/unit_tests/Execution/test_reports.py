"""
Tests for execution report utilities.

This file validates report initialization, status updates, JSON serialization,
and report saving.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import json
from pathlib import Path

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Execution.reports import (
    get_run_dir,
    initialize_execution_report,
    make_json_serializable,
    mark_execution_failed,
    mark_execution_finished,
    save_execution_report,
)


#############################################
# Helpers
#############################################


def make_runtime_context(tmp_path: Path) -> dict:
    return {
        "paths": {
            "run_dir": str(tmp_path / "run"),
        },
    }


#############################################
# Runtime context helpers
#############################################


def test_get_run_dir_returns_runtime_context_run_dir(tmp_path: Path) -> None:
    runtime_context = make_runtime_context(tmp_path)

    assert get_run_dir(runtime_context) == str(tmp_path / "run")


#############################################
# JSON serialization
#############################################


def test_make_json_serializable_preserves_primitives() -> None:
    value = {
        "string": "text",
        "integer": 1,
        "float": 1.5,
        "boolean": True,
        "none": None,
    }

    assert make_json_serializable(value) == value


def test_make_json_serializable_converts_path_to_string(tmp_path: Path) -> None:
    path = tmp_path / "run"

    assert make_json_serializable(path) == str(path)


def test_make_json_serializable_converts_nested_structures(tmp_path: Path) -> None:
    value = {
        "path": tmp_path / "run",
        "tuple": (tmp_path / "a", "b"),
        "nested": {
            "list": [tmp_path / "c", 1],
        },
    }

    serialized_value = make_json_serializable(value)

    assert serialized_value == {
        "path": str(tmp_path / "run"),
        "tuple": [str(tmp_path / "a"), "b"],
        "nested": {
            "list": [str(tmp_path / "c"), 1],
        },
    }


#############################################
# Report status
#############################################


def test_initialize_execution_report_creates_running_report(tmp_path: Path) -> None:
    runtime_context = make_runtime_context(tmp_path)

    report = initialize_execution_report(
        runtime_context=runtime_context,
        execution_type="train",
    )

    assert report["execution_type"] == "train"
    assert report["status"] == "running"
    assert report["run_dir"] == str(tmp_path / "run")
    assert report["runtime_context"] == runtime_context


def test_mark_execution_finished_sets_finished_status() -> None:
    report = {"status": "running"}

    mark_execution_finished(report)

    assert report["status"] == "finished"


def test_mark_execution_failed_sets_failed_status_and_error() -> None:
    report = {"status": "running"}
    error = ValueError("dummy error")

    mark_execution_failed(
        execution_report=report,
        error=error,
    )

    assert report["status"] == "failed"
    assert report["error"] == repr(error)


#############################################
# Report saving
#############################################


def test_save_execution_report_writes_json_file(tmp_path: Path) -> None:
    runtime_context = make_runtime_context(tmp_path)
    report = {
        "execution_type": "train",
        "status": "finished",
    }

    save_execution_report(
        execution_report=report,
        runtime_context=runtime_context,
        filename="report.json",
    )

    report_path = tmp_path / "run" / "report.json"

    assert report_path.is_file()

    with report_path.open("r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == report