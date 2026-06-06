import json
from pathlib import Path
from typing import Any


#############################################
# Execution reports
#############################################

# Utilities for lightweight execution reports.
#
# Reports are meant to be shared by train / evaluation / prediction scripts.
# They must remain JSON-serializable and should not store built Python objects.


def initialize_execution_report(
    runtime_context: dict,
    execution_type: str,
) -> dict:
    """
    Initialize a lightweight execution report.

    Args:
        runtime_context:
            Runtime facts returned by Setup/.

        execution_type:
            Name of the executed process, e.g. "train", "validate", "test".

    Returns:
        JSON-serializable execution report.
    """
    return {
        "execution_type": execution_type,
        "status": "running",
        "run_dir": get_run_dir(runtime_context),
        "runtime_context": make_json_serializable(runtime_context),
    }


def mark_execution_finished(execution_report: dict) -> None:
    """
    Mark an execution report as successfully finished.
    """
    execution_report["status"] = "finished"


def mark_execution_failed(
    execution_report: dict,
    error: Exception,
) -> None:
    """
    Mark an execution report as failed while keeping the error readable.
    """
    execution_report["status"] = "failed"
    execution_report["error"] = repr(error)


def save_execution_report(
    execution_report: dict,
    runtime_context: dict,
    filename: str = "execution_report.json",
) -> None:
    """
    Save an execution report alongside the run outputs.
    """
    run_dir = Path(get_run_dir(runtime_context))
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / filename

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            execution_report,
            file,
            indent=4,
            ensure_ascii=False,
        )


def get_run_dir(runtime_context: dict) -> str:
    """
    Retrieve the run output directory from the runtime context.
    """
    return runtime_context["paths"]["run_dir"]


def make_json_serializable(value: Any):
    """
    Convert common non-JSON objects into JSON-compatible values.
    """
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(sub_value)
            for key, sub_value in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_serializable(sub_value) for sub_value in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return repr(value)