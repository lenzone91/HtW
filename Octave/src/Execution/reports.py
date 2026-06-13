import json
from pathlib import Path
from typing import Any


def initialize_execution_report(
    runtime_context: dict,
    execution_type: str,
) -> dict:
    return {
        "execution_type": execution_type,
        "status": "running",
        "run_dir": get_run_dir(runtime_context),
        "runtime_context": make_json_serializable(runtime_context),
        "outputs": None,
    }


def mark_execution_finished(
    execution_report: dict,
    outputs=None,
) -> None:
    execution_report["status"] = "finished"
    execution_report["outputs"] = make_json_serializable(outputs)


def mark_execution_failed(
    execution_report: dict,
    error: Exception,
) -> None:
    execution_report["status"] = "failed"
    execution_report["error"] = repr(error)


def save_execution_report(
    execution_report: dict,
    runtime_context: dict,
    filename: str,
) -> Path:
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

    return report_path


def get_run_dir(runtime_context: dict) -> str:
    return str(runtime_context["paths"]["run_dir"])


def make_json_serializable(value: Any):
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
