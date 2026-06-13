from pathlib import Path

from Octave.src.Execution.reports import (
    initialize_execution_report,
    mark_execution_finished,
    save_execution_report,
)


def test_execution_report_is_saved_as_json(tmp_path: Path) -> None:
    runtime_context = {
        "paths": {
            "run_dir": tmp_path / "run",
        },
    }
    report = initialize_execution_report(
        runtime_context=runtime_context,
        execution_type="train",
    )
    mark_execution_finished(report, outputs=[{"metric": 1.0}])

    report_path = save_execution_report(
        execution_report=report,
        runtime_context=runtime_context,
        filename="report.json",
    )

    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8").startswith("{")
