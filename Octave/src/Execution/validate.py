from pathlib import Path

from .cleanup import close_external_services
from .factory import build_validation_objects
from .reports import (
    initialize_execution_report,
    mark_execution_failed,
    mark_execution_finished,
    save_execution_report,
)
from ..Setup.config_resolution import load_config
from ..Setup.setup import setup_runtime


def run_validation(
    config: dict | None = None,
    config_path: str | Path | None = None,
    runtime_context: dict | None = None,
) -> dict:
    config = resolve_execution_config(config=config, config_path=config_path)
    runtime_context = runtime_context or config.get("runtime_context")

    if runtime_context is None:
        runtime_context = setup_runtime(
            setup_config=config.get("setup", {}),
            config_path=config_path,
        )

    execution_report = initialize_execution_report(
        runtime_context=runtime_context,
        execution_type="validate",
    )

    try:
        objects = build_validation_objects(
            config=config,
            runtime_context=runtime_context,
        )
        outputs = objects["trainer"].validate(
            model=objects["module"],
            datamodule=objects["datamodule"],
        )
        mark_execution_finished(execution_report, outputs=outputs)

    except Exception as error:
        mark_execution_failed(execution_report, error)
        raise

    finally:
        save_execution_report(
            execution_report=execution_report,
            runtime_context=runtime_context,
            filename="validate_execution_report.json",
        )
        close_external_services()

    return execution_report


def resolve_execution_config(
    config: dict | None = None,
    config_path: str | Path | None = None,
) -> dict:
    if config is not None:
        return config

    if config_path is None:
        raise KeyError("Either config or config_path must be provided.")

    return load_config(config_path)
