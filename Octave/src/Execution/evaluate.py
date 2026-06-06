from pathlib import Path

from ..Setup.setup import setup_runtime

from .cleanup import close_external_services
from .factory import build_evaluation_objects
from .reports import (
    initialize_execution_report,
    mark_execution_failed,
    mark_execution_finished,
    save_execution_report,
)

from ..Configs.savings import save_execution_snapshots


#############################################
# Evaluation execution
#############################################

# Final orchestration layer for evaluation.
#
# This file does not build objects directly.
# Object construction is delegated to Execution/factory.py.
#
# This file only:
#   - runs Setup/;
#   - requests built evaluation objects;
#   - launches trainer.validate or trainer.test;
#   - updates and saves the execution report;
#   - closes external services.


def run_evaluation(
    config: dict,
    evaluation_step: str = "test",
    config_path: str | Path | None = None,
) -> dict:
    """
    Run an evaluation process from a global experiment config.

    Args:
        config:
            Global experiment config.

        evaluation_step:
            Evaluation method to run. Expected values are:
                - "validate"
                - "test"

    Returns:
        Lightweight JSON-serializable execution report.
    """
    check_evaluation_step(evaluation_step)
    check_evaluation_restoration_config(config)

    runtime_context = setup_runtime(
        setup_config=config["setup"],
        config_path=config_path,
    )


    save_execution_snapshots(
        config=config,
        runtime_context=runtime_context,
        snapshot_dir=Path(runtime_context["paths"]["run_dir"]) / "configs",
    )

    execution_report = initialize_execution_report(
        runtime_context=runtime_context,
        execution_type=evaluation_step,
    )

    try:
        evaluation_objects = build_evaluation_objects(
            config=config,
            runtime_context=runtime_context,
        )

        evaluation_outputs = run_lightning_evaluation(
            trainer=evaluation_objects["trainer"],
            module=evaluation_objects["module"],
            datamodule=evaluation_objects["datamodule"],
            evaluation_step=evaluation_step,
        )

        execution_report["outputs"] = evaluation_outputs
        mark_execution_finished(execution_report)

    except Exception as error:
        mark_execution_failed(
            execution_report=execution_report,
            error=error,
        )
        raise

    finally:
        save_execution_report(
            execution_report=execution_report,
            runtime_context=runtime_context,
            filename=f"{evaluation_step}_execution_report.json",
        )

        close_external_services()

    return execution_report


##########################################################################
# Helpers
##########################################################################

def run_lightning_evaluation(
    trainer,
    module,
    datamodule,
    evaluation_step: str,
):
    """
    Dispatch to the requested Lightning evaluation method.
    """
    if evaluation_step == "validate":
        return trainer.validate(
            model=module,
            datamodule=datamodule,
        )

    if evaluation_step == "test":
        return trainer.test(
            model=module,
            datamodule=datamodule,
        )

    raise ValueError(f"Unknown evaluation step: {evaluation_step}.")


def check_evaluation_step(evaluation_step: str) -> None:
    """
    Check that the requested evaluation step is supported.
    """
    valid_evaluation_steps = {"validate", "test"}

    if evaluation_step not in valid_evaluation_steps:
        raise ValueError(
            f"Invalid evaluation_step: {evaluation_step}. "
            f"Expected one of {valid_evaluation_steps}."
        )
    

def check_evaluation_restoration_config(config: dict) -> None:
    """
    Prevent Trainer resume logic during evaluation.
    """
    resume_enabled = config.get("resume", {}).get("enabled", False)

    if resume_enabled:
        raise ValueError(
            "Invalid evaluation config: resume is enabled. "
            "Use module loading for evaluation, not Trainer resume."
        )