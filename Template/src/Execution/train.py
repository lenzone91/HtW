from pathlib import Path

from ..Setup.setup import setup_runtime

from .cleanup import close_external_services
from .factory import build_training_objects
from .reports import (
    initialize_execution_report,
    mark_execution_failed,
    mark_execution_finished,
    save_execution_report,
)

from ..Configs.savings import save_execution_snapshots


#############################################
# Training execution
#############################################

# Final orchestration layer for training.
#
# This file does not build objects directly.
# Object construction is delegated to Execution/factory.py.
#
# This file only:
#   - runs Setup/;
#   - requests built training objects;
#   - launches trainer.fit;
#   - updates and saves the execution report;
#   - closes external services.


def run_training(
    config: dict,
    config_path: str | Path | None = None,
) -> dict:
    """
    Run a full training process from a global experiment config.
    """
    check_training_restoration_config(config)

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
        execution_type="train",
    )

    try:
        training_objects = build_training_objects(
            config=config,
            runtime_context=runtime_context,
        )

        training_objects["trainer"].fit(
            model=training_objects["module"],
            datamodule=training_objects["datamodule"],
            ckpt_path=get_resume_checkpoint_path(config.get("resume", {})),
        )

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
            filename="train_execution_report.json",
        )

        close_external_services()

    return execution_report



#############################################
# Resume handling
#############################################


def get_resume_checkpoint_path(resume_config: dict | None) -> str | None:
    """
    Return the checkpoint path used for Lightning Trainer resume.
    """
    if resume_config is None:
        return None

    if not resume_config.get("enabled", False):
        return None

    checkpoint_path = resume_config.get("checkpoint_path")

    if checkpoint_path is None:
        raise ValueError(
            "Training resume is enabled, but no checkpoint_path was provided."
        )

    return checkpoint_path


def check_training_restoration_config(config: dict) -> None:
    """
    Prevent ambiguous restoration modes during training.
    """
    resume_enabled = config.get("resume", {}).get("enabled", False)

    module_loading_enabled = (
        config.get("loading", {})
        .get("module", {})
        .get("enabled", False)
    )

    if resume_enabled and module_loading_enabled:
        raise ValueError(
            "Invalid training config: resume and module loading are both enabled. "
            "Use resume for continuing a Lightning training run, and module loading "
            "only for weight initialization / evaluation / fine-tuning."
        )