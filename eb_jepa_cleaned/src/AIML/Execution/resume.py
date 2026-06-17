"""
Resume-training run.

Like the training run, but restores the full training state (weights, optimizer,
step/epoch) from a Lightning checkpoint and continues. The checkpoint path comes
from the explicit argument, or `config["run"]["checkpoint_path"]`.
"""

from .Runs.cleanup import close_external_services
from .Runs.factory import build_training_objects
from .snapshots import snapshot_execution


def run_resume_training(
    config: dict,
    runtime_context: dict | None = None,
    checkpoint_path: str | None = None,
) -> dict:
    """Build the training objects and resume `trainer.fit` from a checkpoint."""
    checkpoint_path = checkpoint_path or _checkpoint_path_from_config(config)
    if not checkpoint_path:
        raise ValueError(
            "run_resume_training requires a checkpoint path (the --ckpt CLI flag "
            "or config['run']['checkpoint_path'])."
        )

    objects = build_training_objects(config, runtime_context=runtime_context)
    snapshot_execution(config, runtime_context)
    try:
        objects["trainer"].fit(
            objects["module"], objects["datamodule"], ckpt_path=checkpoint_path
        )
    finally:
        close_external_services()
    return objects


def _checkpoint_path_from_config(config: dict) -> str | None:
    return config.get("run", {}).get("checkpoint_path")
