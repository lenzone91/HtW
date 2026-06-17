"""
Validation run.

Composes the evaluation objects (datamodule + module + Trainer, no training
callbacks) — module weights are loaded if `config["loading"]["module"]` is set —
snapshots the run, and runs `trainer.validate`.
"""

from .Runs.cleanup import close_external_services
from .Runs.factory import build_evaluation_objects
from .snapshots import snapshot_execution


def run_validation(config: dict, runtime_context: dict | None = None) -> dict:
    """Build the evaluation objects and run `trainer.validate`."""
    objects = build_evaluation_objects(config, runtime_context=runtime_context)
    snapshot_execution(config, runtime_context)
    try:
        objects["trainer"].validate(objects["module"], objects["datamodule"])
    finally:
        close_external_services()
    return objects
