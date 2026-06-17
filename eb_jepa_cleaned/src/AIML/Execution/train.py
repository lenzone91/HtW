"""
Training run.

Composes the training objects (datamodule + module + Trainer), snapshots the
run, and fits from scratch. Generic: works for any registered module/datamodule.
"""

from .Runs.cleanup import close_external_services
from .Runs.factory import build_training_objects
from .snapshots import snapshot_execution


def run_training(config: dict, runtime_context: dict | None = None) -> dict:
    """Build the training objects and run `trainer.fit` from scratch."""
    objects = build_training_objects(config, runtime_context=runtime_context)
    snapshot_execution(config, runtime_context)
    try:
        objects["trainer"].fit(objects["module"], objects["datamodule"])
    finally:
        close_external_services()
    return objects
