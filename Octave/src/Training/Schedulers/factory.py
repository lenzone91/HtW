import torch

from . import scheduler  # noqa: F401
from .scheduler import ConfiguredSchedulerBuilder, DisabledSchedulerBuilder
from ...Workflow.Factory.builder import RegistryBuilder
from .registry import SCHEDULER_REGISTRY


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict | None,
) -> dict | None:
    scheduler_builder = build_scheduler_builder(
        scheduler_config=scheduler_config,
    )

    return scheduler_builder(optimizer)


def build_scheduler_builder(
    scheduler_config: dict | None,
) -> ConfiguredSchedulerBuilder | DisabledSchedulerBuilder:
    if not is_scheduler_enabled(scheduler_config):
        return DisabledSchedulerBuilder()

    builder = RegistryBuilder(
        registry=SCHEDULER_REGISTRY,
        type_field="scheduler_type",
    )

    return builder.build_one(config=scheduler_config)


def is_scheduler_enabled(scheduler_config: dict | None) -> bool:
    if scheduler_config is None:
        return False

    if not isinstance(scheduler_config, dict):
        raise TypeError(
            "Scheduler config must be a dictionary, "
            f"got {type(scheduler_config).__name__}."
        )

    return scheduler_config.get("enabled", False)
