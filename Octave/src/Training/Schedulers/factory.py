import torch

from . import schedulers  # noqa: F401
from .schedulers import ConfiguredSchedulerBuilder, DisabledSchedulerBuilder
from ...Workflow.Factory.builder import RegistryBuilder
from .registry import SCHEDULER_REGISTRY


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict | None,
    strict: bool = True,
) -> dict | None:
    scheduler_builder = build_scheduler_builder(
        scheduler_config=scheduler_config,
        strict=strict,
    )

    return scheduler_builder(optimizer)


def build_scheduler_builder(
    scheduler_config: dict | None,
    strict: bool = True,
) -> ConfiguredSchedulerBuilder | DisabledSchedulerBuilder:
    if not is_scheduler_enabled(scheduler_config):
        return DisabledSchedulerBuilder()

    builder = RegistryBuilder(
        registry=SCHEDULER_REGISTRY,
        strict=strict,
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