import torch

from . import schedulers  # noqa: F401  (registers schedulers)
from .registry import SCHEDULER_BUILDER


#############################################
# Scheduler building wrappers
#############################################


def check_matching_optimizer_groups(
    scheduler_configs: dict,
    optimizer_groups: dict,
) -> None:
    """
    Check that every scheduler config has a matching optimizer.
    """
    if not isinstance(optimizer_groups, dict):
        raise TypeError(
            f"optimizer_groups must be a dictionary, got {type(optimizer_groups)}."
        )

    missing_names = set(scheduler_configs.keys()) - set(optimizer_groups.keys())

    if missing_names:
        raise KeyError(
            f"Missing optimizers for {sorted(missing_names)}. "
            f"Available optimizers are: {sorted(optimizer_groups.keys())}."
        )


def build_schedulers(
    optimizer_groups: dict[str, torch.optim.Optimizer],
    scheduler_configs: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Build one scheduler per named optimizer.
    """
    check_matching_optimizer_groups(
        scheduler_configs=scheduler_configs,
        optimizer_groups=optimizer_groups,
    )

    return {
        scheduler_name: SCHEDULER_BUILDER.build_one(
            config=scheduler_config,
            runtime_context=runtime_context,
            optimizer=optimizer_groups[scheduler_name],
        )
        for scheduler_name, scheduler_config in scheduler_configs.items()
    }
