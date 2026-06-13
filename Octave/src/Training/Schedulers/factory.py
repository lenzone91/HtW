from copy import deepcopy

import torch

from .configs import (
    DEFAULT_COSINE_ANNEALING_LR_CONFIG,
    DEFAULT_REDUCE_LR_ON_PLATEAU_CONFIG,
    DEFAULT_STEP_LR_CONFIG,
)


LIGHTNING_SCHEDULER_KEYS = {
    "interval",
    "frequency",
    "monitor",
    "strict",
    "name",
}


SCHEDULER_REGISTRY = {
    "cosine_annealing_lr": (
        torch.optim.lr_scheduler.CosineAnnealingLR,
        DEFAULT_COSINE_ANNEALING_LR_CONFIG,
    ),
    "reduce_lr_on_plateau": (
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        DEFAULT_REDUCE_LR_ON_PLATEAU_CONFIG,
    ),
    "step_lr": (
        torch.optim.lr_scheduler.StepLR,
        DEFAULT_STEP_LR_CONFIG,
    ),
}


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict | None,
) -> dict | None:
    """
    Build a Lightning-compatible scheduler dictionary.
    """
    if not is_scheduler_enabled(scheduler_config):
        return None

    prepared_config = prepare_scheduler_config(scheduler_config)
    scheduler_type = prepared_config.pop("scheduler_type")
    scheduler_class, _ = SCHEDULER_REGISTRY[scheduler_type]

    lightning_config = {
        key: prepared_config.pop(key)
        for key in LIGHTNING_SCHEDULER_KEYS
    }
    lightning_config = {
        key: value
        for key, value in lightning_config.items()
        if value is not None
    }

    scheduler = scheduler_class(optimizer, **prepared_config)
    return {
        "scheduler": scheduler,
        **lightning_config,
    }


def is_scheduler_enabled(scheduler_config: dict | None) -> bool:
    if scheduler_config is None:
        return False

    if not isinstance(scheduler_config, dict):
        raise TypeError(
            "Scheduler config must be a dictionary, "
            f"got {type(scheduler_config).__name__}."
        )

    return scheduler_config.get("enabled", False)


def prepare_scheduler_config(scheduler_config: dict) -> dict:
    if not isinstance(scheduler_config, dict):
        raise TypeError(
            "Scheduler config must be a dictionary, "
            f"got {type(scheduler_config).__name__}."
        )

    scheduler_type = scheduler_config.get("scheduler_type")

    if scheduler_type not in SCHEDULER_REGISTRY:
        raise KeyError(
            f"Unknown scheduler_type '{scheduler_type}'. "
            f"Available scheduler types are: {sorted(SCHEDULER_REGISTRY)}."
        )

    _, default_config = SCHEDULER_REGISTRY[scheduler_type]
    prepared_config = deepcopy(default_config)
    user_config = deepcopy(scheduler_config)

    unknown_keys = set(user_config) - set(default_config)
    if unknown_keys:
        raise KeyError(
            "Unknown scheduler config keys: "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(default_config)}."
        )

    prepared_config.update(user_config)
    prepared_config.pop("enabled")

    return prepared_config
