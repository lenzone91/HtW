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


class ConfiguredSchedulerBuilder:
    """
    Callable scheduler construction policy built from a plain config.
    """

    def __init__(self, scheduler_config: dict | None) -> None:
        self.enabled = is_scheduler_enabled(scheduler_config)
        if not self.enabled:
            self.scheduler_class = None
            self.scheduler_kwargs = {}
            self.lightning_config = {}
            return

        prepared_config = prepare_scheduler_config(scheduler_config)
        scheduler_type = prepared_config.pop("scheduler_type")
        self.scheduler_class, _ = SCHEDULER_REGISTRY[scheduler_type]

        lightning_config = {
            key: prepared_config.pop(key)
            for key in LIGHTNING_SCHEDULER_KEYS
        }
        self.lightning_config = {
            key: value
            for key, value in lightning_config.items()
            if value is not None
        }
        self.scheduler_kwargs = prepared_config

    def __call__(self, optimizer: torch.optim.Optimizer) -> dict | None:
        if not self.enabled:
            return None

        scheduler = self.scheduler_class(
            optimizer,
            **deepcopy(self.scheduler_kwargs),
        )
        return {
            "scheduler": scheduler,
            **deepcopy(self.lightning_config),
        }


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict | None,
) -> dict | None:
    """
    Build a Lightning-compatible scheduler dictionary.
    """
    scheduler_builder = build_scheduler_builder(scheduler_config=scheduler_config)
    return scheduler_builder(optimizer)


def build_scheduler_builder(
    scheduler_config: dict | None,
) -> ConfiguredSchedulerBuilder:
    return ConfiguredSchedulerBuilder(scheduler_config=scheduler_config)


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
