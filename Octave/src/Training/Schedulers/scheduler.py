from copy import deepcopy

import torch

from .configs import (
    DEFAULT_COSINE_ANNEALING_LR_CONFIG,
    DEFAULT_REDUCE_LR_ON_PLATEAU_CONFIG,
    DEFAULT_STEP_LR_CONFIG,
)
from .registry import SCHEDULER_REGISTRY


LIGHTNING_SCHEDULER_KEYS = (
    "interval",
    "frequency",
    "monitor",
    "strict",
    "name",
)


class DisabledSchedulerBuilder:
    """
    Callable scheduler policy for disabled scheduler configs.
    """

    def __call__(self, optimizer: torch.optim.Optimizer) -> None:
        return None


class ConfiguredSchedulerBuilder:
    """
    Callable scheduler construction policy.

    The object is built from a plain scheduler config first.
    The actual PyTorch scheduler is instantiated later, once the optimizer exists.
    """

    scheduler_class = None

    def __init__(
        self,
        enabled: bool = True,
        interval: str | None = "epoch",
        frequency: int | None = 1,
        monitor: str | None = None,
        strict: bool | None = True,
        name: str | None = None,
        **scheduler_kwargs,
    ) -> None:
        if not enabled:
            raise ValueError(
                "ConfiguredSchedulerBuilder should only receive enabled scheduler configs. "
                "Disabled configs must be handled before registry construction."
            )

        if self.scheduler_class is None:
            raise ValueError(
                f"{self.__class__.__name__} must define scheduler_class."
            )

        lightning_config = {
            "interval": interval,
            "frequency": frequency,
            "monitor": monitor,
            "strict": strict,
            "name": name,
        }

        self.scheduler_kwargs = deepcopy(scheduler_kwargs)
        self.lightning_config = {
            key: value
            for key, value in lightning_config.items()
            if value is not None
        }

    def __call__(self, optimizer: torch.optim.Optimizer) -> dict:
        scheduler = self.scheduler_class(
            optimizer,
            **deepcopy(self.scheduler_kwargs),
        )

        return {
            "scheduler": scheduler,
            **deepcopy(self.lightning_config),
        }


@SCHEDULER_REGISTRY.register_class(
    name="step_lr",
    default_config=DEFAULT_STEP_LR_CONFIG,
    type_field="scheduler_type",
)
class StepLRSchedulerBuilder(ConfiguredSchedulerBuilder):
    scheduler_class = torch.optim.lr_scheduler.StepLR


@SCHEDULER_REGISTRY.register_class(
    name="cosine_annealing_lr",
    default_config=DEFAULT_COSINE_ANNEALING_LR_CONFIG,
    type_field="scheduler_type",
)
class CosineAnnealingLRSchedulerBuilder(ConfiguredSchedulerBuilder):
    scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR


@SCHEDULER_REGISTRY.register_class(
    name="reduce_lr_on_plateau",
    default_config=DEFAULT_REDUCE_LR_ON_PLATEAU_CONFIG,
    type_field="scheduler_type",
)
class ReduceLROnPlateauSchedulerBuilder(ConfiguredSchedulerBuilder):
    scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau