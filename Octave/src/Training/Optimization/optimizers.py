from copy import deepcopy

import torch

from .configs import DEFAULT_ADAM_CONFIG, DEFAULT_ADAMW_CONFIG, DEFAULT_SGD_CONFIG
from .registry import OPTIMIZER_REGISTRY


class ConfiguredOptimizerBuilder:
    """
    Callable optimizer construction policy.

    The object is built from a plain optimizer config first.
    The actual PyTorch optimizer is instantiated later, once parameters exist.
    """

    optimizer_class = None

    def __init__(self, **optimizer_kwargs) -> None:
        if self.optimizer_class is None:
            raise ValueError(
                f"{self.__class__.__name__} must define optimizer_class."
            )

        self.optimizer_kwargs = self.prepare_optimizer_kwargs(optimizer_kwargs)

    def __call__(self, parameters) -> torch.optim.Optimizer:
        return self.optimizer_class(
            parameters,
            **deepcopy(self.optimizer_kwargs),
        )

    def prepare_optimizer_kwargs(self, optimizer_kwargs: dict) -> dict:
        prepared_kwargs = deepcopy(optimizer_kwargs)

        if "betas" in prepared_kwargs:
            prepared_kwargs["betas"] = tuple(prepared_kwargs["betas"])

        return prepared_kwargs


@OPTIMIZER_REGISTRY.register_class(
    name="adamw",
    default_config=DEFAULT_ADAMW_CONFIG,
    type_field="optimizer_type",
)
class AdamWOptimizerBuilder(ConfiguredOptimizerBuilder):
    optimizer_class = torch.optim.AdamW


@OPTIMIZER_REGISTRY.register_class(
    name="adam",
    default_config=DEFAULT_ADAM_CONFIG,
    type_field="optimizer_type",
)
class AdamOptimizerBuilder(ConfiguredOptimizerBuilder):
    optimizer_class = torch.optim.Adam


@OPTIMIZER_REGISTRY.register_class(
    name="sgd",
    default_config=DEFAULT_SGD_CONFIG,
    type_field="optimizer_type",
)
class SGDOptimizerBuilder(ConfiguredOptimizerBuilder):
    optimizer_class = torch.optim.SGD