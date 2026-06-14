from copy import deepcopy

import torch

from .configs import DEFAULT_ADAM_CONFIG, DEFAULT_ADAMW_CONFIG, DEFAULT_SGD_CONFIG


OPTIMIZER_REGISTRY = {
    "adam": (torch.optim.Adam, DEFAULT_ADAM_CONFIG),
    "adamw": (torch.optim.AdamW, DEFAULT_ADAMW_CONFIG),
    "sgd": (torch.optim.SGD, DEFAULT_SGD_CONFIG),
}


class ConfiguredOptimizerBuilder:
    """
    Callable optimizer construction policy built from a plain config.
    """

    def __init__(self, optimizer_config: dict) -> None:
        prepared_config = prepare_optimizer_config(optimizer_config)
        optimizer_type = prepared_config.pop("optimizer_type")
        self.optimizer_class, _ = OPTIMIZER_REGISTRY[optimizer_type]
        self.optimizer_kwargs = prepared_config

    def __call__(self, parameters) -> torch.optim.Optimizer:
        return self.optimizer_class(parameters, **deepcopy(self.optimizer_kwargs))


def build_optimizer(
    parameters,
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    """
    Build one optimizer from a plain dictionary config.
    """
    optimizer_builder = build_optimizer_builder(optimizer_config=optimizer_config)
    return optimizer_builder(parameters)


def build_optimizer_builder(optimizer_config: dict) -> ConfiguredOptimizerBuilder:
    return ConfiguredOptimizerBuilder(optimizer_config=optimizer_config)


def prepare_optimizer_config(optimizer_config: dict) -> dict:
    if not isinstance(optimizer_config, dict):
        raise TypeError(
            "Optimizer config must be a dictionary, "
            f"got {type(optimizer_config).__name__}."
        )

    optimizer_type = optimizer_config.get("optimizer_type")

    if optimizer_type not in OPTIMIZER_REGISTRY:
        raise KeyError(
            f"Unknown optimizer_type '{optimizer_type}'. "
            f"Available optimizer types are: {sorted(OPTIMIZER_REGISTRY)}."
        )

    _, default_config = OPTIMIZER_REGISTRY[optimizer_type]
    prepared_config = deepcopy(default_config)
    user_config = deepcopy(optimizer_config)

    unknown_keys = set(user_config) - set(default_config)
    if unknown_keys:
        raise KeyError(
            "Unknown optimizer config keys: "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(default_config)}."
        )

    prepared_config.update(user_config)
    prepared_config["betas"] = tuple(prepared_config["betas"])

    return prepared_config
