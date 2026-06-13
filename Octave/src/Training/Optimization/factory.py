from copy import deepcopy

import torch

from .configs import DEFAULT_ADAM_CONFIG, DEFAULT_ADAMW_CONFIG, DEFAULT_SGD_CONFIG


OPTIMIZER_REGISTRY = {
    "adam": (torch.optim.Adam, DEFAULT_ADAM_CONFIG),
    "adamw": (torch.optim.AdamW, DEFAULT_ADAMW_CONFIG),
    "sgd": (torch.optim.SGD, DEFAULT_SGD_CONFIG),
}


def build_optimizer(
    parameters,
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    """
    Build one optimizer from a plain dictionary config.
    """
    prepared_config = prepare_optimizer_config(optimizer_config)
    optimizer_type = prepared_config.pop("optimizer_type")
    optimizer_class, _ = OPTIMIZER_REGISTRY[optimizer_type]

    return optimizer_class(parameters, **prepared_config)


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
