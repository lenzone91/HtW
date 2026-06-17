import torch

from .registry import OPTIMIZER_REGISTRY


#############################################
# Optimizers (thin torch.optim wrappers)
#############################################

# Grouped per family (Decision 27): thin registered wrappers around torch.optim
# classes. Each object is preceded by its own default config.


DEFAULT_SGD_CONFIG = {
    "lr": 1e-3,
    "momentum": 0.0,
    "dampening": 0.0,
    "weight_decay": 0.0,
    "nesterov": False,
    "maximize": False,
    "foreach": None,
    "differentiable": False,
    "fused": None,
}


@OPTIMIZER_REGISTRY.register_class(
    name="sgd",
    default_config={"optimizer_type": "sgd", **DEFAULT_SGD_CONFIG},
    type_field="optimizer_type",
)
class SGD(torch.optim.SGD):
    pass


DEFAULT_ADAM_CONFIG = {
    "lr": 1e-3,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
}


@OPTIMIZER_REGISTRY.register_class(
    name="adam",
    default_config={"optimizer_type": "adam", **DEFAULT_ADAM_CONFIG},
    type_field="optimizer_type",
)
class Adam(torch.optim.Adam):
    pass


DEFAULT_ADAMW_CONFIG = {
    "lr": 1e-3,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 1e-2,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
}


@OPTIMIZER_REGISTRY.register_class(
    name="adamw",
    default_config={"optimizer_type": "adamw", **DEFAULT_ADAMW_CONFIG},
    type_field="optimizer_type",
)
class AdamW(torch.optim.AdamW):
    pass
