DEFAULT_ADAMW_CONFIG = {
    "optimizer_type": "adamw",
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


DEFAULT_ADAM_CONFIG = {
    "optimizer_type": "adam",
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


DEFAULT_SGD_CONFIG = {
    "optimizer_type": "sgd",
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
