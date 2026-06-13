DEFAULT_DISABLED_SCHEDULER_CONFIG = {
    "enabled": False,
}


DEFAULT_STEP_LR_CONFIG = {
    "enabled": True,
    "scheduler_type": "step_lr",
    "step_size": 1,
    "gamma": 0.1,
    "last_epoch": -1,
    "interval": "epoch",
    "frequency": 1,
    "monitor": None,
    "strict": True,
    "name": None,
}


DEFAULT_COSINE_ANNEALING_LR_CONFIG = {
    "enabled": True,
    "scheduler_type": "cosine_annealing_lr",
    "T_max": 10,
    "eta_min": 0.0,
    "last_epoch": -1,
    "interval": "epoch",
    "frequency": 1,
    "monitor": None,
    "strict": True,
    "name": None,
}


DEFAULT_REDUCE_LR_ON_PLATEAU_CONFIG = {
    "enabled": True,
    "scheduler_type": "reduce_lr_on_plateau",
    "mode": "min",
    "factor": 0.1,
    "patience": 10,
    "threshold": 1e-4,
    "threshold_mode": "rel",
    "cooldown": 0,
    "min_lr": 0.0,
    "eps": 1e-8,
    "interval": "epoch",
    "frequency": 1,
    "monitor": "val/loss",
    "strict": True,
    "name": None,
}


DEFAULT_SCHEDULER_CONFIG = DEFAULT_DISABLED_SCHEDULER_CONFIG
