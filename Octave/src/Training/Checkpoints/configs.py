DEFAULT_LAST_CHECKPOINT_CONFIG = {
    "checkpoint_type": "last",
    "dirpath": None,
    "filename": "last",
    "every_n_epochs": 1,
}


DEFAULT_PERIODIC_CHECKPOINT_CONFIG = {
    "checkpoint_type": "periodic",
    "dirpath": None,
    "filename": "epoch={epoch}-step={step}",
    "every_n_epochs": 1,
}


DEFAULT_BEST_VALUE_CHECKPOINT_CONFIG = {
    "checkpoint_type": "best_value",
    "dirpath": None,
    "filename": "best-{epoch}-{step}",
    "monitor": "val/loss",
    "mode": "min",
    "save_top_k": 1,
}


DEFAULT_CHECKPOINT_CONFIGS = {
    "last": DEFAULT_LAST_CHECKPOINT_CONFIG,
}
