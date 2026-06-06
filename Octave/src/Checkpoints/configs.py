####################################################################
# Save last
####################################################################

DEFAULT_LAST_CHECKPOINT_CONFIG = {
    "dirpath": None,
    "filename": "last",
    "every_n_epochs": 1,
}


####################################################################
# Save periodically
####################################################################

DEFAULT_PERIODIC_CHECKPOINT_CONFIG = {
    "dirpath": None,
    "filename": "epoch={epoch}-step={step}",
    "every_n_epochs": 1,
}


####################################################################
# Save best
####################################################################

DEFAULT_BEST_VALUE_CHECKPOINT_CONFIG = {
    "dirpath": None,
    "filename": "best-{epoch}-{step}-{val_loss:.4f}",
    "monitor": "val/loss",
    "mode": "min",
    "save_top_k": 1,
}


#####################################################################
#####################################################################
#####################################################################
# Overall config
#####################################################################
#####################################################################
#####################################################################


DEFAULT_CHECKPOINT_CONFIGS = {
    "last": {
        "checkpoint_type": "last",
        **DEFAULT_LAST_CHECKPOINT_CONFIG,
    },
    "best_val_loss": {
        "checkpoint_type": "best_value",
        **DEFAULT_BEST_VALUE_CHECKPOINT_CONFIG,
    },
}