#############################################################
# Stagnation
#############################################################

DEFAULT_BEST_VALUE_STAGNATION_EARLY_STOPPING_CONFIG = {
    "monitor": "val/loss",
    "mode": "min",
    "patience": 10,
    "min_delta": 0.0,
}

#############################################################
# Threshold
#############################################################

DEFAULT_THRESHOLD_EARLY_STOPPING_CONFIG = {
    "monitor": "val/loss",
    "mode": "min",
    "stopping_threshold": 0.01,
}

################################################################
# Training Divergence
################################################################

DEFAULT_DIVERGENCE_EARLY_STOPPING_CONFIG = {
    "monitor": "val/loss",
    "mode": "min",
    "divergence_threshold": 1e3,
    "patience": 0,
}


################################################################
# Training Divergence (check for NaN)
################################################################


DEFAULT_FINITE_VALUE_EARLY_STOPPING_CONFIG = {
    "monitor": "val/loss",
    "mode": "min",
}


#############################################################
# Overall early stopping configs
#############################################################

DEFAULT_EARLY_STOPPING_CONFIGS = {
    "best_val_loss_stagnation": {
        "early_stopping_type": "best_value_stagnation",
        **DEFAULT_BEST_VALUE_STAGNATION_EARLY_STOPPING_CONFIG,
    },

    "finite_val_loss": {
        "early_stopping_type": "finite_value",
        **DEFAULT_FINITE_VALUE_EARLY_STOPPING_CONFIG,
    },
}