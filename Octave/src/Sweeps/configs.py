from ..Execution.configs import DEFAULT_TRAINING_EXECUTION_CONFIG


#############################################
# WandB sweep default config
#############################################

# WandB sweep config.
#
# This file only defines the sweep wrapper config.
# The actual training config is provided by Execution/.


DEFAULT_WANDB_SWEEP_CONFIG = {
    "base_config": DEFAULT_TRAINING_EXECUTION_CONFIG,

    "wandb_sweep": {
        "method": "grid",

        "metric": {
            "name": "val/loss",
            "goal": "minimize",
        },

        "parameters": {
            "trainer.max_epochs": {
                "values": [1],
            },
        },
    },

    "agent": {
        "project": None,
        "entity": None,
        "count": None,
    },
}


DEFAULT_EXISTING_WANDB_SWEEP_CONFIG = {
    "base_config": DEFAULT_TRAINING_EXECUTION_CONFIG,

    "sweep_id": None,

    "agent": {
        "project": None,
        "entity": None,
        "count": None,
    },
}