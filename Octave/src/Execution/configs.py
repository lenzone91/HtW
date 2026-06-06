from ..Setup.configs import DEFAULT_SETUP_CONFIG
from ..DataModules.configs import DEFAULT_DATAMODULE_CONFIG
from ..Modules.configs import DEFAULT_LIGHTNING_MODULE_CONFIG
from ..Loggers.configs import DEFAULT_LOGGER_CONFIGS
from ..Checkpoints.configs import DEFAULT_CHECKPOINT_CONFIGS
from ..EarlyStoppings.configs import DEFAULT_EARLY_STOPPING_CONFIGS
from ..Loading.configs import DEFAULT_LOADING_CONFIG



#############################################
# Execution default configs
#############################################

# Execution/ only assembles subsystem defaults.
# It does not define model, data, metric, logger or callback defaults itself.


#############################################
# Trainer configs
#############################################


DEFAULT_TRAINER_CONFIG = {
    "max_epochs": 1,
    "accelerator": "auto",
    "devices": "auto",
    "log_every_n_steps": 10,
    "enable_checkpointing": True,
}


DEFAULT_EVALUATION_TRAINER_CONFIG = {
    "accelerator": "auto",
    "devices": "auto",
    "enable_checkpointing": False,
}

#############################################
# Resume config
#############################################

# Usefull if one wants to continue the training of a model

DEFAULT_RESUME_CONFIG = {
    "enabled": False,
    "checkpoint_path": None,
}


#############################################
# Training execution config
#############################################


DEFAULT_TRAINING_EXECUTION_CONFIG = {
    "setup": DEFAULT_SETUP_CONFIG,

    "datamodule": DEFAULT_DATAMODULE_CONFIG,

    "module": DEFAULT_LIGHTNING_MODULE_CONFIG,

    "loading": DEFAULT_LOADING_CONFIG,

    "resume": DEFAULT_RESUME_CONFIG,

    "loggers": DEFAULT_LOGGER_CONFIGS,

    "checkpoints": DEFAULT_CHECKPOINT_CONFIGS,

    "early_stoppings": DEFAULT_EARLY_STOPPING_CONFIGS,

    "trainer": DEFAULT_TRAINER_CONFIG,
}


#############################################
# Evaluation execution config
#############################################


DEFAULT_EVALUATION_EXECUTION_CONFIG = {
    "setup": DEFAULT_SETUP_CONFIG,

    "datamodule": DEFAULT_DATAMODULE_CONFIG,

    "module": DEFAULT_LIGHTNING_MODULE_CONFIG,

    "loading": DEFAULT_LOADING_CONFIG,

    # Keep logger config available, but disable Lightning logging by default
    # through DEFAULT_EVALUATION_TRAINER_CONFIG["logger"] = False.
    "loggers": DEFAULT_LOGGER_CONFIGS,

    "trainer": DEFAULT_EVALUATION_TRAINER_CONFIG,
}