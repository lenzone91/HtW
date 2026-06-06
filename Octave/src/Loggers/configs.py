#############################
# CSV Logger default config
#############################

DEFAULT_CSV_LOGGER_CONFIG = {
    "save_dir": "logs",
    "name": "csv_reports",
    "version": None,
    "prefix": "",
    "flush_logs_every_n_steps": 100,
}


#############################
# WandB Logger default config
#############################

DEFAULT_WANDB_LOGGER_CONFIG = {
    "name": None,
    "save_dir": "logs",
    "version": None,
    "offline": False,
    "dir": None,
    "id": None,
    "anonymous": None,
    "project": "tse",
    "log_model": False,
    "experiment": None,
    "prefix": "",
    "checkpoint_name": None,
    "add_file_policy": "mutable",

    
}


#############################
# Overall logger configs
#############################

DEFAULT_LOGGER_CONFIGS = {
    "csv": DEFAULT_CSV_LOGGER_CONFIG,
}


DEFAULT_LOGGER_CONFIGS_WITH_WANDB = {
    "csv": DEFAULT_CSV_LOGGER_CONFIG,
    "wandb": DEFAULT_WANDB_LOGGER_CONFIG,
}