DEFAULT_CSV_LOGGER_CONFIG = {
    "logger_type": "csv",
    "save_dir": "logs",
    "name": "csv",
    "version": None,
    "prefix": "",
    "flush_logs_every_n_steps": 100,
}


DEFAULT_WANDB_WATCH_CONFIG = {
    "enabled": False,
    "log": "all",
    "log_freq": 100,
    "log_graph": True,
}


DEFAULT_WANDB_LOGGER_CONFIG = {
    "logger_type": "wandb",
    "name": None,
    "save_dir": "logs",
    "version": None,
    "offline": False,
    "dir": None,
    "id": None,
    "anonymous": None,
    "project": "htw-ac-video-jepa",
    "log_model": False,
    "experiment": None,
    "prefix": "",
    "checkpoint_name": None,
    "add_file_policy": "mutable",
    "watch": DEFAULT_WANDB_WATCH_CONFIG,
}


DEFAULT_LOGGER_CONFIGS = {}
