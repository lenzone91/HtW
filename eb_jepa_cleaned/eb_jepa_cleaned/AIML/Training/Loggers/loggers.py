from lightning.pytorch.loggers import CSVLogger as LightningCSVLogger
from lightning.pytorch.loggers import WandbLogger as LightningWandbLogger

from .registry import LOGGER_REGISTRY


#############################################
# Loggers (thin Lightning logger wrappers)
#############################################

# Each object is preceded by its own default config.


DEFAULT_CSV_LOGGER_CONFIG = {
    "save_dir": "logs",
    "name": "csv_reports",
    "version": None,
    "prefix": "",
    "flush_logs_every_n_steps": 100,
}


@LOGGER_REGISTRY.register_class(name="csv", default_config=DEFAULT_CSV_LOGGER_CONFIG)
class CSVLogger(LightningCSVLogger):
    pass


DEFAULT_WANDB_LOGGER_CONFIG = {
    "name": None,
    "save_dir": "logs",
    "version": None,
    "offline": False,
    "dir": None,
    "id": None,
    "anonymous": None,
    "project": None,
    "log_model": False,
    "experiment": None,
    "prefix": "",
    "checkpoint_name": None,
}


@LOGGER_REGISTRY.register_class(name="wandb", default_config=DEFAULT_WANDB_LOGGER_CONFIG)
class WandbLogger(LightningWandbLogger):
    pass
