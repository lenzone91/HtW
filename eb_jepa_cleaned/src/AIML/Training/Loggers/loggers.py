from lightning.pytorch.loggers import CSVLogger as LightningCSVLogger
from lightning.pytorch.loggers import WandbLogger as LightningWandbLogger

from ....Workflow.Factory.registry import FieldResolution
from .registry import LOGGER_REGISTRY


#############################################
# save_dir resolution (runtime-context paths)
#############################################

# Resolve a logger's save_dir to the run's logs directory when a runtime context
# provides one (Decision 14); otherwise keep the configured save_dir. Defensive:
# a None runtime_context (e.g. unit tests) leaves the config value untouched.


def resolve_logger_save_dir(config: dict, runtime_context: dict | None = None, **kwargs):
    if runtime_context:
        logs_dir = runtime_context.get("paths", {}).get("logs_dir")
        if logs_dir:
            return logs_dir
    return config.get("save_dir", "logs")


LOGGER_SAVE_DIR_FIELD = FieldResolution(
    target_key="save_dir",
    resolver=resolve_logger_save_dir,
)


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


@LOGGER_REGISTRY.register_class(
    name="csv",
    default_config=DEFAULT_CSV_LOGGER_CONFIG,
    field_resolutions=(LOGGER_SAVE_DIR_FIELD,),
)
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


@LOGGER_REGISTRY.register_class(
    name="wandb",
    default_config=DEFAULT_WANDB_LOGGER_CONFIG,
    field_resolutions=(LOGGER_SAVE_DIR_FIELD,),
)
class WandbLogger(LightningWandbLogger):
    pass
