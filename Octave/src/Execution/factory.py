from copy import deepcopy

from . import trainers  # noqa: F401
from .configs import DEFAULT_LIGHTNING_TRAINER_CONFIG
from .registry import TRAINER_REGISTRY
from ..Workflow.Factory.builder import RegistryBuilder

from lightning.pytorch import Trainer

from ..Data.DataModules.factory import build_ac_video_jepa_datamodule
from ..Loggers.factory import (
    build_logger_callbacks,
    build_loggers,
    watch_module_with_wandb_loggers,
)
from ..Models.Loading.factory import load_module_if_needed
from ..Models.Modules.factory import build_ac_video_jepa_module
from ..Training.Checkpoints.factory import build_checkpoint_callbacks


def build_training_objects(
    config: dict,
    runtime_context: dict,
) -> dict:
    datamodule = build_ac_video_jepa_datamodule(
        config=config["datamodule"],
        runtime_context=runtime_context,
    )
    module = build_ac_video_jepa_module(
        config=config["module"],
        runtime_context=runtime_context,
    )
    trainer = build_trainer(
        trainer_config=config.get("trainer", {}),
        logger_configs=config.get("loggers", {}),
        checkpoint_configs=config.get("checkpoints", {}),
        runtime_context=runtime_context,
    )
    watch_module_with_wandb_loggers(
        module=module,
        loggers=get_trainer_loggers(trainer),
        logger_configs=config.get("loggers", {}),
    )

    return {
        "trainer": trainer,
        "module": module,
        "datamodule": datamodule,
    }


def build_validation_objects(
    config: dict,
    runtime_context: dict,
) -> dict:
    objects = build_training_objects(
        config=config,
        runtime_context=runtime_context,
    )
    objects["module"] = load_module_if_needed(
        module=objects["module"],
        loading_config=config.get("loading", {}).get("module"),
        runtime_context=runtime_context,
    )

    return objects


def build_trainer(
    trainer_config: dict,
    logger_configs: dict,
    checkpoint_configs: dict,
    runtime_context: dict,
):
    user_trainer_config = deepcopy(trainer_config)

    if "logger" in user_trainer_config:
        raise KeyError(
            "Trainer config must not define 'logger'. "
            "Use the execution-level 'loggers' config instead."
        )

    if "callbacks" in user_trainer_config:
        raise KeyError(
            "Trainer config must not define 'callbacks'. "
            "Use the execution-level 'checkpoints' and logger callback configs instead."
        )

    trainer_config = {
        **deepcopy(DEFAULT_LIGHTNING_TRAINER_CONFIG),
        **user_trainer_config,
    }

    loggers = build_loggers(
        logger_configs=logger_configs,
        runtime_context=runtime_context,
    )

    callbacks = build_checkpoint_callbacks(
        checkpoint_configs=checkpoint_configs,
        runtime_context=runtime_context,
    )
    callbacks.extend(build_logger_callbacks(logger_configs=logger_configs))

    trainer_config["logger"] = loggers
    trainer_config["callbacks"] = callbacks

    builder = RegistryBuilder(
        registry=TRAINER_REGISTRY,
        strict=True,
        type_field="trainer_type",
    )

    return builder.build_one(
        config=trainer_config,
        runtime_context=runtime_context,
    )


def get_trainer_loggers(trainer) -> list:
    loggers = getattr(trainer, "loggers", None)
    if loggers is not None:
        return loggers

    logger = getattr(trainer, "logger", None)
    if logger is None or logger is False:
        return []

    if isinstance(logger, list):
        return logger

    return [logger]
