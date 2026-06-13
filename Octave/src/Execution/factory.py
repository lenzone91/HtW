from copy import deepcopy

from lightning.pytorch import Trainer

from ..Data.DataModules.factory import build_ac_video_jepa_datamodule
from ..Loggers.factory import build_loggers, watch_module_with_wandb_loggers
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
        loggers=trainer.logger,
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
) -> Trainer:
    trainer_kwargs = deepcopy(trainer_config)

    if "logger" in trainer_kwargs:
        raise KeyError(
            "Trainer config must not define 'logger'. "
            "Use the execution-level 'loggers' config instead."
        )

    loggers = build_loggers(
        logger_configs=logger_configs,
        runtime_context=runtime_context,
    )
    callbacks = build_checkpoint_callbacks(
        checkpoint_configs=checkpoint_configs,
        runtime_context=runtime_context,
    )

    return Trainer(
        logger=loggers,
        callbacks=callbacks,
        **trainer_kwargs,
    )
