from lightning import Trainer

from ..DataModules.factory import build_datamodule
from ..Modules.factory import build_lightning_module
from ..Loggers.factory import build_loggers
from ..Checkpoints.factory import build_checkpoint_callbacks
from ..EarlyStoppings.factory import build_early_stopping_callbacks
from ..Loading.factory import load_module_if_needed


#############################################
# Execution factory
#############################################


def build_training_objects(
    config: dict,
    runtime_context: dict,
) -> dict:
    """
    Build all objects required for trainer.fit.
    """
    datamodule = build_datamodule(
        datamodule_configs=config["datamodule"],
        runtime_context=runtime_context,
    )

    module = build_lightning_module(
        lightning_module_configs=config["module"],
        runtime_context=runtime_context,
        loading_config=config.get("loading"),
    )

    module = load_module_if_needed(
        module=module,
        loading_config=config.get("loading", {}).get("module"),
        runtime_context=runtime_context,
    )

    trainer = build_training_trainer(
        config=config,
        runtime_context=runtime_context,
    )

    return {
        "trainer": trainer,
        "module": module,
        "datamodule": datamodule,
    }


def build_evaluation_objects(
    config: dict,
    runtime_context: dict,
) -> dict:
    """
    Build all objects required for trainer.validate / trainer.test.
    """
    datamodule = build_datamodule(
        datamodule_configs=config["datamodule"],
        runtime_context=runtime_context,
    )

    module = build_lightning_module(
        lightning_module_configs=config["module"],
        runtime_context=runtime_context,
        loading_config=config.get("loading"),
    )

    module = load_module_if_needed(
        module=module,
        loading_config=config.get("loading", {}).get("module"),
        runtime_context=runtime_context,
    )

    trainer = build_evaluation_trainer(
        config=config,
        runtime_context=runtime_context,
    )

    return {
        "trainer": trainer,
        "module": module,
        "datamodule": datamodule,
    }


#############################################
# Trainer construction
#############################################


def build_training_trainer(
    config: dict,
    runtime_context: dict,
) -> Trainer:
    """
    Build the Lightning Trainer used for training.
    """
    loggers = build_loggers(
        logger_configs=config.get("loggers", {}),
        runtime_context=runtime_context,
    )

    callbacks = build_training_callbacks(
        config=config,
        runtime_context=runtime_context,
    )

    return build_trainer(
        trainer_config=config.get("trainer", {}),
        loggers=loggers,
        callbacks=callbacks,
    )


def build_evaluation_trainer(
    config: dict,
    runtime_context: dict,
) -> Trainer:
    """
    Build the Lightning Trainer used for evaluation.
    """
    loggers = build_loggers(
        logger_configs=config.get("loggers", {}),
        runtime_context=runtime_context,
    )

    return build_trainer(
        trainer_config=config.get("trainer", {}),
        loggers=loggers,
        callbacks=[],
    )


def build_trainer(
    trainer_config: dict,
    loggers,
    callbacks: list,
) -> Trainer:
    """
    Instantiate the Lightning Trainer from already-built objects.
    """
    return Trainer(
        logger=loggers,
        callbacks=callbacks,
        **trainer_config,
    )


#############################################
# Callbacks
#############################################


def build_training_callbacks(
    config: dict,
    runtime_context: dict,
) -> list:
    """
    Build and gather all training callbacks.
    """
    callbacks = []

    callbacks.extend(
        build_checkpoint_callbacks(
            checkpoint_configs=config.get("checkpoints", {}),
            runtime_context=runtime_context,
        )
    )

    callbacks.extend(
        build_early_stopping_callbacks(
            early_stopping_configs=config.get("early_stoppings", {}),
            runtime_context=runtime_context,
        )
    )

    return callbacks