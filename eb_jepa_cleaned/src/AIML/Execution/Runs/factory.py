from lightning import Trainer

from ...Data.DataModules.factory import build_datamodule
from ...Models.Loading.factory import load_module_if_needed
from ...Models.Modules.factory import build_lightning_module
from ...Training.Checkpoints.factory import build_checkpoint_callbacks
from ...Training.EarlyStoppings.factory import build_early_stopping_callbacks
from ...Training.Loggers.factory import build_loggers


#############################################
# Execution composition factory
#############################################

# Composes a run from a resolved plain-dict config: datamodule + module +
# Lightning Trainer (with loggers and callbacks). It receives runtime_context;
# it does not construct it (that is Setup's job, deferred — Decision 22).
#
# Expected config sections (all optional unless noted):
#   datamodule (required), module (required), loading, loggers, checkpoints,
#   early_stoppings, trainer.


def build_training_objects(
    config: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Build all objects required for trainer.fit.
    """
    datamodule = build_datamodule(
        datamodule_configs=config["datamodule"],
        runtime_context=runtime_context,
    )

    module = _build_module(config=config, runtime_context=runtime_context)

    trainer = build_training_trainer(config=config, runtime_context=runtime_context)

    return {"trainer": trainer, "module": module, "datamodule": datamodule}


def build_evaluation_objects(
    config: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Build all objects required for trainer.validate / trainer.test.
    """
    datamodule = build_datamodule(
        datamodule_configs=config["datamodule"],
        runtime_context=runtime_context,
    )

    module = _build_module(config=config, runtime_context=runtime_context)

    trainer = build_evaluation_trainer(config=config, runtime_context=runtime_context)

    return {"trainer": trainer, "module": module, "datamodule": datamodule}


def _build_module(
    config: dict,
    runtime_context: dict | None = None,
):
    """
    Build the Lightning module and optionally load module-level weights.

    Construction and loading are separate concerns: the module is built first,
    then weights are restored if a module loading config is enabled.
    """
    module = build_lightning_module(
        lightning_module_configs=config["module"],
        runtime_context=runtime_context,
    )

    return load_module_if_needed(
        module=module,
        loading_config=config.get("loading", {}).get("module"),
        runtime_context=runtime_context,
    )


#############################################
# Trainer construction
#############################################


def build_training_trainer(
    config: dict,
    runtime_context: dict | None = None,
) -> Trainer:
    """
    Build the Lightning Trainer used for training (with loggers and callbacks).
    """
    loggers = build_loggers(
        logger_configs=config.get("loggers", {}),
        runtime_context=runtime_context,
    )

    callbacks = build_training_callbacks(config=config, runtime_context=runtime_context)

    return build_trainer(
        trainer_config=config.get("trainer", {}),
        loggers=loggers,
        callbacks=callbacks,
    )


def build_evaluation_trainer(
    config: dict,
    runtime_context: dict | None = None,
) -> Trainer:
    """
    Build the Lightning Trainer used for evaluation (loggers, no callbacks).
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
    return Trainer(logger=loggers, callbacks=callbacks, **trainer_config)


#############################################
# Callbacks
#############################################


def build_training_callbacks(
    config: dict,
    runtime_context: dict | None = None,
) -> list:
    """
    Build and gather all training callbacks (checkpoints + early stoppings).
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
