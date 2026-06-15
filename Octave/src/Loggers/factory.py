from copy import deepcopy

from lightning.pytorch.loggers import WandbLogger

from . import loggers  # noqa: F401
from .configs import (
    DEFAULT_LOGGER_CONFIGS,
    DEFAULT_WANDB_LOGGER_CONFIG,
    DEFAULT_WANDB_METRICS_CONFIG,
    DEFAULT_WANDB_WATCH_CONFIG,
)
from .registry import LOGGER_REGISTRY
from .wandb_metrics import WandbScalarMetricsCallback
from ..Workflow.Factory.builder import RegistryBuilder


def build_loggers(
    logger_configs: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    logger_configs = logger_configs or DEFAULT_LOGGER_CONFIGS

    if logger_configs == {}:
        return False

    if not isinstance(logger_configs, dict):
        raise TypeError(
            "Logger configs must be a dictionary, "
            f"got {type(logger_configs).__name__}."
        )

    validate_logger_nested_configs(logger_configs=logger_configs)

    built_loggers = []

    for logger_name, logger_config in deepcopy(logger_configs).items():
        built_loggers.append(
            build_logger(
                logger_name=logger_name,
                logger_config=logger_config,
                runtime_context=runtime_context,
                strict=strict,
            )
        )

    return built_loggers


def validate_logger_nested_configs(logger_configs: dict) -> None:
    get_wandb_watch_config(logger_configs)
    get_wandb_metrics_config(logger_configs)


def build_logger(
    logger_name: str,
    logger_config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    builder = RegistryBuilder(
        registry=LOGGER_REGISTRY,
        strict=strict,
        type_field="logger_type",
    )

    return builder.build_one(
        config=logger_config,
        runtime_context=runtime_context,
        name=logger_name,
    )


def build_logger_callbacks(logger_configs: dict | None = None) -> list:
    logger_configs = logger_configs or DEFAULT_LOGGER_CONFIGS

    metrics_config = get_wandb_metrics_config(logger_configs)

    if not metrics_config["enabled"]:
        return []

    callback_config = deepcopy(metrics_config)
    callback_config.pop("enabled")

    return [WandbScalarMetricsCallback(**callback_config)]


def get_wandb_metrics_config(logger_configs: dict) -> dict:
    return get_wandb_nested_config(
        logger_configs=logger_configs,
        nested_key="metrics",
        default_nested_config=DEFAULT_WANDB_METRICS_CONFIG,
    )


def get_wandb_watch_config(logger_configs: dict) -> dict:
    return get_wandb_nested_config(
        logger_configs=logger_configs,
        nested_key="watch",
        default_nested_config=DEFAULT_WANDB_WATCH_CONFIG,
    )


def get_wandb_nested_config(
    logger_configs: dict,
    nested_key: str,
    default_nested_config: dict,
) -> dict:
    if not isinstance(logger_configs, dict):
        raise TypeError(
            "Logger configs must be a dictionary, "
            f"got {type(logger_configs).__name__}."
        )

    wandb_config = logger_configs.get("wandb")

    if wandb_config is None:
        return {
            **deepcopy(default_nested_config),
            "enabled": False,
        }

    if not isinstance(wandb_config, dict):
        raise TypeError(
            "Wandb logger config must be a dictionary, "
            f"got {type(wandb_config).__name__}."
        )

    nested_config = wandb_config.get(nested_key, {})

    if nested_config is None:
        nested_config = {"enabled": False}

    if not isinstance(nested_config, dict):
        raise TypeError(
            f"Wandb {nested_key} config must be a dictionary, "
            f"got {type(nested_config).__name__}."
        )

    unknown_keys = set(nested_config) - set(default_nested_config)

    if unknown_keys:
        raise KeyError(
            f"Unknown wandb {nested_key} config keys: "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(default_nested_config)}."
        )

    prepared_config = deepcopy(default_nested_config)
    prepared_config.update(nested_config)

    return prepared_config


def watch_module_with_wandb_loggers(
    module,
    loggers,
    logger_configs: dict,
) -> None:
    watch_config = get_wandb_watch_config(logger_configs)

    if not watch_config["enabled"]:
        return

    for logger in normalize_loggers(loggers):
        if isinstance(logger, WandbLogger):
            logger.watch(
                model=module,
                log=watch_config["log"],
                log_freq=watch_config["log_freq"],
                log_graph=watch_config["log_graph"],
            )


def normalize_loggers(loggers) -> list:
    if loggers is False or loggers is None:
        return []

    if isinstance(loggers, list):
        return loggers

    return [loggers]
