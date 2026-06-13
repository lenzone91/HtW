from copy import deepcopy
from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, WandbLogger

from .configs import (
    DEFAULT_CSV_LOGGER_CONFIG,
    DEFAULT_WANDB_LOGGER_CONFIG,
    DEFAULT_WANDB_METRICS_CONFIG,
    DEFAULT_WANDB_WATCH_CONFIG,
)
from .wandb_metrics import WandbScalarMetricsCallback


LOGGER_REGISTRY = {
    "csv": (CSVLogger, DEFAULT_CSV_LOGGER_CONFIG),
    "wandb": (WandbLogger, DEFAULT_WANDB_LOGGER_CONFIG),
}


def build_loggers(
    logger_configs: dict,
    runtime_context: dict | None = None,
):
    """
    Build Lightning loggers from named logger configs.
    """
    if logger_configs == {}:
        return False

    if not isinstance(logger_configs, dict):
        raise TypeError(
            "Logger configs must be a dictionary, "
            f"got {type(logger_configs).__name__}."
        )

    loggers = []

    for logger_name, logger_config in deepcopy(logger_configs).items():
        loggers.append(
            build_logger(
                logger_name=logger_name,
                logger_config=logger_config,
                runtime_context=runtime_context,
            )
        )

    return loggers


def build_logger(
    logger_name: str,
    logger_config: dict,
    runtime_context: dict | None = None,
):
    if logger_name not in LOGGER_REGISTRY:
        raise KeyError(
            f"Unknown logger '{logger_name}'. "
            f"Available loggers are: {sorted(LOGGER_REGISTRY)}."
        )

    logger_class, default_config = LOGGER_REGISTRY[logger_name]
    prepared_config = prepare_logger_config(
        logger_name=logger_name,
        logger_config=logger_config,
        default_config=default_config,
        runtime_context=runtime_context,
    )
    prepared_config.pop("logger_type", None)
    prepared_config.pop("watch", None)
    prepared_config.pop("metrics", None)

    return logger_class(**prepared_config)


def prepare_logger_config(
    logger_name: str,
    logger_config: dict,
    default_config: dict,
    runtime_context: dict | None = None,
) -> dict:
    if not isinstance(logger_config, dict):
        raise TypeError(
            f"Logger config for '{logger_name}' must be a dictionary, "
            f"got {type(logger_config).__name__}."
        )

    prepared_config = deepcopy(default_config)
    user_config = deepcopy(logger_config)

    unknown_keys = set(user_config) - set(default_config)
    if unknown_keys:
        raise KeyError(
            f"Unknown logger config keys for '{logger_name}': "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(default_config)}."
        )

    prepared_config.update(user_config)
    prepared_config = prepare_wandb_watch_config(
        logger_name=logger_name,
        prepared_config=prepared_config,
    )
    prepared_config = prepare_wandb_metrics_config(
        logger_name=logger_name,
        prepared_config=prepared_config,
    )
    prepared_config = resolve_logger_paths(prepared_config, runtime_context)

    return prepared_config


def prepare_wandb_watch_config(
    logger_name: str,
    prepared_config: dict,
) -> dict:
    if logger_name != "wandb":
        return prepared_config

    watch_config = prepared_config.get("watch", {})
    if watch_config is None:
        watch_config = {"enabled": False}

    if not isinstance(watch_config, dict):
        raise TypeError(
            "Wandb watch config must be a dictionary, "
            f"got {type(watch_config).__name__}."
        )

    unknown_keys = set(watch_config) - set(DEFAULT_WANDB_WATCH_CONFIG)
    if unknown_keys:
        raise KeyError(
            "Unknown wandb watch config keys: "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(DEFAULT_WANDB_WATCH_CONFIG)}."
        )

    prepared_watch_config = deepcopy(DEFAULT_WANDB_WATCH_CONFIG)
    prepared_watch_config.update(watch_config)
    prepared_config["watch"] = prepared_watch_config

    return prepared_config


def prepare_wandb_metrics_config(
    logger_name: str,
    prepared_config: dict,
) -> dict:
    if logger_name != "wandb":
        return prepared_config

    metrics_config = prepared_config.get("metrics", {})
    if metrics_config is None:
        metrics_config = {"enabled": False}

    if not isinstance(metrics_config, dict):
        raise TypeError(
            "Wandb metrics config must be a dictionary, "
            f"got {type(metrics_config).__name__}."
        )

    unknown_keys = set(metrics_config) - set(DEFAULT_WANDB_METRICS_CONFIG)
    if unknown_keys:
        raise KeyError(
            "Unknown wandb metrics config keys: "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(DEFAULT_WANDB_METRICS_CONFIG)}."
        )

    prepared_metrics_config = deepcopy(DEFAULT_WANDB_METRICS_CONFIG)
    prepared_metrics_config.update(metrics_config)
    prepared_config["metrics"] = prepared_metrics_config

    return prepared_config


def build_logger_callbacks(logger_configs: dict) -> list:
    metrics_config = get_wandb_metrics_config(logger_configs)

    if not metrics_config["enabled"]:
        return []

    callback_config = deepcopy(metrics_config)
    callback_config.pop("enabled")
    return [WandbScalarMetricsCallback(**callback_config)]


def get_wandb_metrics_config(logger_configs: dict) -> dict:
    if not isinstance(logger_configs, dict):
        raise TypeError(
            "Logger configs must be a dictionary, "
            f"got {type(logger_configs).__name__}."
        )

    wandb_config = logger_configs.get("wandb")
    if wandb_config is None:
        return {**deepcopy(DEFAULT_WANDB_METRICS_CONFIG), "enabled": False}

    default_config = deepcopy(DEFAULT_WANDB_LOGGER_CONFIG)
    prepared_config = prepare_logger_config(
        logger_name="wandb",
        logger_config=wandb_config,
        default_config=default_config,
        runtime_context=None,
    )
    return prepared_config["metrics"]


def watch_module_with_wandb_loggers(
    module,
    loggers,
    logger_configs: dict,
) -> None:
    """
    Enable WandbLogger.watch from plain logger config.
    """
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


def get_wandb_watch_config(logger_configs: dict) -> dict:
    if not isinstance(logger_configs, dict):
        raise TypeError(
            "Logger configs must be a dictionary, "
            f"got {type(logger_configs).__name__}."
        )

    wandb_config = logger_configs.get("wandb")
    if wandb_config is None:
        return deepcopy(DEFAULT_WANDB_WATCH_CONFIG)

    default_config = deepcopy(DEFAULT_WANDB_LOGGER_CONFIG)
    prepared_config = prepare_logger_config(
        logger_name="wandb",
        logger_config=wandb_config,
        default_config=default_config,
        runtime_context=None,
    )
    return prepared_config["watch"]


def normalize_loggers(loggers) -> list:
    if loggers is False or loggers is None:
        return []

    if isinstance(loggers, list):
        return loggers

    return [loggers]


def resolve_logger_paths(
    config: dict,
    runtime_context: dict | None = None,
) -> dict:
    config = dict(config)

    if "save_dir" in config:
        config["save_dir"] = resolve_logger_path(
            path=config["save_dir"],
            runtime_context=runtime_context,
        )

    if "dir" in config and config["dir"] is not None:
        config["dir"] = resolve_logger_path(
            path=config["dir"],
            runtime_context=runtime_context,
        )

    return config


def resolve_logger_path(
    path: str,
    runtime_context: dict | None = None,
) -> str:
    resolved_path = Path(path).expanduser()

    if resolved_path.is_absolute():
        return str(resolved_path.resolve())

    if runtime_context is None:
        return str(resolved_path)

    run_dir = runtime_context.get("paths", {}).get("run_dir")
    if run_dir is None:
        raise KeyError("runtime_context['paths']['run_dir'] is required.")

    return str((Path(run_dir).expanduser().resolve() / resolved_path).resolve())
