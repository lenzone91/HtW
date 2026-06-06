from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, WandbLogger

from ..Factory.base import BaseBuilder, BaseBuilderDispatcher
from .configs import (
    DEFAULT_CSV_LOGGER_CONFIG,
    DEFAULT_WANDB_LOGGER_CONFIG,
)


#############################
# Logger builder
#############################


class LoggerBuilder(BaseBuilder):
    """
    Build one Lightning logger from one logger config.
    """

    def __init__(
        self,
        logger_class,
        default_config: dict,
        strict: bool = True,
    ):
        super().__init__(
            default_config=default_config,
            strict=strict,
            check_default_keys=True,
        )
        self.logger_class = logger_class

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ):
        config = resolve_logger_paths(
            config=config,
            runtime_context=runtime_context,
        )

        return self.logger_class(**config)


##########################
# Registry
##########################


LOGGER_BUILDERS_REGISTRY = {
    "csv": LoggerBuilder(
        logger_class=CSVLogger,
        default_config=DEFAULT_CSV_LOGGER_CONFIG,
    ),

    "wandb": LoggerBuilder(
        logger_class=WandbLogger,
        default_config=DEFAULT_WANDB_LOGGER_CONFIG,
    ),
}


###########################
# Dispatcher
###########################


class LoggerBuilderDispatcher(BaseBuilderDispatcher):
    """
    Build Lightning loggers from named logger configs.
    """

    def __init__(
        self,
        builder_registry: dict = LOGGER_BUILDERS_REGISTRY,
        strict: bool = True,
    ):
        super().__init__(
            default_config={},
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
        )


##############################
# Path helpers
##############################


def resolve_logger_paths(
    config: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Resolve logger path fields when runtime_context is available.
    """
    config = dict(config)

    if "save_dir" in config:
        config["save_dir"] = resolve_logger_path(
            path=config["save_dir"],
            runtime_context=runtime_context,
            relative_to="run_dir",
        )

    if "dir" in config and config["dir"] is not None:
        config["dir"] = resolve_logger_path(
            path=config["dir"],
            runtime_context=runtime_context,
            relative_to="run_dir",
        )

    return config


def resolve_logger_path(
    path: str,
    runtime_context: dict | None = None,
    relative_to: str = "run_dir",
) -> str:
    """
    Resolve a logger path.

    Absolute paths are used as-is.
    Relative paths are resolved against runtime_context["paths"][relative_to]
    when runtime_context is available.
    """
    resolved_path = Path(path).expanduser()

    if resolved_path.is_absolute():
        return str(resolved_path.resolve())

    if runtime_context is None:
        return str(resolved_path)

    paths_context = runtime_context.get("paths", {})

    if relative_to not in paths_context:
        raise KeyError(
            f"Unknown logger path root: {relative_to}. "
            f"Available roots are: {sorted(paths_context.keys())}."
        )

    root = Path(paths_context[relative_to]).expanduser().resolve()
    return str((root / resolved_path).resolve())


##############################
# Wrapper
##############################


def build_loggers(
    logger_configs: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    """
    Build Lightning loggers from logger configs.

    Empty logger configs explicitly disable Lightning logging.
    """
    if logger_configs == {}:
        return False

    dispatcher = LoggerBuilderDispatcher(strict=strict)
    
    loggers = dispatcher(
        config=logger_configs,
        runtime_context=runtime_context,
    )

    return list(loggers.values())