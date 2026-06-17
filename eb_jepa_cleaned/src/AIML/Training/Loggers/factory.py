from . import loggers  # noqa: F401  (registers CSV/Wandb loggers)
from .registry import LOGGER_BUILDER


#############################################
# Logger building wrapper
#############################################

# Logger save_dir is resolved against the runtime-context run paths by a field
# resolution on each logger entry (see loggers.py): when runtime_context carries
# paths.logs_dir, save_dir points at the run's logs directory; otherwise the
# configured save_dir is used as-is.


def build_loggers(
    logger_configs: dict,
    runtime_context: dict | None = None,
):
    """
    Build Lightning loggers from logger configs.

    An empty logger config explicitly disables Lightning logging (returns False,
    the Lightning sentinel). Otherwise returns a list of logger objects.
    """
    if logger_configs == {}:
        return False

    loggers = LOGGER_BUILDER.build_named(
        configs=logger_configs,
        runtime_context=runtime_context,
    )

    return list(loggers.values())
