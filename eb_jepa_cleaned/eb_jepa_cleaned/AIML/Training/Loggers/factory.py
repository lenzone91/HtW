from . import loggers  # noqa: F401  (registers CSV/Wandb loggers)
from .registry import LOGGER_BUILDER


#############################################
# Logger building wrapper
#############################################

# Path resolution of logger save_dir/dir against the runtime-context paths is
# deferred to the Setup migration (Decision 22); for now paths are taken from the
# config as-is. Lightning resolves relative paths against the working directory.


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
