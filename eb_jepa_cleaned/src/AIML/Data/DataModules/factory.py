from . import default  # noqa: F401  (registers DefaultDataModule)
from .registry import DATAMODULE_BUILDER


#############################################
# DataModule building wrappers
#############################################


def build_datamodules(
    datamodule_configs: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Build DataModules from a named DataModule config dictionary.
    """
    return DATAMODULE_BUILDER.build_named(
        configs=datamodule_configs,
        runtime_context=runtime_context,
    )


def build_datamodule(
    datamodule_configs: dict,
    runtime_context: dict | None = None,
) -> object:
    """
    Build exactly one DataModule from a named DataModule config dictionary.
    """
    datamodules = build_datamodules(
        datamodule_configs=datamodule_configs,
        runtime_context=runtime_context,
    )

    if len(datamodules) != 1:
        raise ValueError(
            "build_datamodule expects exactly one named DataModule config, "
            f"got {len(datamodules)}."
        )

    return next(iter(datamodules.values()))
