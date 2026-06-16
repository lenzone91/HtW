from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, WandbLogger

from .configs import DEFAULT_CSV_LOGGER_CONFIG, DEFAULT_WANDB_LOGGER_CONFIG
from .registry import LOGGER_REGISTRY
from ..Workflow.Factory.registry import FieldResolution


def resolve_logger_path(
    path: str | None,
    runtime_context: dict | None = None,
) -> str | None:
    if path is None:
        return None

    resolved_path = Path(path).expanduser()

    if resolved_path.is_absolute():
        return str(resolved_path.resolve())

    if runtime_context is None:
        return str(resolved_path)

    run_dir = runtime_context.get("paths", {}).get("run_dir")

    if run_dir is None:
        raise KeyError("runtime_context['paths']['run_dir'] is required.")

    return str((Path(run_dir).expanduser().resolve() / resolved_path).resolve())


def resolve_save_dir(
    config: dict,
    runtime_context: dict | None = None,
    **kwargs,
) -> str | None:
    return resolve_logger_path(
        path=config.get("save_dir"),
        runtime_context=runtime_context,
    )


def resolve_wandb_dir(
    config: dict,
    runtime_context: dict | None = None,
    **kwargs,
) -> str | None:
    return resolve_logger_path(
        path=config.get("dir"),
        runtime_context=runtime_context,
    )


SAVE_DIR_FIELD = FieldResolution(
    target_key="save_dir",
    resolver=resolve_save_dir,
)

WANDB_DIR_FIELD = FieldResolution(
    target_key="dir",
    resolver=resolve_wandb_dir,
)


@LOGGER_REGISTRY.register_class(
    name="csv",
    default_config=DEFAULT_CSV_LOGGER_CONFIG,
    type_field="logger_type",
    field_resolutions=(SAVE_DIR_FIELD,),
)
class CSVLogger_Wrapper(CSVLogger):
    def __init__(
        self,
        logger_type: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)


@LOGGER_REGISTRY.register_class(
    name="wandb",
    default_config=DEFAULT_WANDB_LOGGER_CONFIG,
    type_field="logger_type",
    field_resolutions=(SAVE_DIR_FIELD, WANDB_DIR_FIELD),
)
class WandbLogger_Wrapper(WandbLogger):
    def __init__(
        self,
        logger_type: str | None = None,
        watch: dict | None = None,
        metrics: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
