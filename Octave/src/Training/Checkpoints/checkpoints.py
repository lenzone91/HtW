from pathlib import Path

from .base import NamedModelCheckpoint
from .configs import (
    DEFAULT_BEST_VALUE_CHECKPOINT_CONFIG,
    DEFAULT_LAST_CHECKPOINT_CONFIG,
    DEFAULT_PERIODIC_CHECKPOINT_CONFIG,
)
from .registry import CHECKPOINT_REGISTRY
from ...Workflow.Factory.registry import FieldResolution


def resolve_checkpoint_dirpath(
    config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
    **kwargs,
) -> str | None:
    dirpath = config.get("dirpath")

    if dirpath is None:
        return get_default_checkpoint_dir(runtime_context)

    path = Path(dirpath).expanduser()

    if path.is_absolute():
        return str(path.resolve())

    root = get_path_root(runtime_context, root_key="run_dir")
    return str((root / path).resolve())


def get_default_checkpoint_dir(runtime_context: dict | None = None) -> str | None:
    if runtime_context is None:
        return None

    checkpoints_dir = runtime_context.get("paths", {}).get("checkpoints_dir")

    if checkpoints_dir is None:
        raise KeyError("runtime_context['paths']['checkpoints_dir'] is required.")

    return checkpoints_dir


def get_path_root(runtime_context: dict | None, root_key: str) -> Path:
    if runtime_context is None:
        raise ValueError("runtime_context is required to resolve relative dirpath.")

    root = runtime_context.get("paths", {}).get(root_key)

    if root is None:
        raise KeyError(f"runtime_context['paths']['{root_key}'] is required.")

    return Path(root).expanduser().resolve()


CHECKPOINT_DIRPATH_FIELD = FieldResolution(
    target_key="dirpath",
    resolver=resolve_checkpoint_dirpath,
)


@CHECKPOINT_REGISTRY.register_class(
    name="last",
    default_config=DEFAULT_LAST_CHECKPOINT_CONFIG,
    type_field="checkpoint_type",
    field_resolutions=(CHECKPOINT_DIRPATH_FIELD,),
)
class LastModelCheckpoint(NamedModelCheckpoint):
    def __init__(self, checkpoint_name: str, **kwargs) -> None:
        super().__init__(
            checkpoint_name=checkpoint_name,
            save_last=True,
            save_top_k=0,
            **kwargs,
        )


@CHECKPOINT_REGISTRY.register_class(
    name="periodic",
    default_config=DEFAULT_PERIODIC_CHECKPOINT_CONFIG,
    type_field="checkpoint_type",
    field_resolutions=(CHECKPOINT_DIRPATH_FIELD,),
)
class PeriodicModelCheckpoint(NamedModelCheckpoint):
    def __init__(self, checkpoint_name: str, **kwargs) -> None:
        super().__init__(
            checkpoint_name=checkpoint_name,
            save_last=False,
            save_top_k=-1,
            **kwargs,
        )


@CHECKPOINT_REGISTRY.register_class(
    name="best_value",
    default_config=DEFAULT_BEST_VALUE_CHECKPOINT_CONFIG,
    type_field="checkpoint_type",
    field_resolutions=(CHECKPOINT_DIRPATH_FIELD,),
)
class BestValueModelCheckpoint(NamedModelCheckpoint):
    def __init__(self, checkpoint_name: str, **kwargs) -> None:
        super().__init__(
            checkpoint_name=checkpoint_name,
            save_last=False,
            **kwargs,
        )