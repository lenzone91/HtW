from pathlib import Path

from lightning.pytorch import LightningModule

from .module_loading import load_module_from_lightning_checkpoint


def load_module_if_needed(
    module: LightningModule,
    loading_config: dict | None,
    runtime_context: dict | None = None,
) -> LightningModule:
    if not is_loading_enabled(loading_config):
        return module

    check_loading_type(
        loading_config=loading_config,
        expected_type="lightning_module",
    )

    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=loading_config["checkpoint_path"],
        runtime_context=runtime_context,
        relative_to=loading_config.get("relative_to", "run_dir"),
    )

    return load_module_from_lightning_checkpoint(
        module=module,
        checkpoint_path=checkpoint_path,
        strict=loading_config.get("strict", True),
        map_location=loading_config.get("map_location", "cpu"),
        state_dict_key=loading_config.get("state_dict_key", "state_dict"),
    )


def is_loading_enabled(loading_config: dict | None) -> bool:
    if loading_config is None:
        return False

    return loading_config.get("enabled", False)


def check_loading_type(
    loading_config: dict,
    expected_type: str,
) -> None:
    loading_type = loading_config.get("type")

    if loading_type != expected_type:
        raise ValueError(
            f"Invalid loading type: {loading_type}. Expected {expected_type}."
        )


def resolve_checkpoint_path(
    checkpoint_path: str | None,
    runtime_context: dict | None = None,
    relative_to: str = "run_dir",
) -> str:
    if checkpoint_path is None:
        raise ValueError("Loading is enabled, but checkpoint_path is None.")

    path = Path(checkpoint_path).expanduser()

    if path.is_absolute():
        return str(path.resolve())

    root = get_path_root(
        runtime_context=runtime_context,
        relative_to=relative_to,
    )

    return str((root / path).resolve())


def get_path_root(
    runtime_context: dict | None,
    relative_to: str,
) -> Path:
    if runtime_context is None:
        raise ValueError(
            "Cannot resolve relative checkpoint path without runtime_context."
        )

    paths_context = runtime_context.get("paths", {})

    if relative_to not in paths_context:
        raise KeyError(
            f"Unknown checkpoint path root: {relative_to}. "
            f"Available roots are: {sorted(paths_context.keys())}."
        )

    return Path(paths_context[relative_to]).expanduser().resolve()
