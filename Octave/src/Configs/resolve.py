from pathlib import Path
from typing import Any

from .conversion import load_config, save_config


CONFIG_REFERENCE_EXTENSIONS = {".yaml", ".yml", ".json", ".toml"}
DEFAULT_FOLDER_ENTRYPOINT = "execution.yaml"

NON_CONFIG_REFERENCE_KEYS = {
    "path",
    "checkpoint_path",
    "save_dir",
    "dir",
    "dirpath",
    "idea_root",
    "run_root",
}


def resolve_config(config_path: str | Path) -> Path:
    """
    Resolve a config file or elementary config folder.

    If config_path is a file, return it unchanged.
    If config_path is a folder, resolve its execution.yaml recursively and
    save the resolved config as a sibling YAML file.
    """
    config_path = Path(config_path).expanduser().resolve()

    if config_path.is_file():
        _check_supported_config_file(config_path)
        return config_path

    if config_path.is_dir():
        return _resolve_config_folder(config_path)

    raise FileNotFoundError(f"Config path does not exist: {config_path}")


def _resolve_config_folder(config_dir: Path) -> Path:
    entrypoint_path = config_dir / DEFAULT_FOLDER_ENTRYPOINT

    if not entrypoint_path.is_file():
        raise FileNotFoundError(
            f"Expected config folder to contain {DEFAULT_FOLDER_ENTRYPOINT}: "
            f"{entrypoint_path}"
        )

    resolved_config = _load_and_resolve_config(
        config_path=entrypoint_path,
        config_dir=config_dir,
        resolution_stack=[],
    )

    resolved_path = config_dir.with_suffix(".yaml")

    save_config(
        config=resolved_config,
        path=resolved_path,
    )

    return resolved_path


def _load_and_resolve_config(
    config_path: Path,
    config_dir: Path,
    resolution_stack: list[Path],
) -> dict:
    config_path = config_path.expanduser().resolve()

    _check_supported_config_file(config_path)
    _check_config_file_exists(config_path)
    _check_no_reference_cycle(config_path, resolution_stack)

    config = load_config(config_path)

    return _resolve_references(
        obj=config,
        config_dir=config_dir,
        resolution_stack=resolution_stack + [config_path],
        parent_key=None,
    )


def _resolve_references(
    obj: Any,
    config_dir: Path,
    resolution_stack: list[Path],
    parent_key: str | None,
) -> Any:
    if isinstance(obj, dict):
        return {
            key: _resolve_references(
                obj=value,
                config_dir=config_dir,
                resolution_stack=resolution_stack,
                parent_key=str(key),
            )
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            _resolve_references(
                obj=value,
                config_dir=config_dir,
                resolution_stack=resolution_stack,
                parent_key=parent_key,
            )
            for value in obj
        ]

    if _is_config_reference(
        value=obj,
        parent_key=parent_key,
    ):
        referenced_path = (config_dir / obj).resolve()

        return _load_and_resolve_config(
            config_path=referenced_path,
            config_dir=config_dir,
            resolution_stack=resolution_stack,
        )

    return obj


def _is_config_reference(
    value: Any,
    parent_key: str | None,
) -> bool:
    """
    Return whether a value is a same-folder config reference.

    Some string values ending in .yaml/.json/.toml are runtime paths, not
    config references. These are excluded through their parent key.
    """
    if parent_key in NON_CONFIG_REFERENCE_KEYS:
        return False

    if not isinstance(value, str):
        return False

    reference_path = Path(value)

    return (
        len(reference_path.parts) == 1
        and reference_path.suffix.lower() in CONFIG_REFERENCE_EXTENSIONS
    )


def _check_supported_config_file(path: Path) -> None:
    suffix = path.suffix.lower()

    if suffix not in CONFIG_REFERENCE_EXTENSIONS:
        raise ValueError(
            f"Unsupported config file extension: {suffix}. "
            f"Supported extensions are: {sorted(CONFIG_REFERENCE_EXTENSIONS)}."
        )


def _check_config_file_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Referenced config file does not exist: {path}")


def _check_no_reference_cycle(
    config_path: Path,
    resolution_stack: list[Path],
) -> None:
    if config_path not in resolution_stack:
        return

    cycle = resolution_stack + [config_path]
    cycle_message = " -> ".join(path.name for path in cycle)

    raise ValueError(f"Config reference cycle detected: {cycle_message}")