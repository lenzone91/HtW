from pathlib import Path
import json
import tomllib

import yaml

from .errors import ConfigError


#############################################
# Config conversion
#############################################

# This file handles config file <-> plain Python dict conversion.
#
# It does not:
#   - compose configs (Hydra does that, see compose.py);
#   - resolve interpolation (see resolve.py);
#   - validate subsystem-specific config content;
#   - build objects.


def load_config(path: str | Path) -> dict:
    """
    Load a config file as a plain Python dictionary.
    """
    config_path = _resolve_path(path)
    suffix = config_path.suffix.lower()

    if suffix not in LOAD_DISPATCHER:
        raise ConfigError(
            f"Unsupported config file extension: {suffix}. "
            f"Supported extensions are: {list(LOAD_DISPATCHER.keys())}."
        )

    config = LOAD_DISPATCHER[suffix](config_path)
    _check_config_is_dict(config=config, path=config_path)

    return config


def save_config(
    config: dict,
    path: str | Path,
) -> None:
    """
    Save a plain Python dictionary to a config file.
    """
    _check_input_config_is_dict(config)

    config_path = _resolve_path(path)
    suffix = config_path.suffix.lower()

    if suffix not in SAVE_DISPATCHER:
        raise ConfigError(
            f"Unsupported config file extension: {suffix}. "
            f"Supported extensions are: {list(SAVE_DISPATCHER.keys())}."
        )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    SAVE_DISPATCHER[suffix](config=config, path=config_path)


#############################################
# Path handling
#############################################


def _resolve_path(path: str | Path) -> Path:
    """
    Convert a user-provided path to an absolute resolved Path.
    """
    return Path(path).expanduser().resolve()


#############################################
# YAML conversion
#############################################


def _load_yaml(path: Path) -> dict:
    """
    Load a YAML config file.
    """
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _save_yaml(
    config: dict,
    path: Path,
) -> None:
    """
    Save a config dictionary as YAML.
    """
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


#############################################
# JSON conversion
#############################################


def _load_json(path: Path) -> dict:
    """
    Load a JSON config file.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(
    config: dict,
    path: Path,
) -> None:
    """
    Save a config dictionary as JSON.
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False,
        )


#############################################
# TOML conversion
#############################################


def _load_toml(path: Path) -> dict:
    """
    Load a TOML config file.
    """
    with path.open("rb") as file:
        return tomllib.load(file)


def _save_toml(
    config: dict,
    path: Path,
) -> None:
    """
    Save a config dictionary as TOML.
    """
    try:
        import tomli_w
    except ImportError as error:
        raise ImportError(
            "Saving TOML configs requires the optional dependency 'tomli-w'."
        ) from error

    with path.open("wb") as file:
        tomli_w.dump(config, file)


#############################################
# Validation helpers
#############################################


def _check_config_is_dict(
    config: object,
    path: Path,
) -> None:
    """
    Check that a loaded config is a dictionary.
    """
    if not isinstance(config, dict):
        raise ConfigError(
            f"Config loaded from {path} must be a dictionary. "
            f"Got {type(config).__name__}."
        )


def _check_input_config_is_dict(config: object) -> None:
    """
    Check that a config to save is a dictionary.
    """
    if not isinstance(config, dict):
        raise ConfigError(
            f"Config to save must be a dictionary. Got {type(config).__name__}."
        )


#############################################
# Dispatchers
#############################################


LOAD_DISPATCHER = {
    ".yaml": _load_yaml,
    ".yml": _load_yaml,
    ".json": _load_json,
    ".toml": _load_toml,
}


SAVE_DISPATCHER = {
    ".yaml": _save_yaml,
    ".yml": _save_yaml,
    ".json": _save_json,
    ".toml": _save_toml,
}
