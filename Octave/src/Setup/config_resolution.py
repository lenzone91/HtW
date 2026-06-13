from copy import deepcopy
from pathlib import Path
import json

import yaml


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    suffix = config_path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    elif suffix == ".json":
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    else:
        raise ValueError(
            f"Unsupported config extension '{suffix}'. "
            "Supported extensions are: ['.yaml', '.yml', '.json']."
        )

    if not isinstance(config, dict):
        raise TypeError(
            f"Config loaded from {config_path} must be a dictionary, "
            f"got {type(config).__name__}."
        )

    return config


def merge_configs(
    base_config: dict,
    override_config: dict | None = None,
    strict: bool = True,
) -> dict:
    if not isinstance(base_config, dict):
        raise TypeError(
            f"base_config must be a dictionary, got {type(base_config).__name__}."
        )

    if override_config is None:
        return deepcopy(base_config)

    if not isinstance(override_config, dict):
        raise TypeError(
            "override_config must be a dictionary, "
            f"got {type(override_config).__name__}."
        )

    merged = deepcopy(base_config)
    merge_into(merged, override_config, strict=strict, key_path="")
    return merged


def merge_into(
    base_config: dict,
    override_config: dict,
    strict: bool,
    key_path: str,
) -> None:
    for key, override_value in override_config.items():
        current_key_path = str(key) if key_path == "" else f"{key_path}.{key}"

        if key not in base_config:
            if strict:
                raise KeyError(f"Unknown config key: {current_key_path}.")

            base_config[key] = deepcopy(override_value)
            continue

        base_value = base_config[key]

        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merge_into(
                base_config=base_value,
                override_config=override_value,
                strict=strict,
                key_path=current_key_path,
            )
            continue

        base_config[key] = deepcopy(override_value)
