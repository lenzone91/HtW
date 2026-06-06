from copy import deepcopy


#############################################
# Config merging
#############################################

# This file handles recursive dictionary merging.
#
# It does not:
#   - load config files;
#   - save config files;
#   - resolve runtime paths;
#   - parse CLI overrides;
#   - validate subsystem-specific config content;
#   - build objects.


def merge_configs(
    base_config: dict,
    override_config: dict | None = None,
    strict: bool = True,
) -> dict:
    """
    Merge a nested override config into a base config.

    Nested dictionaries are merged recursively.
    Non-dictionary override values replace base values.

    The input dictionaries are not modified.
    """
    _check_config_is_dict(config=base_config, name="base_config")

    if override_config is None:
        return deepcopy(base_config)

    _check_config_is_dict(config=override_config, name="override_config")

    merged_config = deepcopy(base_config)

    _merge_into(
        base_config=merged_config,
        override_config=override_config,
        strict=strict,
        current_path="",
    )

    return merged_config


def _merge_into(
    base_config: dict,
    override_config: dict,
    strict: bool,
    current_path: str,
) -> None:
    """
    Recursively merge override_config into base_config in-place.
    """
    for key, override_value in override_config.items():
        key_path = _join_key_path(
            current_path=current_path,
            key=key,
        )

        if key not in base_config:
            _handle_unknown_key(
                key_path=key_path,
                strict=strict,
            )

            base_config[key] = deepcopy(override_value)
            continue

        base_value = base_config[key]

        if _should_merge_recursively(
            base_value=base_value,
            override_value=override_value,
        ):
            _merge_into(
                base_config=base_value,
                override_config=override_value,
                strict=strict,
                current_path=key_path,
            )
            continue

        base_config[key] = deepcopy(override_value)


#############################################
# Merge policy
#############################################


def _should_merge_recursively(
    base_value: object,
    override_value: object,
) -> bool:
    """
    Return whether two values should be merged recursively.
    """
    return isinstance(base_value, dict) and isinstance(override_value, dict)


#############################################
# Key handling
#############################################


def _handle_unknown_key(
    key_path: str,
    strict: bool,
) -> None:
    """
    Handle an override key missing from the base config.
    """
    if strict:
        raise KeyError(
            f"Unknown config key: {key_path}."
        )


def _join_key_path(
    current_path: str,
    key: object,
) -> str:
    """
    Join nested config keys for readable error messages.
    """
    if current_path == "":
        return str(key)

    return f"{current_path}.{key}"


#############################################
# Validation helpers
#############################################


def _check_config_is_dict(
    config: object,
    name: str,
) -> None:
    """
    Check that a config object is a dictionary.
    """
    if not isinstance(config, dict):
        raise TypeError(
            f"{name} must be a dictionary. "
            f"Got {type(config).__name__}."
        )