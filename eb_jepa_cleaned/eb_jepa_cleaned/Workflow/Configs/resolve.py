from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from .errors import ConfigError


#############################################
# Config resolution: the plain-dict boundary
#############################################

# This file owns the hard boundary between the Hydra/OmegaConf config layer and
# the rest of the project.
#
# After this point, project logic (Factory, AIML, Audio, SDE, Projects) sees
# only plain Python dicts/lists/scalars. No DictConfig or ListConfig may cross
# this boundary.
#
# It does not:
#   - compose configs (see compose.py);
#   - load/save config files (see conversion.py);
#   - validate subsystem-specific config content;
#   - build objects.


def resolve_to_plain_dict(config: DictConfig) -> dict:
    """
    Resolve all interpolations and convert a composed config to a plain dict.

    This is the single sanctioned way to leave the OmegaConf config layer.
    Interpolations are resolved eagerly; missing mandatory values (`???`) raise.
    """
    if not isinstance(config, DictConfig):
        raise ConfigError(
            "resolve_to_plain_dict expects a DictConfig at the config boundary, "
            f"got {type(config)}."
        )

    try:
        container = OmegaConf.to_container(
            config,
            resolve=True,
            throw_on_missing=True,
        )
    except OmegaConfBaseException as error:
        raise ConfigError(
            f"Failed to resolve config to a plain dict: {error}"
        ) from error

    # to_container on a DictConfig always yields a dict, but we assert the whole
    # tree is OmegaConf-free as a strict boundary post-condition.
    check_plain_config(container)

    return container


def check_plain_config(obj: Any, path: str = "<root>") -> None:
    """
    Assert that a value contains no OmegaConf objects anywhere in its tree.

    Raises ConfigError on the first OmegaConf object found. Used as a strict
    boundary post-condition and as a guard in tests.
    """
    if isinstance(obj, (DictConfig, ListConfig)):
        raise ConfigError(
            f"OmegaConf object leaked past the plain-dict boundary at {path}: "
            f"{type(obj)}."
        )

    if isinstance(obj, dict):
        for key, value in obj.items():
            check_plain_config(value, f"{path}.{key}")
        return

    if isinstance(obj, list):
        for index, value in enumerate(obj):
            check_plain_config(value, f"{path}[{index}]")
