from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import HydraException
from omegaconf import DictConfig

from .errors import ConfigError
from .resolve import resolve_to_plain_dict


#############################################
# Config composition (Hydra)
#############################################

# This file owns Hydra-based config composition.
#
# Hydra is used for composition only: config groups, defaults lists,
# command-line-style overrides, and interpolation. It is NOT used to instantiate
# project objects (no hydra.utils.instantiate).
#
# It does not:
#   - resolve interpolation into plain values (see resolve.py);
#   - validate subsystem-specific config content;
#   - build objects (see Workflow/Factory).


def compose_config(
    config_dir: str | Path,
    config_name: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Compose a config with Hydra from a config directory.

    `config_dir` is the root of the config tree (containing the entrypoint and
    any config groups). `config_name` is the entrypoint config (without the
    `.yaml` suffix). `overrides` are Hydra-style override strings, e.g.
    `["trainer.max_epochs=5", "model=conv"]`.

    Returns the composed DictConfig. Interpolations are not yet resolved; use
    `resolve_to_plain_dict` (or `load_resolved_config`) to cross the boundary.
    """
    absolute_dir = Path(config_dir).expanduser().resolve()

    if not absolute_dir.is_dir():
        raise ConfigError(f"Config directory does not exist: {absolute_dir}")

    override_list = list(overrides) if overrides is not None else []

    # initialize_config_dir requires no other global Hydra instance to be active.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    try:
        with initialize_config_dir(
            config_dir=str(absolute_dir),
            version_base=None,
        ):
            config = compose(
                config_name=config_name,
                overrides=override_list,
            )
    except HydraException as error:
        raise ConfigError(
            f"Hydra failed to compose '{config_name}' from {absolute_dir}: {error}"
        ) from error

    return config


def load_resolved_config(
    config_dir: str | Path,
    config_name: str,
    overrides: list[str] | None = None,
) -> dict:
    """
    Compose a config with Hydra and return a resolved plain Python dict.

    This is the full config-layer entrypoint: compose -> resolve interpolation
    -> plain dict. The returned dict is safe to hand to project factories.
    """
    config = compose_config(
        config_dir=config_dir,
        config_name=config_name,
        overrides=overrides,
    )

    return resolve_to_plain_dict(config)
