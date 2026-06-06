from .merge import merge_configs


#############################################
# Config overrides
#############################################

# This file handles explicit Python-level config overrides.
#
# It does not:
#   - load config files;
#   - save config files;
#   - parse CLI override strings;
#   - resolve runtime paths;
#   - validate subsystem-specific config content;
#   - build objects.


def apply_overrides(
    config: dict,
    overrides: dict | None = None,
    strict: bool = True,
    **kwargs,
) -> dict:
    """
    Apply nested dictionary overrides and keyword overrides to a config.

    Keyword overrides are interpreted as top-level config entries.
    For nested overrides, prefer the overrides dictionary.
    """
    merged_overrides = _merge_override_sources(
        overrides=overrides,
        kwargs=kwargs,
    )

    return merge_configs(
        base_config=config,
        override_config=merged_overrides,
        strict=strict,
    )


def _merge_override_sources(
    overrides: dict | None,
    kwargs: dict,
) -> dict | None:
    """
    Merge dictionary overrides and keyword overrides into one override dict.
    """
    if overrides is None and len(kwargs) == 0:
        return None

    if overrides is None:
        return dict(kwargs)

    if len(kwargs) == 0:
        return overrides

    return merge_configs(
        base_config=overrides,
        override_config=kwargs,
        strict=False,
    )