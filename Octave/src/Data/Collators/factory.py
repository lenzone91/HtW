from . import ac_video_jepa_collator  # noqa: F401
from .configs import (
    DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG,
    DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIGS,
)
from ...Workflow.Factory.builder import RegistryBuilder
from .registry import COLLATOR_REGISTRY


def build_collator_from_config(
    collator_config: dict,
    collator_name: str,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    builder = RegistryBuilder(
        registry=COLLATOR_REGISTRY,
        strict=strict,
    )

    return builder.build_one(
        config=collator_config,
        runtime_context=runtime_context,
        name=collator_name,
    )


def build_collators(
    collator_configs: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict:
    builder = RegistryBuilder(
        registry=COLLATOR_REGISTRY,
        strict=strict,
    )

    return builder.build_named(
        configs=collator_configs,
        runtime_context=runtime_context,
    )


def build_collator(
    collator_config: dict | None = None,
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    if collator_config is None:
        collator_config = config

    if collator_config is None:
        collator_config = DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIGS
    elif is_single_collator_config(collator_config):
        collator_config = {
            "ac_video_jepa": collator_config,
        }

    collators = build_collators(
        collator_configs=collator_config,
        runtime_context=runtime_context,
        strict=strict,
    )

    if collators is None:
        return None

    if len(collators) != 1:
        raise ValueError(
            "build_collator expects exactly one collator config, "
            f"got {len(collators)}."
        )

    return next(iter(collators.values()))


def is_single_collator_config(config: dict) -> bool:
    if any(key in DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG for key in config):
        return True

    return not all(isinstance(value, dict) for value in config.values())


def build_ac_video_jepa_collator(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    return build_collator_from_config(
        collator_config=config or DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG,
        collator_name="ac_video_jepa",
        runtime_context=runtime_context,
        strict=strict,
    )
