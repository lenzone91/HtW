from . import ac_video_jepa_collator  # noqa: F401
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
    collator_config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
):
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