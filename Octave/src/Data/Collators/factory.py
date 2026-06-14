from . import ac_video_jepa_collator as _ac_video_jepa_collator  # noqa: F401
from .ac_video_jepa_collator import AcVideoJepaCollator
from .configs import DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG
from .registry import COLLATOR_REGISTRY
from ...Workflow.Factory.builder import RegistryBuilder


class CollatorBuilder(RegistryBuilder):
    """Shared-registry builder specialized for collators."""

    def __init__(
        self,
        strict: bool = True,
        check_default_keys: bool = True,
    ) -> None:
        super().__init__(
            registry=COLLATOR_REGISTRY,
            strict=strict,
            check_default_keys=check_default_keys,
            type_field="collator_type",
        )


def make_collator_builder(strict: bool = True) -> CollatorBuilder:
    return CollatorBuilder(strict=strict)


def build_collator(
    config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> AcVideoJepaCollator:
    builder = make_collator_builder(strict=strict)
    return builder.build_one(
        config=config,
        runtime_context=runtime_context,
    )


def build_ac_video_jepa_collator(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> AcVideoJepaCollator:
    builder = make_collator_builder(strict=strict)
    return builder.build_one(
        config=config or DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG,
        runtime_context=runtime_context,
        name="ac_video_jepa",
    )