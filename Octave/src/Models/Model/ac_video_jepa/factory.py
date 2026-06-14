from .components import AcVideoJepaComponents
from .configs import DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG
from .registry import AC_VIDEO_JEPA_COMPONENTS_REGISTRY
from ....Workflow.Factory.builder import RegistryBuilder


def build_ac_video_jepa_components(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict:
    builder = RegistryBuilder(
        registry=AC_VIDEO_JEPA_COMPONENTS_REGISTRY,
        strict=strict,
        type_field="model_type",
    )

    components = builder.build_one(
        config=config or DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG,
        runtime_context=runtime_context,
    )

    if components is None:
        return None

    if not isinstance(components, AcVideoJepaComponents):
        raise TypeError(
            "Expected AcVideoJepaComponents, "
            f"got {type(components).__name__}."
        )

    return components.as_dict()