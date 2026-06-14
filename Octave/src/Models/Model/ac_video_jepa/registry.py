from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


AC_VIDEO_JEPA_COMPONENTS_REGISTRY = Registry(
    object_family="ac_video_jepa_components"
)

AC_VIDEO_JEPA_COMPONENTS_BUILDER = RegistryBuilder(
    registry=AC_VIDEO_JEPA_COMPONENTS_REGISTRY,
    type_field="model_type",
)