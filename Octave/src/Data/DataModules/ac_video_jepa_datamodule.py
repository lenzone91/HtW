from .base import BaseDataModule
from .configs import DEFAULT_AC_VIDEO_JEPA_DATAMODULE_CONFIG
from .registry import (
    COLLATORS_SUB_BUILD,
    DATAMODULE_REGISTRY,
    DATASETS_SUB_BUILD,
)


@DATAMODULE_REGISTRY.register_class(
    name="ac_video_jepa",
    default_config=DEFAULT_AC_VIDEO_JEPA_DATAMODULE_CONFIG,
    sub_builds=(
        DATASETS_SUB_BUILD,
        COLLATORS_SUB_BUILD,
    ),
)
class AcVideoJepaDataModule(BaseDataModule):
    """Thin AcVideoJepa DataModule wrapper."""
