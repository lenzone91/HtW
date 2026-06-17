from .base import BaseDataModule
from .registry import DATAMODULE_REGISTRY, DATAMODULE_SUB_BUILDS


#############################################
# Default data module
#############################################

# default_config is the top-level key allow-list (datasets / collators /
# dataloader_configs); the builder validates keys against it and does not merge
# its values. The actual per-phase contents come from the composed config.

DEFAULT_DATAMODULE_CONFIG = {
    "datasets": {},
    "collators": {},
    "dataloader_configs": {},
}


@DATAMODULE_REGISTRY.register_class(
    name="default",
    default_config=DEFAULT_DATAMODULE_CONFIG,
    sub_builds=DATAMODULE_SUB_BUILDS,
)
class DefaultDataModule(BaseDataModule):
    """
    Default project DataModule.

    Adds no behavior over BaseDataModule; it is the standard concrete DataModule
    exposed to the factory. It is generic: it composes whichever registered
    datasets and collators the config selects.
    """

    pass
