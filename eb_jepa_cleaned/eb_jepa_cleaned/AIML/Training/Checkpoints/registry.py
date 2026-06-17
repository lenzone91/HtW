from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Checkpoint registry (anchor)
#############################################

# Checkpoint callbacks route by a `checkpoint_type` field. check_default_keys is
# False: ModelCheckpoint accepts many kwargs not enumerated in the defaults.


CHECKPOINT_REGISTRY = Registry(object_family="checkpoint")
CHECKPOINT_BUILDER = RegistryBuilder(
    registry=CHECKPOINT_REGISTRY,
    type_field="checkpoint_type",
    check_default_keys=False,
)
