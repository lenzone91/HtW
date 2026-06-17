from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Early stopping registry (anchor)
#############################################

# Early stopping callbacks route by an `early_stopping_type` field.
# check_default_keys is False: EarlyStopping accepts many kwargs not enumerated
# in the defaults.


EARLY_STOPPING_REGISTRY = Registry(object_family="early stopping")
EARLY_STOPPING_BUILDER = RegistryBuilder(
    registry=EARLY_STOPPING_REGISTRY,
    type_field="early_stopping_type",
    check_default_keys=False,
)
