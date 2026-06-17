from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Optimizer registry (anchor)
#############################################

# Optimizers route by an `optimizer_type` field. Imports only Workflow.Factory.


OPTIMIZER_REGISTRY = Registry(object_family="optimizer")
OPTIMIZER_BUILDER = RegistryBuilder(
    registry=OPTIMIZER_REGISTRY,
    type_field="optimizer_type",
)
