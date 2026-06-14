from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry


OPTIMIZER_REGISTRY = Registry(object_family="optimizer")

OPTIMIZER_BUILDER = RegistryBuilder(
    registry=OPTIMIZER_REGISTRY,
    type_field="optimizer_type",
)