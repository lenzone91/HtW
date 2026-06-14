from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry


MODULE_REGISTRY = Registry(object_family="module")
MODULE_BUILDER = RegistryBuilder(
    registry=MODULE_REGISTRY,
    type_field="module_type",
)