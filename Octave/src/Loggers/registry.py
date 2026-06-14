from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry


LOGGER_REGISTRY = Registry(object_family="logger")

LOGGER_BUILDER = RegistryBuilder(
    registry=LOGGER_REGISTRY,
    type_field="logger_type",
)