from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry


LOADING_REGISTRY = Registry(object_family="loading")

LOADING_BUILDER = RegistryBuilder(
    registry=LOADING_REGISTRY,
    type_field="type",
)