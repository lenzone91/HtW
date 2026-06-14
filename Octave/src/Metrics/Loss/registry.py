from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry


LOSS_REGISTRY = Registry(object_family="loss")

LOSS_BUILDER = RegistryBuilder(
    registry=LOSS_REGISTRY,
    type_field="loss_type",
)