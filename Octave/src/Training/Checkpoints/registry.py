from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry


CHECKPOINT_REGISTRY = Registry(object_family="checkpoint")

CHECKPOINT_BUILDER = RegistryBuilder(
    registry=CHECKPOINT_REGISTRY,
    type_field="checkpoint_type",
)