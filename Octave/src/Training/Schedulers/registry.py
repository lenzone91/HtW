from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry


SCHEDULER_REGISTRY = Registry(object_family="scheduler")

SCHEDULER_BUILDER = RegistryBuilder(
    registry=SCHEDULER_REGISTRY,
    type_field="scheduler_type",
)