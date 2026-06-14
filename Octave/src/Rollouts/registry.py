from ..Workflow.Factory.builder import RegistryBuilder
from ..Workflow.Factory.registry import Registry


ROLLOUT_REGISTRY = Registry(object_family="rollout")
ROLLOUT_BUILDER = RegistryBuilder(
    registry=ROLLOUT_REGISTRY,
    type_field="rollout_type",
)
