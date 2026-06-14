from ..Workflow.Factory.builder import RegistryBuilder
from ..Workflow.Factory.registry import Registry


TRAINER_REGISTRY = Registry(object_family="trainer")

TRAINER_BUILDER = RegistryBuilder(
    registry=TRAINER_REGISTRY,
    type_field="trainer_type",
)