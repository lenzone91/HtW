from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry


COLLATOR_REGISTRY = Registry(object_family="collator")
COLLATOR_BUILDER = RegistryBuilder(registry=COLLATOR_REGISTRY)