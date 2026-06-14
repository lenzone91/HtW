from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry


DATASET_REGISTRY = Registry(object_family="dataset")
DATASET_BUILDER = RegistryBuilder(registry=DATASET_REGISTRY)