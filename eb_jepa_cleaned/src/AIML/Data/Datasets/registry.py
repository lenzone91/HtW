from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Dataset registry (anchor)
#############################################

# Object files import DATASET_REGISTRY from here to register themselves.
# This module imports only Workflow.Factory, so it never participates in a
# decorator-registry import cycle.


DATASET_REGISTRY = Registry(object_family="dataset")
DATASET_BUILDER = RegistryBuilder(registry=DATASET_REGISTRY)
