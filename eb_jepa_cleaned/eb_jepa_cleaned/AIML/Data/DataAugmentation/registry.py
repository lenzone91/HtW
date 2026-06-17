from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Data augmentation registry (anchor)
#############################################

# Object files import AUGMENTATION_REGISTRY from here to register themselves.
# This module imports only Workflow.Factory, so it never participates in a
# decorator-registry import cycle.


AUGMENTATION_REGISTRY = Registry(object_family="augmentation")
AUGMENTATION_BUILDER = RegistryBuilder(registry=AUGMENTATION_REGISTRY)
