from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Data adaptation registry (anchor)
#############################################

# Object files import ADAPTATION_REGISTRY from here to register themselves.
# This module imports only Workflow.Factory, so it never participates in a
# decorator-registry import cycle.


ADAPTATION_REGISTRY = Registry(object_family="adaptation")
ADAPTATION_BUILDER = RegistryBuilder(registry=ADAPTATION_REGISTRY)
