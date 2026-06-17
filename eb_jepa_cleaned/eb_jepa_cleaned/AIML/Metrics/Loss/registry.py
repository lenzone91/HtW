from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Loss registry (anchor)
#############################################

# Object files import LOSS_REGISTRY from here to register themselves. This module
# imports only Workflow.Factory, so it never participates in a decorator-registry
# import cycle. Losses route by a `loss_type` field.


LOSS_REGISTRY = Registry(object_family="loss")
LOSS_BUILDER = RegistryBuilder(registry=LOSS_REGISTRY, type_field="loss_type")
