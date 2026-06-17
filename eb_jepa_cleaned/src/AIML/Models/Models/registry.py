from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Model registry (anchor)
#############################################

# Models are plain nn.Module objects keyed by their registered name. Imports only
# Workflow.Factory.


MODEL_REGISTRY = Registry(object_family="model")
MODEL_BUILDER = RegistryBuilder(registry=MODEL_REGISTRY)
