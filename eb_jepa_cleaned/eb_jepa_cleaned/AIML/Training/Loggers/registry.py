from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Logger registry (anchor)
#############################################

# Loggers are keyed by their registered name (no routing type field).


LOGGER_REGISTRY = Registry(object_family="logger")
LOGGER_BUILDER = RegistryBuilder(registry=LOGGER_REGISTRY)
