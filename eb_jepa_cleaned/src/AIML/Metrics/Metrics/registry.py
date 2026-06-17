from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Metric registry (anchor)
#############################################

# Object files import METRIC_REGISTRY from here to register themselves. This
# module imports only Workflow.Factory, so it never participates in a
# decorator-registry import cycle.


METRIC_REGISTRY = Registry(object_family="metric")
METRIC_BUILDER = RegistryBuilder(registry=METRIC_REGISTRY)
