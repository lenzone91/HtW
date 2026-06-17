from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Scheduler registry (anchor)
#############################################

# Schedulers route by a `scheduler_type` field. Imports only Workflow.Factory.


SCHEDULER_REGISTRY = Registry(object_family="scheduler")
SCHEDULER_BUILDER = RegistryBuilder(
    registry=SCHEDULER_REGISTRY,
    type_field="scheduler_type",
)
