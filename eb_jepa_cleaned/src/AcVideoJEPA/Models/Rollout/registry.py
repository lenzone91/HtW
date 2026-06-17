from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Rollout registry (anchor)
#############################################

# Rollout is an experiment-specific object family (no AIML rollout concept), so
# AcVideoJEPA owns a local registry. Concrete rollouts route by `rollout_type`
# and register at import time. Imports only Workflow.Factory.

ROLLOUT_REGISTRY = Registry(object_family="rollout")
ROLLOUT_BUILDER = RegistryBuilder(
    registry=ROLLOUT_REGISTRY,
    type_field="rollout_type",
)
