from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Lightning module registry (anchor)
#############################################

# Concrete modules route by a `module_type` field. Imports only Workflow.Factory.
# Concrete modules declare their own field resolutions to build their model /
# metrics / loss sub-objects; those are audio/TSE-specific (Phase 3).


LIGHTNING_MODULE_REGISTRY = Registry(object_family="lightning module")
LIGHTNING_MODULE_BUILDER = RegistryBuilder(
    registry=LIGHTNING_MODULE_REGISTRY,
    type_field="module_type",
)
