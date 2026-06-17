from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry


#############################################
# Collator registry (anchor)
#############################################

# Object files import COLLATOR_REGISTRY from here to register themselves.
# This module imports only Workflow.Factory, so it never participates in a
# decorator-registry import cycle.
#
# Concrete collators declare their own batch-transform sub-builds (augmentations
# / adaptations). The current concrete collator is audio-specific (Phase 3), so
# that wiring is not defined here yet.


COLLATOR_REGISTRY = Registry(object_family="collator")
COLLATOR_BUILDER = RegistryBuilder(registry=COLLATOR_REGISTRY)
