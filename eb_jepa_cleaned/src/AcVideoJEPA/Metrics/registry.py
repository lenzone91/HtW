from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry, SubBuildDeclaration


#############################################
# Prediction-cost registry (anchor + sub-build wiring)
#############################################

# The prediction metrics sub-build a "prediction cost" (the latent distance,
# e.g. square_loss_seq). It is an experiment-local object family, so AcVideoJEPA
# owns this small registry. Routed by `prediction_cost_type`. The JEPA objective
# *metrics* themselves register onto the AIML metric registry, not here.

PREDICTION_COST_REGISTRY = Registry(object_family="prediction cost")
PREDICTION_COST_BUILDER = RegistryBuilder(
    registry=PREDICTION_COST_REGISTRY,
    type_field="prediction_cost_type",
)

PREDICTION_COST_SUB_BUILD = SubBuildDeclaration(
    source_key="prediction_cost",
    target_key="prediction_cost",
    builder=PREDICTION_COST_BUILDER,
    build_method="one",
    type_field="prediction_cost_type",
)
