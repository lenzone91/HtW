from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry, SubBuildDeclaration


PREDICTION_COST_REGISTRY = Registry(object_family="prediction_cost")
PREDICTION_COST_BUILDER = RegistryBuilder(
    registry=PREDICTION_COST_REGISTRY,
    type_field="prediction_cost_type",
)


METRIC_REGISTRY = Registry(object_family="metric")
METRIC_BUILDER = RegistryBuilder(
    registry=METRIC_REGISTRY,
    type_field="metric_type",
)


PREDICTION_COST_SUB_BUILD = SubBuildDeclaration(
    source_key="prediction_cost",
    target_key="prediction_cost",
    builder=PREDICTION_COST_BUILDER,
    build_method="one",
    type_field="prediction_cost_type",
    forwarded_kwargs=(),
)
