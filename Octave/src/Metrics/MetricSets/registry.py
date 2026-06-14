from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry, SubBuildDeclaration
from ..Metrics import factory as metric_factory  # noqa: F401
from ..Metrics.registry import METRIC_BUILDER


METRIC_SET_REGISTRY = Registry(object_family="metric_set")

METRIC_SET_BUILDER = RegistryBuilder(
    registry=METRIC_SET_REGISTRY,
    type_field="set_type",
)


METRICS_SUB_BUILD = SubBuildDeclaration(
    source_key="metrics",
    target_key="metrics",
    builder=METRIC_BUILDER,
    build_method="named",
)