from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry, SubBuildDeclaration
from ..Metrics.registry import METRIC_BUILDER


#############################################
# Metric set registry (anchor + sub-build wiring)
#############################################

# Anchors the metric-set family and declares the cross-subsystem sub-build that
# turns a named `metrics` config into built metric objects. Imports the metric
# builder, but never imports metric-set object files, so there is no cycle.
#
# check_default_keys is False: metric-set configs carry a flexible `metrics`
# mapping whose keys are metric names, not a fixed allow-list.

METRIC_SET_REGISTRY = Registry(object_family="metric set")
METRIC_SET_BUILDER = RegistryBuilder(
    registry=METRIC_SET_REGISTRY,
    type_field="set_type",
    check_default_keys=False,
)

METRICS_SUB_BUILD = SubBuildDeclaration(
    source_key="metrics",
    target_key="metrics",
    builder=METRIC_BUILDER,
    build_method="named",
)
