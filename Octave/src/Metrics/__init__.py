"""Metric, metric-set, and loss construction for Octave."""

from .factory import (
    build_ac_video_jepa_metric_stack,
    build_loss,
    build_metric_from_config,
    build_metric_set,
    build_metrics,
)
from .Loss.loss import WeightedMetricLoss
from .MetricSets.metric_set import AcVideoJepaMetricSet, LoggableMetricSet, MetricSet
from .Metrics.prediction_metrics import (
    AutoregressivePredictionLossMetric,
    ParallelPredictionLossMetric,
)
from .Metrics.regularizer_metrics import (
    CovarianceLossMetric,
    HingeStdLossMetric,
    InverseDynamicsLossMetric,
    TemporalSimilarityLossMetric,
)

__all__ = [
    "AcVideoJepaMetricSet",
    "AutoregressivePredictionLossMetric",
    "CovarianceLossMetric",
    "HingeStdLossMetric",
    "InverseDynamicsLossMetric",
    "LoggableMetricSet",
    "MetricSet",
    "ParallelPredictionLossMetric",
    "TemporalSimilarityLossMetric",
    "WeightedMetricLoss",
    "build_ac_video_jepa_metric_stack",
    "build_loss",
    "build_metric_from_config",
    "build_metric_set",
    "build_metrics",
]
