"""
AcVideoJEPA metrics factory.

Importing this module registers the JEPA objective metrics onto the AIML metric
registry (the registration side effect) and the prediction cost onto the local
prediction-cost registry. The metrics are then built through the ordinary AIML
metric / metric-set factories.
"""

from . import prediction  # noqa: F401  (registers prediction metrics + cost)
from . import regularizers  # noqa: F401  (registers regularizer metrics)

# Names registered onto the AIML METRIC_REGISTRY by importing this module.
REGISTERED_METRICS = (
    "autoregressive_prediction_loss",
    "parallel_prediction_loss",
    "hinge_std",
    "covariance",
    "temporal_similarity",
    "inverse_dynamics",
)
