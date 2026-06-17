"""
AcVideoJEPA objective metrics: registration, building through the AIML factories
(incl. the encoder-shape runtime-context path), and evaluation over a
LatentRolloutOutput.
"""

import pytest
import torch

# Importing the subpackage registers the metrics onto METRIC_REGISTRY.
import eb_jepa_cleaned.AcVideoJEPA.Metrics  # noqa: F401
from eb_jepa_cleaned.AcVideoJEPA.Metrics.regularizers import HingeStdLossMetric
from eb_jepa_cleaned.AcVideoJEPA.Models.Backbones.inverse_dynamics import (
    InverseDynamicsModel,
)
from eb_jepa_cleaned.AcVideoJEPA.Models.Backbones.projector import Projector
from eb_jepa_cleaned.AcVideoJEPA.Models.Rollout.output import LatentRolloutOutput
from eb_jepa_cleaned.AIML.Metrics.Metrics.factory import build_metric
from eb_jepa_cleaned.AIML.Metrics.Metrics.registry import METRIC_REGISTRY

B, C, T = 2, 4, 4
ENCODER_SHAPE = {"feature_dim": C, "height": 1, "width": 1}


def default_cfg(name: str) -> dict:
    """Canonical default config for a registered metric (the builder does not
    merge defaults; structural sub-configs must be supplied)."""
    return METRIC_REGISTRY.get_default_config(name)


def _rollout(unroll_mode, *, nsteps=2, ctxt=1, predicted=None):
    encoded = torch.randn(B, C, T, 1, 1)
    return LatentRolloutOutput(
        encoded_states=encoded,
        predicted_states=encoded.clone() if predicted is None else predicted,
        actions_encoded=None,
        nsteps=nsteps,
        unroll_mode=unroll_mode,
        effective_ctxt_window=ctxt,
    )


#############################################
# Registration
#############################################


def test_metrics_registered():
    for name in (
        "autoregressive_prediction_loss",
        "parallel_prediction_loss",
        "hinge_std",
        "covariance",
        "temporal_similarity",
        "inverse_dynamics",
    ):
        assert METRIC_REGISTRY.has(name)


#############################################
# Prediction metrics
#############################################


def test_autoregressive_prediction_loss_zero_when_predictions_match():
    metric = build_metric(
        default_cfg("autoregressive_prediction_loss"),
        metric_name="autoregressive_prediction_loss",
    )
    out = _rollout("autoregressive")  # predicted == encoded
    assert metric(out).item() == pytest.approx(0.0)


def test_prediction_metric_rejects_wrong_rollout_mode():
    metric = build_metric(
        default_cfg("autoregressive_prediction_loss"),
        metric_name="autoregressive_prediction_loss",
    )
    with pytest.raises(ValueError):
        metric(_rollout("parallel"))


def test_parallel_prediction_loss_single_shot():
    metric = build_metric(
        default_cfg("parallel_prediction_loss"),
        metric_name="parallel_prediction_loss",
    )
    out = _rollout("parallel")  # no predicted_steps -> single comparison, == encoded
    assert metric(out).item() == pytest.approx(0.0)


#############################################
# Regularizers (no projector)
#############################################


def test_hinge_std_metric_builds_and_evaluates():
    metric = build_metric({}, metric_name="hinge_std")
    out = _rollout("autoregressive")
    value = metric(out)
    assert value.ndim == 0


def test_covariance_metric_evaluates():
    metric = build_metric({}, metric_name="covariance")
    assert metric(_rollout("autoregressive")).ndim == 0


#############################################
# Encoder-shape-dependent builds (runtime_context)
#############################################


def test_projector_built_from_runtime_context_encoder_shape():
    metric = build_metric(
        {"projector": {"enabled": True, "mlp_spec": f"{C}-8-8", "hidden_multiplier": 4}},
        metric_name="hinge_std",
        runtime_context={"encoder_shape": ENCODER_SHAPE},
    )
    assert isinstance(metric.projector, Projector)
    assert metric(_rollout("autoregressive")).ndim == 0


def test_projector_build_requires_encoder_shape():
    with pytest.raises(ValueError):
        build_metric(
            {"projector": {"enabled": True, "mlp_spec": None, "hidden_multiplier": 4}},
            metric_name="hinge_std",
        )


def test_inverse_dynamics_metric_builds_idm_and_uses_actions():
    metric = build_metric(
        default_cfg("inverse_dynamics"),
        metric_name="inverse_dynamics",
        runtime_context={"encoder_shape": ENCODER_SHAPE},
    )
    assert isinstance(metric.metric.idm, InverseDynamicsModel)
    actions = torch.randn(B, 2, T)
    assert metric(_rollout("autoregressive"), actions).ndim == 0


#############################################
# Direct construction (projector disabled -> identity)
#############################################


def test_regularizer_without_projector_uses_identity():
    metric = HingeStdLossMetric(projector=None)
    out = _rollout("autoregressive")
    assert metric(out).ndim == 0
