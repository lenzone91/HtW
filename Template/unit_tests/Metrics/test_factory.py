"""
Pytest test module for Metrics.factory.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Metrics.configs import (
    DEFAULT_LP_ERROR_CONFIG,
    DEFAULT_SNR_CONFIG,
    DEFAULT_TSE_LOSS_CONFIG,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Metrics.factory import (
    MetricBuilder,
    MetricDispatcher,
    build_loss,
    build_metric_set,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Metrics.loss import (
    WeightedMetricLoss,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Metrics.MetricSet import (
    TSEMetricSet,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Metrics.objective import (
    LPError,
    SNR,
)


#############################################
# Test configs
#############################################

LIGHT_METRICS_CONFIG = {
    "snr": dict(DEFAULT_SNR_CONFIG),
    "lp_error": dict(DEFAULT_LP_ERROR_CONFIG),
}

LIGHT_TSE_METRIC_SET_CONFIG = {
    "set_type": "tse",
    "strict": True,
    "metric_to_input_names": None,
    "metrics": dict(LIGHT_METRICS_CONFIG),
}


#############################################
# Metric builder tests
#############################################

def test_metric_builder_builds_metric() -> None:
    """
    Test that MetricBuilder builds one metric instance.
    """
    builder = MetricBuilder(
        metric_class=SNR,
        default_config=DEFAULT_SNR_CONFIG,
    )

    metric = builder(
        config=DEFAULT_SNR_CONFIG,
        runtime_context=None,
    )

    assert isinstance(metric, SNR)


#############################################
# Metric dispatcher tests
#############################################

def test_metric_dispatcher_builds_metrics() -> None:
    """
    Test that MetricDispatcher builds several registered metrics.
    """
    dispatcher = MetricDispatcher(strict=True)

    metrics = dispatcher(
        config=LIGHT_METRICS_CONFIG,
        runtime_context=None,
    )

    assert isinstance(metrics["snr"], SNR)
    assert isinstance(metrics["lp_error"], LPError)


def test_metric_dispatcher_rejects_unknown_metric() -> None:
    """
    Test that unknown metric names fail explicitly in strict mode.
    """
    dispatcher = MetricDispatcher(strict=True)

    with pytest.raises(RuntimeError):
        dispatcher(
            config={
                "unknown_metric": {},
            },
            runtime_context=None,
        )


def test_metric_dispatcher_non_strict_unknown_metric_returns_none() -> None:
    """
    Test that unknown metric configs do not build in non-strict mode.
    """
    dispatcher = MetricDispatcher(strict=False)

    metrics = dispatcher(
        config={
            "unknown_metric": {},
        },
        runtime_context=None,
    )

    assert metrics is None


#############################################
# Metric set builder tests
#############################################

def test_build_metric_set_builds_tse_metric_set() -> None:
    """
    Test that build_metric_set builds a TSEMetricSet from light configs.
    """
    metric_set = build_metric_set(
        config=LIGHT_TSE_METRIC_SET_CONFIG,
        runtime_context=None,
    )

    assert isinstance(metric_set, TSEMetricSet)
    assert set(metric_set.metrics.keys()) == {"snr", "lp_error"}


def test_build_metric_set_rejects_unknown_set_type() -> None:
    """
    Test that unknown metric set types fail explicitly.
    """
    config = dict(LIGHT_TSE_METRIC_SET_CONFIG)
    config["set_type"] = "unknown"

    with pytest.raises(ValueError):
        build_metric_set(
            config=config,
            runtime_context=None,
        )


#############################################
# Loss builder tests
#############################################

def test_build_loss_builds_weighted_metric_loss() -> None:
    """
    Test that build_loss builds a WeightedMetricLoss from default config.
    """
    loss = build_loss(
        config=DEFAULT_TSE_LOSS_CONFIG,
        runtime_context=None,
    )

    assert isinstance(loss, WeightedMetricLoss)


def test_build_loss_rejects_unknown_loss_type() -> None:
    """
    Test that unknown loss types fail explicitly.
    """
    config = dict(DEFAULT_TSE_LOSS_CONFIG)
    config["loss_type"] = "unknown"

    with pytest.raises(RuntimeError):
        build_loss(
            config=config,
            runtime_context=None,
        )