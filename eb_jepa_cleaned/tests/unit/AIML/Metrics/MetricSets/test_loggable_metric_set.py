"""
Unit tests for AIML.Metrics.MetricSets.loggable_metric_set.LoggableMetricSet.

Covers log-dict flattening of scalar, dict, sequence, and vector tensor outputs.
"""

import pytest
import torch

from eb_jepa_cleaned.AIML.Metrics.MetricSets.loggable_metric_set import (
    LoggableMetricSet,
)


def test_scalar_and_dict_and_sequence_flattening():
    metric_set = LoggableMetricSet(metrics={})

    log = metric_set.to_log_dict(
        {
            "snr": torch.tensor(3.0),
            "parts": {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
            "seq": [torch.tensor(0.0), torch.tensor(1.0)],
        }
    )

    assert log["snr"] == 3.0
    assert log["parts/a"] == 1.0 and log["parts/b"] == 2.0
    assert log["seq/0"] == 0.0 and log["seq/1"] == 1.0


def test_vector_tensor_requires_known_output_names():
    metric_set = LoggableMetricSet(metrics={})

    with pytest.raises(ValueError, match="no output names were provided"):
        metric_set.to_log_dict({"dnsmos": torch.tensor([1.0, 2.0, 3.0])})


def test_vector_tensor_split_with_known_names():
    class Named(LoggableMetricSet):
        known_output_names = {"dnsmos": ("sig", "bak", "ovrl")}

    metric_set = Named(metrics={})

    log = metric_set.to_log_dict({"dnsmos": torch.tensor([1.0, 2.0, 3.0])})

    assert log["dnsmos/sig"] == 1.0
    assert log["dnsmos/bak"] == 2.0
    assert log["dnsmos/ovrl"] == 3.0


def test_sequence_length_mismatch_with_known_names_raises():
    class Named(LoggableMetricSet):
        known_output_names = {"m": ("a", "b")}

    metric_set = Named(metrics={})

    with pytest.raises(ValueError, match="output names were provided"):
        metric_set.to_log_dict({"m": [torch.tensor(1.0)]})
