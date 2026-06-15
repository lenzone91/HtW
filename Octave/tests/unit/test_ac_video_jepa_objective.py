from copy import deepcopy

import pytest
import torch

from Octave.src.Metrics.factory import build_ac_video_jepa_metric_stack
from Octave.src.Metrics.Loss.configs import DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG
from Octave.src.Metrics.Loss.factory import build_loss
from Octave.src.Metrics.Loss.loss import WeightedMetricLoss
from Octave.src.Metrics.MetricSets.configs import (
    DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG,
)
from Octave.src.Metrics.MetricSets.factory import build_ac_video_jepa_metric_set
from Octave.src.Metrics.MetricSets.metric_set import AcVideoJepaMetricSet
from Octave.src.Metrics.Metrics.configs import DEFAULT_METRIC_CONFIGS
from Octave.src.Metrics.Metrics.factory import build_metrics
from Octave.src.Metrics.Metrics.prediction_metrics import (
    AutoregressivePredictionLossMetric,
)
from Octave.src.Metrics.Metrics.regularizer_metrics import (
    CovarianceLossMetric,
    HingeStdLossMetric,
    InverseDynamicsLossMetric,
    TemporalSimilarityLossMetric,
)
from Octave.src.Models.Model.ac_video_jepa.configs import (
    DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG,
)
from Octave.src.Models.Model.ac_video_jepa.factory import (
    build_ac_video_jepa_components,
)
from Octave.src.Models.Modules.ac_video_jepa_module import AcVideoJepaModule
from Octave.src.Rollouts.latent_rollout import LatentRollout


def make_tiny_components_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG)
    config["encoder"].update(
        {
            "stack_sizes": [4, 8, 8],
            "input_channels": 2,
            "input_shape": [2, 32, 32],
            "mlp_output_dim": 32,
        }
    )
    config["predictor"].update(
        {
            "hidden_size": 32,
            "action_dim": 2,
        }
    )
    return config


def make_tiny_metric_set_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG)
    config["metrics"]["idm_loss"]["inverse_dynamics_model"].update(
        {
            "hidden_dim": 16,
            "action_dim": 2,
        }
    )
    return config


def make_tiny_loss_config() -> dict:
    config = deepcopy(DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG)
    config["metric_weights"].update(
        {
            "prediction_loss": 1.0,
            "cov_loss": 0.1,
            "std_loss": 0.1,
            "sim_loss_t": 0.1,
            "idm_loss": 0.1,
        }
    )
    return config


def make_rollout_output():
    components = build_ac_video_jepa_components(config=make_tiny_components_config())
    rollout = LatentRollout(
        nsteps=2,
        unroll_mode="autoregressive",
        ctxt_window_time=1,
    )
    metric_stack = build_ac_video_jepa_metric_stack(
        metric_set_config=make_tiny_metric_set_config(),
        loss_config=make_tiny_loss_config(),
        encoder_shape=components["encoder_shape"],
    )
    jepa = AcVideoJepaModule(
        encoder=components["encoder"],
        action_encoder=components["action_encoder"],
        predictor=components["predictor"],
        encoder_shape=components["encoder_shape"],
        rollout=rollout,
        metric_set=metric_stack["metric_set"],
        loss=metric_stack["loss"],
        optimizer_builder=lambda parameters: torch.optim.AdamW(parameters, lr=1e-3),
        scheduler_builder=lambda optimizer: None,
    )
    return rollout(
        jepa=jepa,
        observations=torch.randn(2, 2, 4, 32, 32),
        actions=torch.randn(2, 2, 4),
    ), components["encoder_shape"]


def test_build_ac_video_jepa_metric_stack_builds_metric_set_and_loss() -> None:
    _, encoder_shape = make_rollout_output()

    stack = build_ac_video_jepa_metric_stack(
        metric_set_config=make_tiny_metric_set_config(),
        loss_config=make_tiny_loss_config(),
        encoder_shape=encoder_shape,
    )

    assert isinstance(stack["metric_set"], AcVideoJepaMetricSet)
    assert isinstance(stack["loss"], WeightedMetricLoss)
    assert isinstance(
        stack["metric_set"].metrics["prediction_loss"],
        AutoregressivePredictionLossMetric,
    )
    assert isinstance(stack["metric_set"].metrics["std_loss"], HingeStdLossMetric)
    assert isinstance(stack["metric_set"].metrics["cov_loss"], CovarianceLossMetric)
    assert isinstance(
        stack["metric_set"].metrics["sim_loss_t"],
        TemporalSimilarityLossMetric,
    )
    assert isinstance(
        stack["metric_set"].metrics["idm_loss"],
        InverseDynamicsLossMetric,
    )


def test_build_ac_video_jepa_metric_stack_does_not_mutate_input_configs() -> None:
    _, encoder_shape = make_rollout_output()
    metric_set_config = make_tiny_metric_set_config()
    loss_config = make_tiny_loss_config()
    original_metric_set_config = deepcopy(metric_set_config)
    original_loss_config = deepcopy(loss_config)

    build_ac_video_jepa_metric_stack(
        metric_set_config=metric_set_config,
        loss_config=loss_config,
        encoder_shape=encoder_shape,
    )

    assert metric_set_config == original_metric_set_config
    assert loss_config == original_loss_config


def test_ac_video_jepa_metric_set_returns_flat_metric_logs() -> None:
    rollout_output, encoder_shape = make_rollout_output()
    metric_set = build_ac_video_jepa_metric_set(
        config=make_tiny_metric_set_config(),
        encoder_shape=encoder_shape,
    )

    metric_values = metric_set(
        rollout_output=rollout_output,
        actions=torch.randn(2, 2, 4),
    )

    assert metric_values["prediction_loss"].shape == torch.Size([])
    assert metric_values["std_loss"].shape == torch.Size([])
    assert metric_values["cov_loss"].shape == torch.Size([])
    assert metric_values["sim_loss_t"].shape == torch.Size([])
    assert metric_values["idm_loss"].shape == torch.Size([])


def test_metric_stack_computes_weighted_scalar_loss_and_logs() -> None:
    rollout_output, encoder_shape = make_rollout_output()
    stack = build_ac_video_jepa_metric_stack(
        metric_set_config=make_tiny_metric_set_config(),
        loss_config=make_tiny_loss_config(),
        encoder_shape=encoder_shape,
    )

    metric_values = stack["metric_set"](
        rollout_output=rollout_output,
        actions=torch.randn(2, 2, 4),
    )
    loss, loss_logs = stack["loss"](metric_values)

    expected_loss = (
        metric_values["prediction_loss"]
        + 0.1 * metric_values["cov_loss"]
        + 0.1 * metric_values["std_loss"]
        + 0.1 * metric_values["sim_loss_t"]
        + 0.1 * metric_values["idm_loss"]
    )

    assert loss.shape == torch.Size([])
    assert loss_logs["loss"] is loss
    assert torch.isclose(loss, expected_loss)
    assert torch.isclose(
        loss_logs["loss/prediction_loss"],
        metric_values["prediction_loss"],
    )
    assert torch.isclose(
        loss_logs["loss/cov_loss"],
        0.1 * metric_values["cov_loss"],
    )


def test_build_metrics_rejects_unknown_metric_type() -> None:
    _, encoder_shape = make_rollout_output()
    config = deepcopy(DEFAULT_METRIC_CONFIGS)
    config["std_loss"]["metric_type"] = "unsupported"

    with pytest.raises(RuntimeError, match="Unknown metric"):
        build_metrics(metric_configs=config, encoder_shape=encoder_shape)


def test_build_metrics_rejects_unknown_nested_key() -> None:
    _, encoder_shape = make_rollout_output()
    config = deepcopy(DEFAULT_METRIC_CONFIGS)
    config["std_loss"]["unknown"] = "bad"

    with pytest.raises(RuntimeError, match="Invalid config keys"):
        build_metrics(metric_configs=config, encoder_shape=encoder_shape)


def test_prediction_cost_rejects_unknown_type() -> None:
    _, encoder_shape = make_rollout_output()
    config = deepcopy(DEFAULT_METRIC_CONFIGS)
    config["prediction_loss"]["prediction_cost"][
        "prediction_cost_type"
    ] = "unsupported"

    with pytest.raises(RuntimeError, match="Unknown prediction_cost"):
        build_metrics(metric_configs=config, encoder_shape=encoder_shape)


def test_prediction_cost_rejects_projector_without_explicit_projector() -> None:
    _, encoder_shape = make_rollout_output()
    config = deepcopy(DEFAULT_METRIC_CONFIGS)
    config["prediction_loss"]["prediction_cost"]["use_projector"] = True

    with pytest.raises(ValueError, match="requires an explicit proj"):
        build_metrics(metric_configs=config, encoder_shape=encoder_shape)


def test_idm_metric_rejects_missing_encoder_shape() -> None:
    with pytest.raises(ValueError, match="encoder_shape is required"):
        build_ac_video_jepa_metric_set(config=make_tiny_metric_set_config())


def test_build_loss_rejects_unknown_loss_type() -> None:
    config = make_tiny_loss_config()
    config["loss_type"] = "unsupported"

    with pytest.raises(RuntimeError, match="Unknown loss"):
        build_loss(loss_config=config)
