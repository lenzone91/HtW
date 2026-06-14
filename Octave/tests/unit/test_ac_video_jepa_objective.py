from copy import deepcopy

import pytest
import torch

from Octave.src.Metrics.ac_video_jepa_objective import AcVideoJepaObjective
from Octave.src.Metrics.Metrics.prediction_metrics import (
    AutoregressivePredictionLossMetric,
)
from Octave.src.Metrics.Metrics.regularizer_metrics import (
    CovarianceLossMetric,
    HingeStdLossMetric,
    InverseDynamicsLossMetric,
    TemporalSimilarityLossMetric,
)
from Octave.src.Metrics.configs import DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG
from Octave.src.Metrics.factory import build_ac_video_jepa_objective
from Octave.src.Metrics.Loss.loss import WeightedMetricLoss
from Octave.src.Metrics.MetricSets.metric_set import AcVideoJepaMetricSet
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


def make_tiny_objective_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG)
    config["metric_set"]["metrics"]["idm_loss"]["inverse_dynamics_model"].update(
        {
            "hidden_dim": 16,
            "action_dim": 2,
        }
    )
    config["loss"]["metric_weights"].update(
        {
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
    jepa = AcVideoJepaModule(
        encoder=components["encoder"],
        action_encoder=components["action_encoder"],
        predictor=components["predictor"],
        encoder_shape=components["encoder_shape"],
        rollout=rollout,
        objective=torch.nn.Identity(),
        optimizer_builder=lambda parameters: torch.optim.AdamW(parameters, lr=1e-3),
        scheduler_builder=lambda optimizer: None,
    )
    return rollout(
        jepa=jepa,
        observations=torch.randn(2, 2, 4, 32, 32),
        actions=torch.randn(2, 2, 4),
    ), components["encoder_shape"]


def test_build_ac_video_jepa_objective_builds_from_plain_config() -> None:
    _, encoder_shape = make_rollout_output()

    objective = build_ac_video_jepa_objective(
        config=make_tiny_objective_config(),
        encoder_shape=encoder_shape,
    )

    assert isinstance(objective, AcVideoJepaObjective)
    assert isinstance(objective.metric_set, AcVideoJepaMetricSet)
    assert isinstance(objective.loss, WeightedMetricLoss)
    assert isinstance(
        objective.metric_set.metrics["prediction_loss"],
        AutoregressivePredictionLossMetric,
    )
    assert isinstance(objective.metric_set.metrics["std_loss"], HingeStdLossMetric)
    assert isinstance(objective.metric_set.metrics["cov_loss"], CovarianceLossMetric)
    assert isinstance(
        objective.metric_set.metrics["sim_loss_t"],
        TemporalSimilarityLossMetric,
    )
    assert isinstance(objective.metric_set.metrics["idm_loss"], InverseDynamicsLossMetric)


def test_build_ac_video_jepa_objective_does_not_mutate_input_config() -> None:
    _, encoder_shape = make_rollout_output()
    config = make_tiny_objective_config()
    original_config = deepcopy(config)

    build_ac_video_jepa_objective(
        config=config,
        encoder_shape=encoder_shape,
    )

    assert config == original_config


def test_ac_video_jepa_metric_set_returns_flat_metric_logs() -> None:
    rollout_output, encoder_shape = make_rollout_output()
    objective = build_ac_video_jepa_objective(
        config=make_tiny_objective_config(),
        encoder_shape=encoder_shape,
    )

    metric_values = objective.metric_set(
        rollout_output=rollout_output,
        actions=torch.randn(2, 2, 4),
    )

    assert metric_values["prediction_loss"].shape == torch.Size([])
    assert metric_values["std_loss"].shape == torch.Size([])
    assert metric_values["cov_loss"].shape == torch.Size([])
    assert metric_values["sim_loss_t"].shape == torch.Size([])
    assert metric_values["idm_loss"].shape == torch.Size([])
    assert {
        "prediction_loss",
        "std_loss",
        "cov_loss",
        "sim_loss_t",
        "idm_loss",
    } <= set(metric_values)


def test_ac_video_jepa_objective_returns_weighted_scalar_loss_and_logs() -> None:
    rollout_output, encoder_shape = make_rollout_output()
    objective = build_ac_video_jepa_objective(
        config=make_tiny_objective_config(),
        encoder_shape=encoder_shape,
    )

    loss, log_dict = objective(
        rollout_output=rollout_output,
        actions=torch.randn(2, 2, 4),
    )

    expected_loss = (
        log_dict["prediction_loss"]
        + 0.1 * log_dict["cov_loss"]
        + 0.1 * log_dict["std_loss"]
        + 0.1 * log_dict["sim_loss_t"]
        + 0.1 * log_dict["idm_loss"]
    )

    assert loss.shape == torch.Size([])
    assert log_dict["loss"] is loss
    assert torch.isclose(loss, expected_loss)
    assert torch.isclose(log_dict["loss/prediction_loss"], log_dict["prediction_loss"])
    assert torch.isclose(log_dict["loss/cov_loss"], 0.1 * log_dict["cov_loss"])
    assert torch.isclose(log_dict["loss/std_loss"], 0.1 * log_dict["std_loss"])
    assert torch.isclose(log_dict["loss/sim_loss_t"], 0.1 * log_dict["sim_loss_t"])
    assert torch.isclose(log_dict["loss/idm_loss"], 0.1 * log_dict["idm_loss"])


def test_ac_video_jepa_objective_rejects_unknown_top_level_key() -> None:
    _, encoder_shape = make_rollout_output()
    config = {
        **make_tiny_objective_config(),
        "unknown": {},
    }

    with pytest.raises(KeyError, match="Unknown AcVideoJepa objective config keys"):
        build_ac_video_jepa_objective(
            config=config,
            encoder_shape=encoder_shape,
        )


def test_ac_video_jepa_objective_rejects_unknown_nested_key() -> None:
    _, encoder_shape = make_rollout_output()
    config = make_tiny_objective_config()
    config["metric_set"]["metrics"]["std_loss"]["unknown"] = "bad"

    with pytest.raises(KeyError, match="Unknown AcVideoJepa std_loss config keys"):
        build_ac_video_jepa_objective(
            config=config,
            encoder_shape=encoder_shape,
        )


def test_ac_video_jepa_objective_rejects_unsupported_objective_type() -> None:
    _, encoder_shape = make_rollout_output()
    config = make_tiny_objective_config()
    config["objective_type"] = "unsupported"

    with pytest.raises(KeyError, match="Only 'ac_video_jepa'"):
        build_ac_video_jepa_objective(
            config=config,
            encoder_shape=encoder_shape,
        )


def test_ac_video_jepa_objective_rejects_missing_encoder_shape() -> None:
    with pytest.raises(ValueError, match="encoder_shape is required"):
        build_ac_video_jepa_objective(config=make_tiny_objective_config())


def test_ac_video_jepa_objective_rejects_unsupported_metric_type() -> None:
    _, encoder_shape = make_rollout_output()
    config = make_tiny_objective_config()
    config["metric_set"]["metrics"]["std_loss"]["metric_type"] = "unsupported"

    with pytest.raises(KeyError, match="Unknown AcVideoJepa metric_type"):
        build_ac_video_jepa_objective(
            config=config,
            encoder_shape=encoder_shape,
        )


def test_ac_video_jepa_objective_rejects_unsupported_prediction_cost_type() -> None:
    _, encoder_shape = make_rollout_output()
    config = make_tiny_objective_config()
    config["metric_set"]["metrics"]["prediction_loss"]["prediction_cost"][
        "prediction_cost_type"
    ] = "unsupported"

    with pytest.raises(KeyError, match="Only 'square_loss_seq'"):
        build_ac_video_jepa_objective(
            config=config,
            encoder_shape=encoder_shape,
        )


def test_ac_video_jepa_objective_rejects_prediction_projector_without_projector() -> None:
    _, encoder_shape = make_rollout_output()
    config = make_tiny_objective_config()
    config["metric_set"]["metrics"]["prediction_loss"]["prediction_cost"][
        "use_projector"
    ] = True

    with pytest.raises(ValueError, match="requires a projector"):
        build_ac_video_jepa_objective(
            config=config,
            encoder_shape=encoder_shape,
        )


def test_ac_video_jepa_objective_rejects_disabled_idm_metric() -> None:
    _, encoder_shape = make_rollout_output()
    config = make_tiny_objective_config()
    config["metric_set"]["metrics"]["idm_loss"]["inverse_dynamics_model"][
        "enabled"
    ] = False

    with pytest.raises(ValueError, match="requires inverse_dynamics_model.enabled"):
        build_ac_video_jepa_objective(
            config=config,
            encoder_shape=encoder_shape,
        )


def test_ac_video_jepa_objective_rejects_unsupported_loss_type() -> None:
    _, encoder_shape = make_rollout_output()
    config = make_tiny_objective_config()
    config["loss"]["loss_type"] = "unsupported"

    with pytest.raises(KeyError, match="Only 'weighted_metric'"):
        build_ac_video_jepa_objective(
            config=config,
            encoder_shape=encoder_shape,
        )
