from copy import deepcopy

from eb_jepa.architectures import InverseDynamicsModel, Projector
from eb_jepa.losses import SquareLossSeq

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
from .ac_video_jepa_objective import AcVideoJepaObjective
from .configs import DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG
from .Loss.loss import WeightedMetricLoss
from .MetricSets.metric_set import AcVideoJepaMetricSet


DEFAULT_PREDICTION_LOSS_METRIC_CONFIG = (
    DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG["metric_set"]["metrics"]["prediction_loss"]
)
DEFAULT_STD_LOSS_METRIC_CONFIG = (
    DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG["metric_set"]["metrics"]["std_loss"]
)
DEFAULT_COV_LOSS_METRIC_CONFIG = (
    DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG["metric_set"]["metrics"]["cov_loss"]
)
DEFAULT_SIM_LOSS_METRIC_CONFIG = (
    DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG["metric_set"]["metrics"]["sim_loss_t"]
)
DEFAULT_IDM_LOSS_METRIC_CONFIG = (
    DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG["metric_set"]["metrics"]["idm_loss"]
)
DEFAULT_METRIC_SET_CONFIG = DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG["metric_set"]
DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG = DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG["loss"]


class AcVideoJepaObjectiveBuilder:
    """
    Build AcVideoJepaObjective from plain dictionary config.
    """

    def __init__(
        self,
        default_config: dict | None = None,
        strict: bool = True,
    ) -> None:
        self.default_config = deepcopy(
            default_config
            if default_config is not None
            else DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG
        )
        self.strict = strict

    def __call__(
        self,
        config: dict,
        encoder_shape: dict,
        runtime_context: dict | None = None,
    ) -> AcVideoJepaObjective:
        prepared_config = self.prepare_config(config)
        metric_set = self.build_metric_set(
            metric_set_config=prepared_config["metric_set"],
            encoder_shape=encoder_shape,
        )
        loss = self.build_loss(prepared_config["loss"])
        return AcVideoJepaObjective(
            metric_set=metric_set,
            loss=loss,
        )

    def build_metric_set(
        self,
        metric_set_config: dict,
        encoder_shape: dict,
    ) -> AcVideoJepaMetricSet:
        set_type = metric_set_config["set_type"]
        if set_type != "ac_video_jepa":
            raise KeyError("Only 'ac_video_jepa' metric set_type is supported.")

        metrics = {
            metric_name: self.build_metric(
                metric_name=metric_name,
                metric_config=metric_config,
                encoder_shape=encoder_shape,
            )
            for metric_name, metric_config in metric_set_config["metrics"].items()
        }
        return AcVideoJepaMetricSet(
            strict=metric_set_config["strict"],
            metric_to_input_names=metric_set_config["metric_to_input_names"],
            **metrics,
        )

    def build_metric(
        self,
        metric_name: str,
        metric_config: dict,
        encoder_shape: dict,
    ):
        metric_type = metric_config["metric_type"]

        if metric_type in {
            "autoregressive_prediction_loss",
            "parallel_prediction_loss",
        }:
            return self.build_prediction_loss_metric(metric_config)

        if metric_type == "hinge_std":
            return self.build_std_loss_metric(metric_config, encoder_shape)

        if metric_type == "covariance":
            return self.build_cov_loss_metric(metric_config, encoder_shape)

        if metric_type == "temporal_similarity":
            return self.build_sim_loss_metric(metric_config, encoder_shape)

        if metric_type == "inverse_dynamics":
            return self.build_idm_loss_metric(metric_config, encoder_shape)

        raise KeyError(
            f"Unknown AcVideoJepa metric_type for {metric_name}: {metric_type}."
        )

    def build_prediction_loss_metric(
        self,
        metric_config: dict,
    ) -> AutoregressivePredictionLossMetric | ParallelPredictionLossMetric:
        prediction_cost = self.build_prediction_cost(
            prediction_cost_config=metric_config["prediction_cost"],
            projector=None,
        )
        if metric_config["metric_type"] == "autoregressive_prediction_loss":
            return AutoregressivePredictionLossMetric(prediction_cost=prediction_cost)

        return ParallelPredictionLossMetric(prediction_cost=prediction_cost)

    def build_std_loss_metric(
        self,
        metric_config: dict,
        encoder_shape: dict,
    ) -> HingeStdLossMetric:
        projector = self.build_projector(
            projector_config=metric_config["projector"],
            encoder_dim=encoder_shape["feature_dim"],
        )
        return HingeStdLossMetric(
            projector=projector,
            std_margin=metric_config["std_margin"],
            first_t_only=metric_config["first_t_only"],
            spatial_as_samples=metric_config["spatial_as_samples"],
        )

    def build_cov_loss_metric(
        self,
        metric_config: dict,
        encoder_shape: dict,
    ) -> CovarianceLossMetric:
        projector = self.build_projector(
            projector_config=metric_config["projector"],
            encoder_dim=encoder_shape["feature_dim"],
        )
        return CovarianceLossMetric(
            projector=projector,
            first_t_only=metric_config["first_t_only"],
            spatial_as_samples=metric_config["spatial_as_samples"],
        )

    def build_sim_loss_metric(
        self,
        metric_config: dict,
        encoder_shape: dict,
    ) -> TemporalSimilarityLossMetric:
        projector = self.build_projector(
            projector_config=metric_config["projector"],
            encoder_dim=encoder_shape["feature_dim"],
        )
        return TemporalSimilarityLossMetric(
            projector=projector,
            after_projection=metric_config["after_projection"],
        )

    def build_idm_loss_metric(
        self,
        metric_config: dict,
        encoder_shape: dict,
    ) -> InverseDynamicsLossMetric:
        projector = self.build_projector(
            projector_config=metric_config["projector"],
            encoder_dim=encoder_shape["feature_dim"],
        )
        idm = self.build_inverse_dynamics_model(
            idm_config=metric_config["inverse_dynamics_model"],
            projector=projector,
            encoder_shape=encoder_shape,
            after_projection=metric_config["after_projection"],
        )
        if idm is None:
            raise ValueError(
                "inverse_dynamics metric requires inverse_dynamics_model.enabled=True. "
                "Remove the metric from the metric set to disable it."
            )

        return InverseDynamicsLossMetric(
            inverse_dynamics_model=idm,
            projector=projector,
            after_projection=metric_config["after_projection"],
        )

    def build_loss(self, loss_config: dict) -> WeightedMetricLoss:
        loss_type = loss_config["loss_type"]
        if loss_type != "weighted_metric":
            raise KeyError("Only 'weighted_metric' loss_type is supported.")

        return WeightedMetricLoss(
            metric_weights=loss_config["metric_weights"],
            strict=loss_config["strict"],
            name=loss_config["name"],
            return_loss_terms=loss_config["return_loss_terms"],
        )

    def prepare_config(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise TypeError(
                "AcVideoJepa objective config must be a dictionary, "
                f"got {type(config).__name__}."
            )

        prepared_config = deepcopy(self.default_config)
        user_config = deepcopy(config)

        self.merge_config(
            target=prepared_config,
            override=user_config,
            section_name="objective",
        )

        self.check_objective_type(prepared_config)
        self.check_metric_set_config(prepared_config["metric_set"])
        self.check_loss_config(prepared_config["loss"])
        return prepared_config

    def merge_config(
        self,
        target: dict,
        override: dict,
        section_name: str,
    ) -> None:
        if section_name in {"metrics", "metric_weights"}:
            target.update(deepcopy(override))
            return

        self.check_known_keys(override, target, section_name=section_name)

        for key, value in override.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self.merge_config(
                    target=target[key],
                    override=value,
                    section_name=key,
                )
            else:
                target[key] = value

    def check_metric_set_config(self, metric_set_config: dict) -> None:
        self.check_known_keys(
            metric_set_config,
            DEFAULT_METRIC_SET_CONFIG,
            section_name="metric_set",
        )

        for metric_name, metric_config in metric_set_config["metrics"].items():
            metric_type = metric_config.get("metric_type")
            if metric_type in {
                "autoregressive_prediction_loss",
                "parallel_prediction_loss",
            }:
                self.check_known_keys(
                    metric_config,
                    DEFAULT_PREDICTION_LOSS_METRIC_CONFIG,
                    section_name=metric_name,
                )
                self.check_known_keys(
                    metric_config["prediction_cost"],
                    DEFAULT_PREDICTION_LOSS_METRIC_CONFIG["prediction_cost"],
                    section_name="prediction_cost",
                )
            elif metric_type == "hinge_std":
                self.check_regularizer_metric_config(
                    metric_config=metric_config,
                    default_config=DEFAULT_STD_LOSS_METRIC_CONFIG,
                    metric_name=metric_name,
                )
            elif metric_type == "covariance":
                self.check_regularizer_metric_config(
                    metric_config=metric_config,
                    default_config=DEFAULT_COV_LOSS_METRIC_CONFIG,
                    metric_name=metric_name,
                )
            elif metric_type == "temporal_similarity":
                self.check_regularizer_metric_config(
                    metric_config=metric_config,
                    default_config=DEFAULT_SIM_LOSS_METRIC_CONFIG,
                    metric_name=metric_name,
                )
            elif metric_type == "inverse_dynamics":
                self.check_regularizer_metric_config(
                    metric_config=metric_config,
                    default_config=DEFAULT_IDM_LOSS_METRIC_CONFIG,
                    metric_name=metric_name,
                )
                self.check_known_keys(
                    metric_config["inverse_dynamics_model"],
                    DEFAULT_IDM_LOSS_METRIC_CONFIG["inverse_dynamics_model"],
                    section_name="inverse_dynamics_model",
                )
            else:
                raise KeyError(
                    f"Unknown AcVideoJepa metric_type for {metric_name}: "
                    f"{metric_type}."
                )

    def check_regularizer_metric_config(
        self,
        metric_config: dict,
        default_config: dict,
        metric_name: str,
    ) -> None:
        self.check_known_keys(
            metric_config,
            default_config,
            section_name=metric_name,
        )
        self.check_known_keys(
            metric_config["projector"],
            default_config["projector"],
            section_name="projector",
        )

    def check_loss_config(self, loss_config: dict) -> None:
        self.check_known_keys(
            loss_config,
            DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG,
            section_name="loss",
        )

    def build_projector(
        self,
        projector_config: dict,
        encoder_dim: int,
    ) -> Projector | None:
        if not projector_config["enabled"]:
            return None

        mlp_spec = projector_config["mlp_spec"]
        if mlp_spec is None:
            hidden_dim = encoder_dim * projector_config["hidden_multiplier"]
            mlp_spec = f"{encoder_dim}-{hidden_dim}-{hidden_dim}"

        return Projector(mlp_spec)

    def build_inverse_dynamics_model(
        self,
        idm_config: dict,
        projector: Projector | None,
        encoder_shape: dict,
        after_projection: bool,
    ) -> InverseDynamicsModel | None:
        if not idm_config["enabled"]:
            return None

        if after_projection and projector is not None:
            state_channels = projector.out_dim
        else:
            state_channels = encoder_shape["feature_dim"]

        state_dim = (
            encoder_shape["height"]
            * encoder_shape["width"]
            * state_channels
        )

        return InverseDynamicsModel(
            state_dim=state_dim,
            hidden_dim=idm_config["hidden_dim"],
            action_dim=idm_config["action_dim"],
        )

    def build_prediction_cost(
        self,
        prediction_cost_config: dict,
        projector: Projector | None,
    ) -> SquareLossSeq:
        prediction_cost_type = prediction_cost_config["prediction_cost_type"]
        if prediction_cost_type != "square_loss_seq":
            raise KeyError(
                "Only 'square_loss_seq' prediction_cost_type is supported "
                "for AcVideoJepa."
            )

        if prediction_cost_config["use_projector"] and projector is None:
            raise ValueError(
                "prediction_cost.use_projector=True requires a projector, but "
                "prediction metrics do not currently own a projector."
            )

        proj = projector if prediction_cost_config["use_projector"] else None
        return SquareLossSeq(proj=proj)

    def check_objective_type(self, config: dict) -> None:
        if config["objective_type"] != "ac_video_jepa":
            raise KeyError("Only 'ac_video_jepa' objective_type is supported.")

    def check_known_keys(
        self,
        config: dict,
        default_config: dict,
        section_name: str,
    ) -> None:
        unknown_keys = set(config) - set(default_config)

        if not unknown_keys:
            return

        message = (
            f"Unknown AcVideoJepa {section_name} config keys: "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(default_config)}."
        )

        if self.strict:
            raise KeyError(message)


def build_ac_video_jepa_objective(
    config: dict | None = None,
    encoder_shape: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> AcVideoJepaObjective:
    if encoder_shape is None:
        raise ValueError("encoder_shape is required to build AcVideoJepaObjective.")

    builder = AcVideoJepaObjectiveBuilder(strict=strict)
    return builder(
        config=config or DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG,
        encoder_shape=encoder_shape,
        runtime_context=runtime_context,
    )
