from copy import deepcopy

from ..Model.ac_video_jepa.factory import build_ac_video_jepa_components
from ...Metrics.factory import build_ac_video_jepa_objective
from ...Rollouts.factory import build_latent_rollout
from ...Training.Optimization.factory import build_optimizer_builder
from ...Training.Schedulers.factory import build_scheduler_builder
from .ac_video_jepa_module import AcVideoJepaModule
from .configs import DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG


class AcVideoJepaModuleBuilder:
    """
    Build AcVideoJepaModule from a plain dictionary config.
    """

    def __init__(
        self,
        default_config: dict | None = None,
        strict: bool = True,
    ) -> None:
        self.default_config = deepcopy(
            default_config
            if default_config is not None
            else DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG
        )
        self.strict = strict

    def __call__(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> AcVideoJepaModule:
        prepared_config = self.prepare_config(config)

        components = build_ac_video_jepa_components(
            config=prepared_config["components_config"],
            runtime_context=runtime_context,
            strict=self.strict,
        )
        rollout = build_latent_rollout(
            config=prepared_config["rollout_config"],
            runtime_context=runtime_context,
            strict=self.strict,
        )
        objective = build_ac_video_jepa_objective(
            config=prepared_config["objective_config"],
            encoder_shape=components["encoder_shape"],
            runtime_context=runtime_context,
            strict=self.strict,
        )
        optimizer_builder = build_optimizer_builder(
            optimizer_config=prepared_config["optimizer_config"],
        )
        scheduler_builder = build_scheduler_builder(
            scheduler_config=prepared_config["scheduler_config"],
        )

        return AcVideoJepaModule(
            encoder=components["encoder"],
            action_encoder=components["action_encoder"],
            predictor=components["predictor"],
            encoder_shape=components["encoder_shape"],
            rollout=rollout,
            objective=objective,
            optimizer_builder=optimizer_builder,
            scheduler_builder=scheduler_builder,
            strict=self.strict,
        )

    def prepare_config(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise TypeError(
                "AcVideoJepa module config must be a dictionary, "
                f"got {type(config).__name__}."
            )

        prepared_config = deepcopy(self.default_config)
        user_config = deepcopy(config)

        unknown_keys = set(user_config) - set(prepared_config)
        if unknown_keys:
            raise KeyError(
                "Unknown AcVideoJepa module config keys: "
                f"{sorted(unknown_keys)}. "
                f"Allowed keys are: {sorted(prepared_config)}."
            )

        for key, value in user_config.items():
            if isinstance(value, dict) and isinstance(prepared_config.get(key), dict):
                prepared_config[key].update(value)
            else:
                prepared_config[key] = value

        if prepared_config["module_type"] != "ac_video_jepa":
            raise KeyError("Only 'ac_video_jepa' module_type is supported.")

        return prepared_config


def build_ac_video_jepa_module(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> AcVideoJepaModule:
    builder = AcVideoJepaModuleBuilder(strict=strict)
    return builder(
        config=config or DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG,
        runtime_context=runtime_context,
    )
