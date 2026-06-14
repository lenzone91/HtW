from copy import deepcopy

from ..Model.ac_video_jepa.factory import build_ac_video_jepa_components
from ...Metrics.factory import build_ac_video_jepa_metric_stack
from ...Rollouts.factory import build_latent_rollout
from ...Training.Optimization.factory import build_optimizer_builder
from ...Training.Schedulers.factory import build_scheduler_builder
from ...Workflow.Factory.builder import RegistryBuilder
from . import ac_video_jepa_module  # noqa: F401
from .ac_video_jepa_module import AcVideoJepaModule
from .configs import DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG
from .registry import MODULE_REGISTRY


def build_ac_video_jepa_module(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> AcVideoJepaModule:
    module_config = prepare_ac_video_jepa_module_config(
        config=config or DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG,
        strict=strict,
    )

    components = build_ac_video_jepa_components(
        config=module_config["components_config"],
        runtime_context=runtime_context,
        strict=strict,
    )

    rollout = build_latent_rollout(
        config=module_config["rollout_config"],
        runtime_context=runtime_context,
        strict=strict,
    )

    metric_stack = build_ac_video_jepa_metric_stack(
        metric_set_config=module_config["metric_set_config"],
        loss_config=module_config["loss_config"],
        encoder_shape=components["encoder_shape"],
        runtime_context=runtime_context,
        strict=strict,
    )

    optimizer_builder = build_optimizer_builder(
        optimizer_config=module_config["optimizer_config"],
        strict=strict,
    )

    scheduler_builder = build_scheduler_builder(
        scheduler_config=module_config["scheduler_config"],
        strict=strict,
    )

    constructor_config = {
        "module_type": "ac_video_jepa",
        "encoder": components["encoder"],
        "action_encoder": components["action_encoder"],
        "predictor": components["predictor"],
        "encoder_shape": components["encoder_shape"],
        "rollout": rollout,
        "metric_set": metric_stack["metric_set"],
        "loss": metric_stack["loss"],
        "optimizer_builder": optimizer_builder,
        "scheduler_builder": scheduler_builder,
        "strict": module_config["strict"],
    }

    builder = RegistryBuilder(
        registry=MODULE_REGISTRY,
        strict=strict,
        type_field="module_type",
    )

    return builder.build_one(
        config=constructor_config,
        runtime_context=runtime_context,
    )


def prepare_ac_video_jepa_module_config(
    config: dict,
    strict: bool = True,
) -> dict:
    if not isinstance(config, dict):
        raise TypeError(
            "AcVideoJepa module config must be a dictionary, "
            f"got {type(config).__name__}."
        )

    prepared_config = deepcopy(DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG)
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