"""
Anti-collapse and auxiliary regularizer metrics for AcVideoJEPA.

These act on the encoded representation `encoded_states [B, C, T, H, W]` from a
`LatentRolloutOutput`. A shared mixin reshapes the 5D representation into the
sample / temporal layouts the primitives expect, optionally after an MLP
projector. The projector and the inverse-dynamics model depend on the encoder
output shape, which is read from `runtime_context["encoder_shape"]` at build time
(the module injects it after probing the encoder).

Uniform signature `forward(rollout_output, actions=None)`; only the
inverse-dynamics metric uses `actions`.
"""

import torch
from torch import nn

from ...AIML.Metrics.Metrics.registry import METRIC_REGISTRY
from ...Workflow.Factory.registry import FieldResolution
from ..Models.Backbones.inverse_dynamics import InverseDynamicsModel
from ..Models.Backbones.projector import Projector
from ..Models.Rollout.output import LatentRolloutOutput
from .primitives import (
    CovarianceLoss,
    HingeStdLoss,
    InverseDynamicsLoss,
    TemporalSimilarityLoss,
)

DEFAULT_PROJECTOR_CONFIG = {"enabled": False, "mlp_spec": None, "hidden_multiplier": 4}
DEFAULT_INVERSE_DYNAMICS_MODEL_CONFIG = {
    "enabled": True,
    "hidden_dim": 256,
    "action_dim": 2,
}


#############################################
# Field resolvers (encoder-shape dependent)
#############################################


def _require_encoder_shape(runtime_context: dict | None) -> dict:
    if not runtime_context or "encoder_shape" not in runtime_context:
        raise ValueError(
            "Building a regularizer projector / inverse-dynamics model requires "
            "runtime_context['encoder_shape'] (feature_dim/height/width)."
        )
    return runtime_context["encoder_shape"]


def resolve_projector(config, runtime_context=None, **kwargs) -> Projector | None:
    projector_config = config.get("projector")
    if projector_config is None or not projector_config["enabled"]:
        return None

    encoder_shape = _require_encoder_shape(runtime_context)
    encoder_dim = encoder_shape["feature_dim"]
    mlp_spec = projector_config["mlp_spec"]
    if mlp_spec is None:
        hidden_dim = encoder_dim * projector_config["hidden_multiplier"]
        mlp_spec = f"{encoder_dim}-{hidden_dim}-{hidden_dim}"
    return Projector(mlp_spec)


def resolve_inverse_dynamics_model(
    config, runtime_context=None, **kwargs
) -> InverseDynamicsModel | None:
    idm_config = config.get("inverse_dynamics_model")
    if idm_config is None or not idm_config["enabled"]:
        return None

    encoder_shape = _require_encoder_shape(runtime_context)
    # `projector` has already been resolved to a built object (or None) by the
    # time this runs (PROJECTOR_FIELD is declared first).
    projector = config.get("projector")
    if config["after_projection"] and projector is not None:
        state_channels = projector.out_dim
    else:
        state_channels = encoder_shape["feature_dim"]

    state_dim = encoder_shape["height"] * encoder_shape["width"] * state_channels
    return InverseDynamicsModel(
        state_dim=state_dim,
        hidden_dim=idm_config["hidden_dim"],
        action_dim=idm_config["action_dim"],
    )


PROJECTOR_FIELD = FieldResolution(target_key="projector", resolver=resolve_projector)
INVERSE_DYNAMICS_MODEL_FIELD = FieldResolution(
    target_key="inverse_dynamics_model",
    resolver=resolve_inverse_dynamics_model,
)


#############################################
# Shared 5D reshaping
#############################################


class RegularizerProjectionMixin:
    """Reshapes `encoded_states [B, C, T, H, W]` for the regularizer primitives."""

    def __init__(
        self,
        projector: nn.Module | None = None,
        first_t_only: bool = True,
        spatial_as_samples: bool = False,
        after_projection: bool = False,
    ) -> None:
        self.projector = nn.Identity() if projector is None else projector
        self.first_t_only = first_t_only
        self.spatial_as_samples = spatial_as_samples
        self.after_projection = after_projection

    def project_states(self, states: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = states.shape
        states_flat = states.permute(0, 2, 3, 4, 1).reshape(-1, channels)
        projected_flat = self.projector(states_flat)
        return projected_flat.view(batch, time, height, width, projected_flat.shape[-1])

    def vc_samples(self, states: torch.Tensor) -> torch.Tensor:
        batch, _, time, height, width = states.shape
        projected = self.project_states(states)
        projected_channels = projected.shape[-1]

        if self.spatial_as_samples:
            if self.first_t_only:
                return projected[:, 0].reshape(batch * height * width, projected_channels)
            return projected.reshape(-1, projected_channels)

        samples = projected.permute(0, 1, 4, 2, 3).reshape(batch, time, -1)
        if self.first_t_only:
            return samples[:, 0]
        return samples.reshape(-1, samples.size(-1))

    def temporal_states(self, states: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = states.shape
        if not self.after_projection:
            return states.permute(2, 0, 1, 3, 4).reshape(time, batch, -1)

        projected = self.project_states(states)
        projected_channels = projected.shape[-1]
        return projected.permute(1, 0, 4, 2, 3).reshape(
            time, batch, projected_channels * height * width
        )


#############################################
# Variance (hinge-std) regularizer
#############################################

DEFAULT_HINGE_STD_LOSS_CONFIG = {
    "projector": dict(DEFAULT_PROJECTOR_CONFIG),
    "std_margin": 1.0,
    "first_t_only": False,
    "spatial_as_samples": False,
}


@METRIC_REGISTRY.register_class(
    name="hinge_std",
    default_config=DEFAULT_HINGE_STD_LOSS_CONFIG,
    field_resolutions=(PROJECTOR_FIELD,),
)
class HingeStdLossMetric(RegularizerProjectionMixin, nn.Module):
    """Per-feature std hinge over the encoded representation (variance term)."""

    def __init__(
        self,
        projector: nn.Module | None = None,
        std_margin: float = 1.0,
        first_t_only: bool = True,
        spatial_as_samples: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        RegularizerProjectionMixin.__init__(
            self,
            projector=projector,
            first_t_only=first_t_only,
            spatial_as_samples=spatial_as_samples,
        )
        self.metric = HingeStdLoss(std_margin=std_margin)

    def forward(self, rollout_output: LatentRolloutOutput, actions=None) -> torch.Tensor:
        return self.metric(self.vc_samples(rollout_output.encoded_states))


#############################################
# Covariance regularizer
#############################################

DEFAULT_COVARIANCE_LOSS_CONFIG = {
    "projector": dict(DEFAULT_PROJECTOR_CONFIG),
    "first_t_only": False,
    "spatial_as_samples": False,
}


@METRIC_REGISTRY.register_class(
    name="covariance",
    default_config=DEFAULT_COVARIANCE_LOSS_CONFIG,
    field_resolutions=(PROJECTOR_FIELD,),
)
class CovarianceLossMetric(RegularizerProjectionMixin, nn.Module):
    """Off-diagonal covariance penalty over the encoded representation."""

    def __init__(
        self,
        projector: nn.Module | None = None,
        first_t_only: bool = True,
        spatial_as_samples: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        RegularizerProjectionMixin.__init__(
            self,
            projector=projector,
            first_t_only=first_t_only,
            spatial_as_samples=spatial_as_samples,
        )
        self.metric = CovarianceLoss()

    def forward(self, rollout_output: LatentRolloutOutput, actions=None) -> torch.Tensor:
        return self.metric(self.vc_samples(rollout_output.encoded_states))


#############################################
# Temporal similarity regularizer
#############################################

DEFAULT_TEMPORAL_SIMILARITY_LOSS_CONFIG = {
    "projector": dict(DEFAULT_PROJECTOR_CONFIG),
    "after_projection": False,
}


@METRIC_REGISTRY.register_class(
    name="temporal_similarity",
    default_config=DEFAULT_TEMPORAL_SIMILARITY_LOSS_CONFIG,
    field_resolutions=(PROJECTOR_FIELD,),
)
class TemporalSimilarityLossMetric(RegularizerProjectionMixin, nn.Module):
    """Penalizes change between consecutive encoded time steps."""

    def __init__(
        self,
        projector: nn.Module | None = None,
        after_projection: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        RegularizerProjectionMixin.__init__(
            self, projector=projector, after_projection=after_projection
        )
        self.metric = TemporalSimilarityLoss()

    def forward(self, rollout_output: LatentRolloutOutput, actions=None) -> torch.Tensor:
        return self.metric(self.temporal_states(rollout_output.encoded_states))


#############################################
# Inverse dynamics regularizer
#############################################

DEFAULT_INVERSE_DYNAMICS_LOSS_CONFIG = {
    "projector": dict(DEFAULT_PROJECTOR_CONFIG),
    "inverse_dynamics_model": dict(DEFAULT_INVERSE_DYNAMICS_MODEL_CONFIG),
    "after_projection": False,
}


@METRIC_REGISTRY.register_class(
    name="inverse_dynamics",
    default_config=DEFAULT_INVERSE_DYNAMICS_LOSS_CONFIG,
    field_resolutions=(PROJECTOR_FIELD, INVERSE_DYNAMICS_MODEL_FIELD),
)
class InverseDynamicsLossMetric(RegularizerProjectionMixin, nn.Module):
    """Action-prediction error from consecutive encoded states."""

    def __init__(
        self,
        inverse_dynamics_model: nn.Module,
        projector: nn.Module | None = None,
        after_projection: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        RegularizerProjectionMixin.__init__(
            self, projector=projector, after_projection=after_projection
        )
        self.metric = InverseDynamicsLoss(inverse_dynamics_model)

    def forward(self, rollout_output: LatentRolloutOutput, actions=None) -> torch.Tensor:
        return self.metric(self.temporal_states(rollout_output.encoded_states), actions)
