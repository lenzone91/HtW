import torch
from torch import nn

from eb_jepa.losses import CovarianceLoss, HingeStdLoss, InverseDynamicsLoss
from eb_jepa.losses import TemporalSimilarityLoss
from Octave.src.Rollouts.latent_rollout import LatentRolloutOutput


class RegularizerProjectionMixin:
    """
    Shared representation preparation for EB-JEPA regularizer metric wrappers.
    """

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
        projected_channels = projected_flat.shape[-1]
        return projected_flat.view(
            batch,
            time,
            height,
            width,
            projected_channels,
        )

    def vc_samples(self, states: torch.Tensor) -> torch.Tensor:
        batch, _, time, height, width = states.shape
        projected = self.project_states(states)
        projected_channels = projected.shape[-1]

        if self.spatial_as_samples:
            if self.first_t_only:
                return projected[:, 0].reshape(
                    batch * height * width,
                    projected_channels,
                )
            return projected.reshape(-1, projected_channels)

        samples = projected.permute(0, 1, 4, 2, 3).reshape(batch, time, -1)
        if self.first_t_only:
            return samples[:, 0]

        return samples.reshape(-1, samples.size(-1))

    def temporal_states(self, states: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = states.shape

        if not self.after_projection:
            return states.permute(2, 0, 1, 3, 4).reshape(
                time,
                batch,
                -1,
            )

        projected = self.project_states(states)
        projected_channels = projected.shape[-1]
        return projected.permute(1, 0, 4, 2, 3).reshape(
            time,
            batch,
            projected_channels * height * width,
        )


class HingeStdLossMetric(RegularizerProjectionMixin, nn.Module):
    """
    EB-JEPA hinge standard-deviation regularizer as one metric.
    """

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

    def forward(self, rollout_output: LatentRolloutOutput) -> torch.Tensor:
        return self.metric(self.vc_samples(rollout_output.encoded_states))


class CovarianceLossMetric(RegularizerProjectionMixin, nn.Module):
    """
    EB-JEPA covariance regularizer as one metric.
    """

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

    def forward(self, rollout_output: LatentRolloutOutput) -> torch.Tensor:
        return self.metric(self.vc_samples(rollout_output.encoded_states))


class TemporalSimilarityLossMetric(RegularizerProjectionMixin, nn.Module):
    """
    EB-JEPA temporal similarity regularizer as one metric.
    """

    def __init__(
        self,
        projector: nn.Module | None = None,
        after_projection: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        RegularizerProjectionMixin.__init__(
            self,
            projector=projector,
            after_projection=after_projection,
        )
        self.metric = TemporalSimilarityLoss()

    def forward(self, rollout_output: LatentRolloutOutput) -> torch.Tensor:
        return self.metric(self.temporal_states(rollout_output.encoded_states))


class InverseDynamicsLossMetric(RegularizerProjectionMixin, nn.Module):
    """
    EB-JEPA inverse dynamics regularizer as one metric.
    """

    def __init__(
        self,
        inverse_dynamics_model: nn.Module,
        projector: nn.Module | None = None,
        after_projection: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        RegularizerProjectionMixin.__init__(
            self,
            projector=projector,
            after_projection=after_projection,
        )
        self.metric = InverseDynamicsLoss(inverse_dynamics_model)

    def forward(
        self,
        rollout_output: LatentRolloutOutput,
        actions: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.metric(
            self.temporal_states(rollout_output.encoded_states),
            actions,
        )
