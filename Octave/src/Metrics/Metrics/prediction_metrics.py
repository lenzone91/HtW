import torch
from torch import nn

from Octave.src.Rollouts.latent_rollout import LatentRolloutOutput

from eb_jepa.losses import SquareLossSeq



from .configs import (
    DEFAULT_AUTOREGRESSIVE_PREDICTION_LOSS_METRIC_CONFIG,
    DEFAULT_PARALLEL_PREDICTION_LOSS_METRIC_CONFIG,
    DEFAULT_SQUARE_LOSS_SEQ_CONFIG
)
from .registry import (
    METRIC_REGISTRY, 
    PREDICTION_COST_SUB_BUILD, 
    PREDICTION_COST_REGISTRY
)


@METRIC_REGISTRY.register_class(
    name="autoregressive_prediction_loss",
    default_config=DEFAULT_AUTOREGRESSIVE_PREDICTION_LOSS_METRIC_CONFIG,
    type_field="metric_type",
    sub_builds=(PREDICTION_COST_SUB_BUILD,),
)
class AutoregressivePredictionLossMetric(nn.Module):
    """
    Prediction loss for autoregressive latent rollouts.
    """

    expected_unroll_mode = "autoregressive"

    def __init__(self, prediction_cost: nn.Module) -> None:
        super().__init__()
        self.prediction_cost = prediction_cost

    def forward(self, rollout_output: LatentRolloutOutput) -> torch.Tensor:
        self.check_rollout_mode(rollout_output)
        context = rollout_output.effective_ctxt_window
        target_end = context + rollout_output.nsteps
        targets = rollout_output.encoded_states[:, :, context:target_end]
        predictions = rollout_output.predicted_states[:, :, context:target_end]
        return self.prediction_cost(targets, predictions)

    def check_rollout_mode(self, rollout_output: LatentRolloutOutput) -> None:
        if rollout_output.unroll_mode != self.expected_unroll_mode:
            raise ValueError(
                f"{self.__class__.__name__} expected rollout mode "
                f"'{self.expected_unroll_mode}', got "
                f"'{rollout_output.unroll_mode}'."
            )



@METRIC_REGISTRY.register_class(
    name="parallel_prediction_loss",
    default_config=DEFAULT_PARALLEL_PREDICTION_LOSS_METRIC_CONFIG,
    type_field="metric_type",
    sub_builds=(PREDICTION_COST_SUB_BUILD,),
)
class ParallelPredictionLossMetric(nn.Module):
    """
    Prediction loss for parallel latent rollouts.
    """

    expected_unroll_mode = "parallel"

    def __init__(self, prediction_cost: nn.Module) -> None:
        super().__init__()
        self.prediction_cost = prediction_cost

    def forward(self, rollout_output: LatentRolloutOutput) -> torch.Tensor:
        self.check_rollout_mode(rollout_output)

        if not rollout_output.predicted_steps:
            return self.prediction_cost(
                rollout_output.encoded_states,
                rollout_output.predicted_states,
            )

        pred_loss = 0.0
        context = rollout_output.effective_ctxt_window
        for predicted_step in rollout_output.predicted_steps:
            predicted_states = torch.cat(
                (
                    rollout_output.encoded_states[:, :, :context],
                    predicted_step,
                ),
                dim=2,
            )
            pred_loss = pred_loss + (
                self.prediction_cost(
                    rollout_output.encoded_states,
                    predicted_states,
                )
                / len(rollout_output.predicted_steps)
            )

        return pred_loss

    def check_rollout_mode(self, rollout_output: LatentRolloutOutput) -> None:
        if rollout_output.unroll_mode != self.expected_unroll_mode:
            raise ValueError(
                f"{self.__class__.__name__} expected rollout mode "
                f"'{self.expected_unroll_mode}', got "
                f"'{rollout_output.unroll_mode}'."
            )


@PREDICTION_COST_REGISTRY.register_class(
    name="square_loss_seq",
    default_config=DEFAULT_SQUARE_LOSS_SEQ_CONFIG,
    type_field="prediction_cost_type",
)
class SquareLossSeqPredictionCost(SquareLossSeq):
    """
    Registered SquareLossSeq adapter.

    The use_projector flag is kept for config compatibility. Prediction
    metrics currently receive no metric-owned projector, so projector usage
    requires an explicit proj object.
    """

    def __init__(
        self,
        use_projector: bool = False,
        proj: nn.Module | None = None,
    ) -> None:
        if use_projector and proj is None:
            raise ValueError(
                "prediction_cost.use_projector=True requires an explicit proj."
            )

        super().__init__(proj=proj if use_projector else None)