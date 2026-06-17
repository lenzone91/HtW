"""
Prediction-loss metrics: the core JEPA signal for AcVideoJEPA.

The latent prediction loss compares the rolled-out predicted states to the
encoded states, in latent space. Autoregressive and parallel rollouts align
their targets differently, so each has its own metric. Both delegate the latent
distance to a sub-built `prediction_cost` (e.g. `square_loss_seq`).

All metrics share the uniform AcVideoJEPA signature
`forward(rollout_output, actions=None)`: every metric receives the whole
`LatentRolloutOutput` (and the raw actions, which most ignore), so the module
can pass the same inputs to the entire metric set.
"""

import torch
from torch import nn

from ...AIML.Metrics.Metrics.registry import METRIC_REGISTRY
from ..Models.Rollout.output import LatentRolloutOutput
from .primitives import SquareLossSeq
from .registry import PREDICTION_COST_REGISTRY, PREDICTION_COST_SUB_BUILD


#############################################
# Prediction cost (sub-built)
#############################################

DEFAULT_SQUARE_LOSS_SEQ_COST_CONFIG = {"use_projector": False, "proj": None}


@PREDICTION_COST_REGISTRY.register_class(
    name="square_loss_seq",
    default_config=DEFAULT_SQUARE_LOSS_SEQ_COST_CONFIG,
    type_field="prediction_cost_type",
)
class SquareLossSeqPredictionCost(SquareLossSeq):
    """
    Square-loss prediction cost. `use_projector` is kept for config
    compatibility; using a projector requires an explicit `proj` object.
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


# Sub-config the prediction metrics carry for their prediction_cost (includes the
# routing field, which the builder strips before construction; Decision 24).
DEFAULT_PREDICTION_COST_SUBCONFIG = {
    "prediction_cost_type": "square_loss_seq",
    "use_projector": False,
    "proj": None,
}


#############################################
# Autoregressive prediction loss
#############################################

DEFAULT_AUTOREGRESSIVE_PREDICTION_LOSS_CONFIG = {
    "prediction_cost": dict(DEFAULT_PREDICTION_COST_SUBCONFIG),
}


@METRIC_REGISTRY.register_class(
    name="autoregressive_prediction_loss",
    default_config=DEFAULT_AUTOREGRESSIVE_PREDICTION_LOSS_CONFIG,
    sub_builds=(PREDICTION_COST_SUB_BUILD,),
)
class AutoregressivePredictionLossMetric(nn.Module):
    """Prediction loss for autoregressive latent rollouts."""

    expected_unroll_mode = "autoregressive"

    def __init__(self, prediction_cost: nn.Module) -> None:
        super().__init__()
        self.prediction_cost = prediction_cost

    def forward(
        self,
        rollout_output: LatentRolloutOutput,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        check_rollout_mode(rollout_output, self.expected_unroll_mode)
        context = rollout_output.effective_ctxt_window
        target_end = context + rollout_output.nsteps
        targets = rollout_output.encoded_states[:, :, context:target_end]
        predictions = rollout_output.predicted_states[:, :, context:target_end]
        return self.prediction_cost(targets, predictions)


#############################################
# Parallel prediction loss
#############################################

DEFAULT_PARALLEL_PREDICTION_LOSS_CONFIG = {
    "prediction_cost": dict(DEFAULT_PREDICTION_COST_SUBCONFIG),
}


@METRIC_REGISTRY.register_class(
    name="parallel_prediction_loss",
    default_config=DEFAULT_PARALLEL_PREDICTION_LOSS_CONFIG,
    sub_builds=(PREDICTION_COST_SUB_BUILD,),
)
class ParallelPredictionLossMetric(nn.Module):
    """Prediction loss for parallel latent rollouts."""

    expected_unroll_mode = "parallel"

    def __init__(self, prediction_cost: nn.Module) -> None:
        super().__init__()
        self.prediction_cost = prediction_cost

    def forward(
        self,
        rollout_output: LatentRolloutOutput,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        check_rollout_mode(rollout_output, self.expected_unroll_mode)

        if not rollout_output.predicted_steps:
            return self.prediction_cost(
                rollout_output.encoded_states,
                rollout_output.predicted_states,
            )

        context = rollout_output.effective_ctxt_window
        n_steps = len(rollout_output.predicted_steps)
        pred_loss = rollout_output.encoded_states.new_zeros(())
        for predicted_step in rollout_output.predicted_steps:
            predicted_states = torch.cat(
                (rollout_output.encoded_states[:, :, :context], predicted_step),
                dim=2,
            )
            pred_loss = pred_loss + self.prediction_cost(
                rollout_output.encoded_states, predicted_states
            ) / n_steps
        return pred_loss


def check_rollout_mode(rollout_output: LatentRolloutOutput, expected: str) -> None:
    if rollout_output.unroll_mode != expected:
        raise ValueError(
            f"Prediction metric expected rollout mode '{expected}', got "
            f"'{rollout_output.unroll_mode}'."
        )
