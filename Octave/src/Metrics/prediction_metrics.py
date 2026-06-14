import torch
from torch import nn

from Octave.src.Rollouts.latent_rollout import LatentRolloutOutput


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
