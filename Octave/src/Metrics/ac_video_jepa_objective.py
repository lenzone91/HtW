import torch
from torch import nn

from Octave.src.Rollouts.latent_rollout import LatentRolloutOutput


class AcVideoJepaObjective(nn.Module):
    """
    AcVideoJepa metric evaluation plus weighted metric loss.
    """

    def __init__(
        self,
        metric_set: nn.Module,
        loss: nn.Module,
    ) -> None:
        super().__init__()
        self.metric_set = metric_set
        self.loss = loss

    def forward(
        self,
        rollout_output: LatentRolloutOutput,
        actions=None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        metric_values = self.metric_set(
            rollout_output=rollout_output,
            actions=actions,
        )
        loss, loss_logs = self.loss(metric_values)

        log_dict = {
            **metric_values,
            **loss_logs,
        }
        return loss, log_dict
