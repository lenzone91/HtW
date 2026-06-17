"""
JEPA objective loss primitives.

Re-implemented from EB-JEPA (no `eb_jepa` dependency, Decision 30). Pure,
framework-free math used by the AcVideoJEPA objective metrics:

- `SquareLossSeq` — MSE between two latent sequences (the prediction cost);
- `HingeStdLoss` — per-feature std hinge (anti-collapse variance term);
- `CovarianceLoss` — off-diagonal covariance penalty (decorrelation term);
- `TemporalSimilarityLoss` — penalizes change between consecutive time steps;
- `InverseDynamicsLoss` — action-prediction error from consecutive states.

Sample-matrix terms (`HingeStdLoss`, `CovarianceLoss`) expect `[N, D]`;
`TemporalSimilarityLoss` expects `[T, N, D]`; `InverseDynamicsLoss` expects
states `[T, B, D]` and actions `[B, A, T]`.
"""

import torch
import torch.nn.functional as F
from torch import nn

VARIANCE_EPS = 1e-4


class SquareLossSeq(nn.Module):
    """MSE between two latent sequences `[B, C, T, H, W]`, with optional projector."""

    def __init__(self, proj: nn.Module | None = None) -> None:
        super().__init__()
        self.proj = nn.Identity() if proj is None else proj

    def forward(self, state: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        state = self.proj(state.transpose(0, 1).flatten(1).transpose(0, 1))
        prediction = self.proj(prediction.transpose(0, 1).flatten(1).transpose(0, 1))
        return F.mse_loss(state, prediction)


class HingeStdLoss(nn.Module):
    """Hinge on per-feature standard deviation. Input `[N, D]`."""

    def __init__(self, std_margin: float = 1.0) -> None:
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + VARIANCE_EPS)
        return torch.mean(F.relu(self.std_margin - std))


class CovarianceLoss(nn.Module):
    """Off-diagonal covariance penalty. Input `[N, D]`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_samples = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        covariance = (x.T @ x) / (num_samples - 1)
        return self.off_diagonal(covariance).pow(2).mean()

    @staticmethod
    def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
        n, m = matrix.shape
        if n != m:
            raise ValueError(f"Expected a square matrix, got shape {(n, m)}.")
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class TemporalSimilarityLoss(nn.Module):
    """Penalizes squared change between consecutive time steps. Input `[T, N, D]`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 1:
            return torch.zeros((), device=x.device)
        return (x[1:] - x[:-1]).pow(2).mean()


class InverseDynamicsLoss(nn.Module):
    """
    Action-prediction error from consecutive states.

    States `[T, B, D]`, actions `[B, A, T]`. Predicts the action between each
    pair of consecutive states with `idm` and compares to the ground truth.
    """

    def __init__(self, idm: nn.Module) -> None:
        super().__init__()
        self.idm = idm

    def forward(self, x: torch.Tensor, actions: torch.Tensor | None) -> torch.Tensor:
        if x.shape[0] <= 1 or actions is None:
            return torch.zeros((), device=x.device)

        _, _, dim = x.shape
        states_t = x[:-1].transpose(0, 1).reshape(-1, dim)  # [B*(T-1), D]
        states_t_plus_1 = x[1:].transpose(0, 1).reshape(-1, dim)  # [B*(T-1), D]

        predicted_actions = self.idm(states_t, states_t_plus_1)  # [B*(T-1), A]
        target_actions = actions.transpose(1, 2)[:, :-1].reshape(-1, actions.size(1))
        return F.mse_loss(predicted_actions, target_actions)
