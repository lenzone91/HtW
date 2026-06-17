import pytest
import torch

from eb_jepa_cleaned.AcVideoJEPA.Metrics.primitives import (
    CovarianceLoss,
    HingeStdLoss,
    InverseDynamicsLoss,
    SquareLossSeq,
    TemporalSimilarityLoss,
)
from eb_jepa_cleaned.AcVideoJEPA.Models.Backbones.inverse_dynamics import (
    InverseDynamicsModel,
)


def test_square_loss_seq_zero_and_mse():
    x = torch.randn(2, 4, 3, 1, 1)
    cost = SquareLossSeq()
    assert cost(x, x).item() == pytest.approx(0.0)
    assert cost(torch.zeros(2, 4, 3, 1, 1), torch.ones(2, 4, 3, 1, 1)).item() == (
        pytest.approx(1.0)
    )


def test_hinge_std_collapse_and_spread():
    assert HingeStdLoss(1.0)(torch.zeros(16, 4)).item() == pytest.approx(1.0, abs=1e-2)
    torch.manual_seed(0)
    assert HingeStdLoss(1.0)(torch.randn(2000, 4) * 10).item() == pytest.approx(
        0.0, abs=1e-4
    )


def test_covariance_decorrelated_small():
    torch.manual_seed(0)
    assert CovarianceLoss()(torch.randn(5000, 5)).item() < 0.05


def test_temporal_similarity_zero_for_constant_in_time():
    x = torch.randn(1, 2, 4)  # single time step
    assert TemporalSimilarityLoss()(x).item() == pytest.approx(0.0)
    constant = torch.randn(2, 4).unsqueeze(0).repeat(3, 1, 1)  # [T=3, N, D] constant
    assert TemporalSimilarityLoss()(constant).item() == pytest.approx(0.0)


def test_inverse_dynamics_loss_runs():
    idm = InverseDynamicsModel(state_dim=4, hidden_dim=8, action_dim=2)
    states = torch.randn(3, 2, 4)  # [T, B, D]
    actions = torch.randn(2, 2, 3)  # [B, A, T]
    loss = InverseDynamicsLoss(idm)(states, actions)
    assert loss.ndim == 0 and loss.item() >= 0.0
    # No-op when no actions or single step.
    assert InverseDynamicsLoss(idm)(states, None).item() == pytest.approx(0.0)
