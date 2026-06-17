"""
Latent rollout: shape/mode contracts (autoregressive with a real RNN predictor,
parallel with a stub predictor), config validation, and factory building.
"""

from types import SimpleNamespace

import pytest
import torch

from src.AcVideoJEPA.Models.Backbones.rnn_predictor import RNNPredictor
from src.AcVideoJEPA.Models.Rollout.factory import build_rollout
from src.AcVideoJEPA.Models.Rollout.latent_rollout import LatentRollout
from src.AcVideoJEPA.Models.Rollout.output import LatentRolloutOutput


class FixedEncoderJepa:
    """Minimal JEPA runtime stub: fixed encoded states + an injected predictor."""

    def __init__(self, encoded, predictor, predict_fn):
        self._encoded = encoded
        self.predictor = predictor
        self._predict_fn = predict_fn

    def encode(self, observations):
        return self._encoded

    def encode_actions(self, actions):
        return actions

    def predict(self, states, actions):
        return self._predict_fn(states, actions)


#############################################
# Autoregressive (real RNN predictor)
#############################################


def test_autoregressive_rollout_grows_time_by_nsteps():
    b, d, t, a = 2, 4, 4, 2
    encoded = torch.randn(b, d, t, 1, 1)
    predictor = RNNPredictor(hidden_size=d, action_dim=a, num_layers=1)
    jepa = FixedEncoderJepa(encoded, predictor, predictor)
    actions = torch.randn(b, a, t)

    rollout = LatentRollout(nsteps=2, unroll_mode="autoregressive")
    out = rollout(jepa, observations=None, actions=actions)

    assert isinstance(out, LatentRolloutOutput)
    assert out.unroll_mode == "autoregressive"
    assert out.effective_ctxt_window == 1  # RNN predictor
    assert out.nsteps == 2
    # starts with 1 context frame, appends nsteps predictions
    assert out.predicted_states.shape == (b, d, 1 + 2, 1, 1)


def test_autoregressive_rejects_nsteps_over_action_length():
    b, d, a = 2, 4, 2
    encoded = torch.randn(b, d, 3, 1, 1)
    predictor = RNNPredictor(hidden_size=d, action_dim=a)
    jepa = FixedEncoderJepa(encoded, predictor, predictor)
    actions = torch.randn(b, a, 1)  # only 1 step available

    rollout = LatentRollout(nsteps=2, unroll_mode="autoregressive")
    with pytest.raises(ValueError):
        rollout(jepa, observations=None, actions=actions)


#############################################
# Parallel (stub identity predictor)
#############################################


def test_parallel_rollout_preserves_shape_and_records_steps():
    b, d, t = 2, 4, 5
    encoded = torch.randn(b, d, t, 1, 1)
    predictor = SimpleNamespace(context_length=1, is_rnn=False)
    jepa = FixedEncoderJepa(encoded, predictor, predict_fn=lambda s, a: s)

    rollout = LatentRollout(nsteps=3, unroll_mode="parallel", return_all_steps=True)
    out = rollout(jepa, observations=None, actions=None)

    assert out.unroll_mode == "parallel"
    assert out.effective_ctxt_window == 1  # predictor.context_length
    assert out.predicted_states.shape == (b, d, t, 1, 1)
    assert out.predicted_steps is not None and len(out.predicted_steps) == 3


#############################################
# Config validation + factory
#############################################


def test_invalid_unroll_mode_raises():
    with pytest.raises(ValueError):
        LatentRollout(nsteps=1, unroll_mode="bogus")


def test_nonpositive_nsteps_raises():
    with pytest.raises(ValueError):
        LatentRollout(nsteps=0, unroll_mode="parallel")


def test_build_rollout_through_factory():
    rollout = build_rollout(
        {"rollout_type": "latent", "nsteps": 2, "unroll_mode": "autoregressive"}
    )
    assert isinstance(rollout, LatentRollout)
    assert rollout.nsteps == 2
