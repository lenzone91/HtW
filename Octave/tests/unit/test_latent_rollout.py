from copy import deepcopy

import pytest
import torch
from torch import nn

from Octave.src.Models.Model.ac_video_jepa.blocks import AcVideoJepaBlocks
from Octave.src.Models.Model.ac_video_jepa.configs import (
    DEFAULT_AC_VIDEO_JEPA_BLOCKS_CONFIG,
)
from Octave.src.Models.Model.ac_video_jepa.factory import build_ac_video_jepa_blocks
from Octave.src.Rollouts.configs import DEFAULT_LATENT_ROLLOUT_CONFIG
from Octave.src.Rollouts.factory import build_latent_rollout
from Octave.src.Rollouts.latent_rollout import LatentRollout, LatentRolloutOutput


class IdentityEncoder(nn.Module):
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class AddOnePredictor(nn.Module):
    is_rnn = False
    context_length = 1

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None,
    ) -> torch.Tensor:
        return states + 1.0


def make_tiny_blocks_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_BLOCKS_CONFIG)
    config["encoder"].update(
        {
            "stack_sizes": [4, 8, 8],
            "input_channels": 2,
            "input_shape": [2, 32, 32],
            "mlp_output_dim": 32,
        }
    )
    config["predictor"].update(
        {
            "hidden_size": 32,
            "action_dim": 2,
        }
    )
    return config


def make_parallel_blocks() -> AcVideoJepaBlocks:
    return AcVideoJepaBlocks(
        encoder=IdentityEncoder(),
        action_encoder=nn.Identity(),
        predictor=AddOnePredictor(),
        encoder_shape={"feature_dim": 2, "height": 3, "width": 3},
    )


def test_latent_rollout_autoregressive_returns_expected_shape() -> None:
    blocks = build_ac_video_jepa_blocks(config=make_tiny_blocks_config())
    rollout = LatentRollout(
        nsteps=2,
        unroll_mode="autoregressive",
        ctxt_window_time=1,
        return_all_steps=False,
    )

    output = rollout(
        blocks=blocks,
        observations=torch.randn(2, 2, 4, 32, 32),
        actions=torch.randn(2, 2, 4),
    )

    assert isinstance(output, LatentRolloutOutput)
    assert output.encoded_states.shape == torch.Size([2, 32, 4, 1, 1])
    assert output.predicted_states.shape == torch.Size([2, 32, 3, 1, 1])
    assert output.actions_encoded.shape == torch.Size([2, 2, 4])
    assert output.nsteps == 2
    assert output.unroll_mode == "autoregressive"
    assert output.effective_ctxt_window == 1
    assert output.predicted_steps is None
    assert not hasattr(output, "losses")


def test_latent_rollout_autoregressive_returns_all_steps() -> None:
    blocks = build_ac_video_jepa_blocks(config=make_tiny_blocks_config())
    rollout = LatentRollout(
        nsteps=2,
        unroll_mode="autoregressive",
        ctxt_window_time=1,
        return_all_steps=True,
    )

    output = rollout(
        blocks=blocks,
        observations=torch.randn(2, 2, 4, 32, 32),
        actions=torch.randn(2, 2, 4),
    )

    assert output.predicted_steps is not None
    assert len(output.predicted_steps) == 2
    assert output.predicted_steps[0].shape == torch.Size([2, 32, 2, 1, 1])
    assert output.predicted_steps[1].shape == torch.Size([2, 32, 3, 1, 1])


def test_latent_rollout_autoregressive_rejects_too_many_steps() -> None:
    blocks = build_ac_video_jepa_blocks(config=make_tiny_blocks_config())
    rollout = LatentRollout(
        nsteps=5,
        unroll_mode="autoregressive",
        ctxt_window_time=1,
    )

    with pytest.raises(ValueError, match="larger than action sequence length"):
        rollout(
            blocks=blocks,
            observations=torch.randn(2, 2, 4, 32, 32),
            actions=torch.randn(2, 2, 4),
        )


def test_latent_rollout_parallel_returns_expected_shape_and_steps() -> None:
    rollout = LatentRollout(
        nsteps=2,
        unroll_mode="parallel",
        ctxt_window_time=1,
        return_all_steps=True,
    )

    output = rollout(
        blocks=make_parallel_blocks(),
        observations=torch.zeros(2, 2, 4, 3, 3),
        actions=None,
    )

    assert output.encoded_states.shape == torch.Size([2, 2, 4, 3, 3])
    assert output.predicted_states.shape == torch.Size([2, 2, 4, 3, 3])
    assert output.actions_encoded is None
    assert output.nsteps == 2
    assert output.unroll_mode == "parallel"
    assert output.effective_ctxt_window == 1
    assert output.predicted_steps is not None
    assert len(output.predicted_steps) == 2
    assert output.predicted_steps[0].shape == torch.Size([2, 2, 3, 3, 3])
    assert output.predicted_steps[1].shape == torch.Size([2, 2, 3, 3, 3])


def test_latent_rollout_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown unroll_mode"):
        LatentRollout(
            nsteps=2,
            unroll_mode="unknown",
            ctxt_window_time=1,
        )


def test_latent_rollout_rejects_invalid_nsteps() -> None:
    with pytest.raises(ValueError, match="nsteps must be a positive integer"):
        LatentRollout(
            nsteps=0,
            unroll_mode="autoregressive",
            ctxt_window_time=1,
        )


def test_build_latent_rollout_builds_from_plain_config() -> None:
    config = deepcopy(DEFAULT_LATENT_ROLLOUT_CONFIG)
    config.update(
        {
            "nsteps": 3,
            "unroll_mode": "autoregressive",
        }
    )

    rollout = build_latent_rollout(config=config)

    assert isinstance(rollout, LatentRollout)
    assert rollout.nsteps == 3
    assert rollout.unroll_mode == "autoregressive"


def test_build_latent_rollout_does_not_mutate_input_config() -> None:
    config = deepcopy(DEFAULT_LATENT_ROLLOUT_CONFIG)
    original_config = deepcopy(config)

    build_latent_rollout(config=config)

    assert config == original_config


def test_build_latent_rollout_rejects_unknown_key() -> None:
    config = {
        **deepcopy(DEFAULT_LATENT_ROLLOUT_CONFIG),
        "unknown": "bad",
    }

    with pytest.raises(KeyError, match="Unknown LatentRollout config keys"):
        build_latent_rollout(config=config)


def test_build_latent_rollout_rejects_unsupported_rollout_type() -> None:
    config = deepcopy(DEFAULT_LATENT_ROLLOUT_CONFIG)
    config["rollout_type"] = "unsupported"

    with pytest.raises(KeyError, match="Only 'latent'"):
        build_latent_rollout(config=config)
