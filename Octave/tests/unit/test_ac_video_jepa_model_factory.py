from copy import deepcopy

import pytest
import torch
from torch import nn

from Octave.src.Models.Model.ac_video_jepa.blocks import AcVideoJepaBlocks
from Octave.src.Models.Model.ac_video_jepa.ac_video_jepa_model import AcVideoJepa
from Octave.src.Models.Model.ac_video_jepa.configs import (
    DEFAULT_AC_VIDEO_JEPA_BLOCKS_CONFIG,
    DEFAULT_AC_VIDEO_JEPA_MODEL_CONFIG,
)
from Octave.src.Models.Model.ac_video_jepa.factory import (
    build_ac_video_jepa,
    build_ac_video_jepa_blocks,
)


def make_tiny_model_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_MODEL_CONFIG)
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
    config["inverse_dynamics_model"].update(
        {
            "hidden_dim": 16,
            "action_dim": 2,
        }
    )
    config["regularizer"].update(
        {
            "cov_coeff": 0.1,
            "std_coeff": 0.1,
            "sim_coeff_t": 0.1,
            "idm_coeff": 0.1,
            "first_t_only": False,
        }
    )
    return config


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


def test_build_ac_video_jepa_blocks_builds_architecture_blocks() -> None:
    blocks = build_ac_video_jepa_blocks(config=make_tiny_blocks_config())

    assert isinstance(blocks, AcVideoJepaBlocks)
    assert blocks.encoder.mlp_output_dim == 32
    assert isinstance(blocks.action_encoder, nn.Identity)
    assert blocks.predictor.rnn.input_size == 2
    assert blocks.predictor.rnn.hidden_size == 32
    assert blocks.encoder_shape == {
        "feature_dim": 32,
        "height": 1,
        "width": 1,
    }


def test_build_ac_video_jepa_blocks_does_not_mutate_input_config() -> None:
    config = make_tiny_blocks_config()
    original_config = deepcopy(config)

    build_ac_video_jepa_blocks(config=config)

    assert config == original_config


def test_build_ac_video_jepa_blocks_rejects_unknown_top_level_key() -> None:
    config = {
        **make_tiny_blocks_config(),
        "regularizer": {},
    }

    with pytest.raises(KeyError, match="Unknown AcVideoJepa blocks config keys"):
        build_ac_video_jepa_blocks(config=config)


def test_build_ac_video_jepa_blocks_rejects_unknown_nested_key() -> None:
    config = make_tiny_blocks_config()
    config["predictor"]["unknown"] = "bad"

    with pytest.raises(KeyError, match="Unknown AcVideoJepa predictor config keys"):
        build_ac_video_jepa_blocks(config=config)


def test_build_ac_video_jepa_blocks_rejects_unsupported_encoder_type() -> None:
    config = make_tiny_blocks_config()
    config["encoder"]["encoder_type"] = "unsupported"

    with pytest.raises(KeyError, match="Only 'impala'"):
        build_ac_video_jepa_blocks(config=config)


def test_build_ac_video_jepa_blocks_rejects_unsupported_predictor_type() -> None:
    config = make_tiny_blocks_config()
    config["predictor"]["predictor_type"] = "unsupported"

    with pytest.raises(KeyError, match="Only 'rnn'"):
        build_ac_video_jepa_blocks(config=config)


def test_build_ac_video_jepa_builds_model_from_plain_config() -> None:
    model = build_ac_video_jepa(config=make_tiny_model_config())

    assert isinstance(model, AcVideoJepa)
    assert model.encoder.mlp_output_dim == 32
    assert model.predictor.rnn.input_size == 2
    assert model.predictor.rnn.hidden_size == 32


def test_build_ac_video_jepa_does_not_mutate_input_config() -> None:
    config = make_tiny_model_config()
    original_config = deepcopy(config)

    build_ac_video_jepa(config=config)

    assert config == original_config


def test_build_ac_video_jepa_rejects_unknown_top_level_key() -> None:
    config = {
        **make_tiny_model_config(),
        "unknown": {},
    }

    with pytest.raises(KeyError, match="Unknown AcVideoJepa model config keys"):
        build_ac_video_jepa(config=config)


def test_build_ac_video_jepa_rejects_unknown_nested_key() -> None:
    config = make_tiny_model_config()
    config["encoder"]["unknown"] = "bad"

    with pytest.raises(KeyError, match="Unknown AcVideoJepa encoder config keys"):
        build_ac_video_jepa(config=config)


def test_build_ac_video_jepa_rejects_other_model_type() -> None:
    config = make_tiny_model_config()
    config["model_type"] = "video_jepa"

    with pytest.raises(KeyError, match="Only 'ac_video_jepa'"):
        build_ac_video_jepa(config=config)


def test_ac_video_jepa_unroll_computes_training_loss() -> None:
    model = build_ac_video_jepa(config=make_tiny_model_config())
    model.train()

    observations = torch.randn(2, 2, 4, 32, 32)
    actions = torch.randn(2, 2, 4)

    _, losses = model.unroll(
        observations,
        actions,
        nsteps=2,
        unroll_mode="autoregressive",
        ctxt_window_time=1,
        compute_loss=True,
        return_all_steps=False,
    )

    loss, reg_loss, reg_loss_unweighted, reg_loss_dict, pred_loss = losses

    assert loss.shape == torch.Size([])
    assert reg_loss.shape == torch.Size([])
    assert reg_loss_unweighted.shape == torch.Size([])
    assert pred_loss.shape == torch.Size([])
    assert set(reg_loss_dict) == {
        "cov_loss",
        "std_loss",
        "sim_loss_t",
        "idm_loss",
    }
