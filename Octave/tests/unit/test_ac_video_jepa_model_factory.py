from copy import deepcopy

import pytest
from torch import nn

from Octave.src.Models.Model.ac_video_jepa.configs import (
    DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG,
)
from Octave.src.Models.Model.ac_video_jepa.factory import build_ac_video_jepa_components


def make_tiny_components_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG)
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


def test_build_ac_video_jepa_components_builds_architecture_components() -> None:
    components = build_ac_video_jepa_components(config=make_tiny_components_config())

    assert components["encoder"].mlp_output_dim == 32
    assert isinstance(components["action_encoder"], nn.Identity)
    assert components["predictor"].rnn.input_size == 2
    assert components["predictor"].rnn.hidden_size == 32
    assert components["encoder_shape"] == {
        "feature_dim": 32,
        "height": 1,
        "width": 1,
    }


def test_build_ac_video_jepa_components_does_not_mutate_input_config() -> None:
    config = make_tiny_components_config()
    original_config = deepcopy(config)

    build_ac_video_jepa_components(config=config)

    assert config == original_config


def test_build_ac_video_jepa_components_rejects_unknown_top_level_key() -> None:
    config = {
        **make_tiny_components_config(),
        "regularizer": {},
    }

    with pytest.raises(KeyError, match="Unknown AcVideoJepa components config keys"):
        build_ac_video_jepa_components(config=config)


def test_build_ac_video_jepa_components_rejects_unknown_nested_key() -> None:
    config = make_tiny_components_config()
    config["predictor"]["unknown"] = "bad"

    with pytest.raises(KeyError, match="Unknown AcVideoJepa predictor config keys"):
        build_ac_video_jepa_components(config=config)


def test_build_ac_video_jepa_components_rejects_unsupported_encoder_type() -> None:
    config = make_tiny_components_config()
    config["encoder"]["encoder_type"] = "unsupported"

    with pytest.raises(KeyError, match="Only 'impala'"):
        build_ac_video_jepa_components(config=config)


def test_build_ac_video_jepa_components_rejects_unsupported_predictor_type() -> None:
    config = make_tiny_components_config()
    config["predictor"]["predictor_type"] = "unsupported"

    with pytest.raises(KeyError, match="Only 'rnn'"):
        build_ac_video_jepa_components(config=config)
