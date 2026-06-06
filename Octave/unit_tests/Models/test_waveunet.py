"""
Pytest test module for Models.waveUnet.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Models.configs import (
    DEFAULT_WAVEUNET_CONFIG,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Models.factory import (
    WaveUNetBuilder,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Models.waveUnet import (
    WaveUNet,
)


#############################################
# Test data
#############################################

BATCH_SIZE = 2
CHANNELS = 1
SIGNAL_LENGTH = 256


#############################################
# Build tests
#############################################

def test_waveunet_builder_builds_waveunet() -> None:
    """
    Test that WaveUNetBuilder builds a WaveUNet instance.
    """
    builder = WaveUNetBuilder()

    model = builder(
        config=DEFAULT_WAVEUNET_CONFIG,
        runtime_context=None,
    )

    assert isinstance(model, WaveUNet)


#############################################
# Forward tests
#############################################

def test_waveunet_forward_runs() -> None:
    """
    Test that WaveUNet forward pass runs on a dummy waveform batch.
    """
    model = WaveUNet(**DEFAULT_WAVEUNET_CONFIG)
    waveform = torch.randn(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH)

    output = model(waveform)

    assert isinstance(output, torch.Tensor)


def test_waveunet_output_shape_matches_input_shape() -> None:
    """
    Test that WaveUNet output shape matches input waveform shape.
    """
    model = WaveUNet(**DEFAULT_WAVEUNET_CONFIG)
    waveform = torch.randn(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH)

    output = model(waveform)

    assert output.shape == waveform.shape


def test_waveunet_output_is_tanh_bounded() -> None:
    """
    Test that WaveUNet output values are bounded by the final tanh activation.
    """
    model = WaveUNet(**DEFAULT_WAVEUNET_CONFIG)
    waveform = torch.randn(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH)

    output = model(waveform)

    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)