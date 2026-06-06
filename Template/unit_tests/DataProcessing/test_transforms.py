"""
Pytest test module for DataProcessing.transforms.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest
import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataProcessing.transforms import (
    AddNoiseAtSNR,
)


#############################################
# Test data
#############################################

BATCH_SIZE = 4
CHANNELS = 1
SIGNAL_LENGTH = 4096


def make_batch() -> dict:
    """
    Build a toy batch for transform tests.
    """
    return {
        "mixture": torch.ones(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH),
        "target": torch.zeros(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH),
        "metadata": [{"id": i} for i in range(BATCH_SIZE)],
    }


#############################################
# AddNoiseAtSNR tests
#############################################

def test_add_noise_at_snr_updates_output_key() -> None:
    """
    Test that AddNoiseAtSNR writes the transformed signal to output_key.
    """
    batch = make_batch()
    transform = AddNoiseAtSNR(
        input_key="mixture",
        output_key="noisy_mixture",
        snr_db=20.0,
    )

    transformed_batch = transform(batch)

    assert "noisy_mixture" in transformed_batch


def test_add_noise_at_snr_preserves_output_shape() -> None:
    """
    Test that transformed signals keep the input tensor shape.
    """
    batch = make_batch()
    transform = AddNoiseAtSNR()

    transformed_batch = transform(batch)

    assert transformed_batch["mixture"].shape == torch.Size(
        [BATCH_SIZE, CHANNELS, SIGNAL_LENGTH]
    )


def test_add_noise_at_snr_preserves_other_keys() -> None:
    """
    Test that unrelated batch entries are preserved.
    """
    batch = make_batch()
    transform = AddNoiseAtSNR()

    transformed_batch = transform(batch)

    assert "target" in transformed_batch
    assert "metadata" in transformed_batch
    assert transformed_batch["metadata"] == batch["metadata"]


def test_add_noise_at_snr_missing_input_key_raises() -> None:
    """
    Test that missing input_key fails explicitly.
    """
    batch = make_batch()
    batch.pop("mixture")

    transform = AddNoiseAtSNR(input_key="mixture")

    with pytest.raises(KeyError):
        transform(batch)


def test_add_noise_at_snr_rejects_non_tensor_input() -> None:
    """
    Test that input_key must contain a tensor.
    """
    batch = make_batch()
    batch["mixture"] = "invalid"

    transform = AddNoiseAtSNR(input_key="mixture")

    with pytest.raises(TypeError):
        transform(batch)


def test_add_noise_at_snr_rejects_unbatched_tensor() -> None:
    """
    Test that input tensors must be batched.
    """
    batch = make_batch()
    batch["mixture"] = torch.ones(SIGNAL_LENGTH)

    transform = AddNoiseAtSNR(input_key="mixture")

    with pytest.raises(ValueError):
        transform(batch)


def test_add_noise_at_snr_compute_power_is_broadcastable() -> None:
    """
    Test that compute_power preserves broadcastable dimensions.
    """
    signal = torch.ones(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH)
    transform = AddNoiseAtSNR()

    power = transform.compute_power(signal)

    assert power.shape == torch.Size([BATCH_SIZE, 1, 1])


def test_add_noise_at_snr_empirical_snr_is_close_to_requested_value() -> None:
    """
    Test that the added noise approximately matches the requested SNR.
    """
    torch.manual_seed(0)

    batch = make_batch()
    clean_mixture = batch["mixture"].clone()

    transform = AddNoiseAtSNR(
        input_key="mixture",
        output_key="mixture",
        snr_db=20.0,
    )

    transformed_batch = transform(batch)

    noisy_mixture = transformed_batch["mixture"]
    added_noise = noisy_mixture - clean_mixture

    signal_power = clean_mixture.pow(2).mean(dim=(1, 2))
    noise_power = added_noise.pow(2).mean(dim=(1, 2))

    empirical_snr = 10.0 * torch.log10(signal_power / noise_power)

    assert torch.allclose(
        empirical_snr,
        torch.full_like(empirical_snr, 20.0),
        atol=0.5,
    )