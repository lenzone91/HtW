"""
Pytest test module for DataProcessing.factory.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataProcessing.collators import (
    TSEWaveformCollator,
)




from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataProcessing.configs import (
    DEFAULT_ADD_NOISE_AT_SNR_CONFIG,
    DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIG,
    DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIGS,
    DEFAULT_NOISY_TSE_WAVEFORM_COLLATOR_CONFIGS,
)

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataProcessing.factory import (
    build_collator,
    build_transforms,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataProcessing.transforms import (
    AddNoiseAtSNR,
)


#############################################
# Transform factory tests
#############################################

def test_build_transforms_none_returns_empty_list() -> None:
    """
    Test that None transform config builds an empty transform list.
    """
    transforms = build_transforms(
        config=None,
        runtime_context=None,
    )

    assert transforms == []


def test_build_transforms_empty_config_returns_empty_list() -> None:
    """
    Test that an empty transform config builds an empty transform list.
    """
    transforms = build_transforms(
        config={},
        runtime_context=None,
    )

    assert transforms == []


def test_build_transforms_builds_add_noise_at_snr() -> None:
    """
    Test that AddNoiseAtSNR can be built from config.
    """
    transforms = build_transforms(
        config={
            "add_noise_at_snr": dict(DEFAULT_ADD_NOISE_AT_SNR_CONFIG),
        },
        runtime_context=None,
    )

    assert len(transforms) == 1
    assert isinstance(transforms[0], AddNoiseAtSNR)


def test_build_transforms_preserves_config_order() -> None:
    """
    Test that the transform list follows the config insertion order.

    Current transform configs use registry names as keys, so only one
    AddNoiseAtSNR instance can be represented without changing the config
    convention.
    """
    transforms = build_transforms(
        config={
            "add_noise_at_snr": dict(DEFAULT_ADD_NOISE_AT_SNR_CONFIG),
        },
        runtime_context=None,
    )

    assert [type(transform) for transform in transforms] == [AddNoiseAtSNR]


def test_build_transforms_rejects_unknown_transform() -> None:
    """
    Test that unknown transform names fail explicitly in strict mode.
    """
    with pytest.raises(RuntimeError):
        build_transforms(
            config={
                "unknown_transform": {},
            },
            runtime_context=None,
            strict=True,
        )


#############################################
# Collator factory tests
#############################################

def test_build_collator_builds_default_tse_waveform_collator() -> None:
    """
    Test that the default collator config builds a TSEWaveformCollator.
    """
    collator = build_collator(
        collator_config=DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIGS,
        runtime_context=None,
    )

    assert isinstance(collator, TSEWaveformCollator)
    assert collator.transforms == []


def test_build_collator_builds_noisy_tse_waveform_collator() -> None:
    """
    Test that the noisy collator config builds a collator with transforms.
    """
    collator = build_collator(
        collator_config=DEFAULT_NOISY_TSE_WAVEFORM_COLLATOR_CONFIGS,
        runtime_context=None,
    )

    assert isinstance(collator, TSEWaveformCollator)
    assert len(collator.transforms) == 1
    assert isinstance(collator.transforms[0], AddNoiseAtSNR)


def test_build_collator_preserves_transform_order() -> None:
    """
    Test that transform order from config is preserved inside the collator.
    """
    collator = build_collator(
        collator_config=DEFAULT_NOISY_TSE_WAVEFORM_COLLATOR_CONFIGS,
        runtime_context=None,
    )

    assert [type(transform) for transform in collator.transforms] == [
        AddNoiseAtSNR,
    ]


def test_build_collator_rejects_unknown_collator_type() -> None:
    """
    Test that unknown collator types fail explicitly in strict mode.
    """
    config = {
        "unknown_collator": dict(DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIG),
    }

    with pytest.raises(RuntimeError):
        build_collator(
            collator_config=config,
            runtime_context=None,
            strict=True,
        )