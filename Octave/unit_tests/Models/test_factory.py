"""
Pytest test module for Models.factory.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Models.configs import (
    DEFAULT_MODELS_CONFIG,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Models.factory import (
    build_models,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Models.waveUnet import (
    WaveUNet,
)


#############################################
# Factory tests
#############################################

def test_build_models_returns_model_dict() -> None:
    """
    Test that build_models returns a dictionary of built models.
    """
    models = build_models(
        model_configs=DEFAULT_MODELS_CONFIG,
        runtime_context=None,
    )

    assert isinstance(models, dict)


def test_build_models_builds_default_waveunet() -> None:
    """
    Test that the default model config builds a WaveUNet instance.
    """
    models = build_models(
        model_configs=DEFAULT_MODELS_CONFIG,
        runtime_context=None,
    )

    assert "waveunet" in models
    assert isinstance(models["waveunet"], WaveUNet)


def test_build_models_rejects_unknown_model_name() -> None:
    """
    Test that unknown model names fail explicitly in strict mode.
    """
    with pytest.raises(RuntimeError):
        build_models(
            model_configs={
                "unknown_model": {},
            },
            runtime_context=None,
            strict=True,
        )


def test_build_models_non_strict_unknown_model_returns_none() -> None:
    """
    Test that unknown model names do not build in non-strict mode.
    """
    models = build_models(
        model_configs={
            "unknown_model": {},
        },
        runtime_context=None,
        strict=False,
    )

    assert models is None