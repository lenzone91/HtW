"""
Unit tests for AIML.Models.Loading.
"""

import pytest
import torch
from torch import nn

from src.AIML.Models.Loading.factory import (
    check_loading_type,
    is_loading_enabled,
    load_model_if_needed,
    resolve_checkpoint_path,
)
from src.AIML.Models.Loading.model_loading import (
    extract_state_dict,
    load_model_state_dict,
)


def test_load_model_state_dict_roundtrip(tmp_path):
    source = nn.Linear(2, 2)
    target = nn.Linear(2, 2)
    path = tmp_path / "ckpt.pt"
    torch.save(source.state_dict(), path)

    load_model_state_dict(target, str(path))

    assert torch.equal(source.weight, target.weight)
    assert torch.equal(source.bias, target.bias)


def test_extract_state_dict_variants():
    sd = nn.Linear(2, 2).state_dict()

    assert extract_state_dict(sd) is sd
    assert extract_state_dict({"state_dict": sd}) is sd
    assert extract_state_dict({"model_state_dict": sd}) is sd
    assert extract_state_dict({"weights": sd}, state_dict_key="weights") is sd


def test_extract_state_dict_unknown_raises():
    with pytest.raises(ValueError, match="Could not infer model state dict"):
        extract_state_dict({"unexpected": 1})


def test_is_loading_enabled():
    assert is_loading_enabled(None) is False
    assert is_loading_enabled({"enabled": False}) is False
    assert is_loading_enabled({"enabled": True}) is True


def test_load_model_if_needed_disabled_returns_same_model():
    model = nn.Linear(2, 2)

    assert load_model_if_needed(model, None) is model
    assert load_model_if_needed(model, {"enabled": False}) is model


def test_check_loading_type_mismatch_raises():
    with pytest.raises(ValueError, match="Invalid loading type"):
        check_loading_type({"type": "lightning_module"}, expected_type="torch_model")


def test_resolve_checkpoint_path_none_raises():
    with pytest.raises(ValueError, match="checkpoint_path is None"):
        resolve_checkpoint_path(None)
