"""
Unit tests for Workflow.Configs.resolve — the plain-dict boundary.

Covers interpolation resolution, missing-value strictness, the OmegaConf-free
post-condition, and the leakage guard.
"""

import pytest
from omegaconf import OmegaConf

from src.Workflow.Configs.errors import (
    ConfigError,
)
from src.Workflow.Configs.resolve import (
    check_plain_config,
    resolve_to_plain_dict,
)


#############################################
# resolve_to_plain_dict
#############################################


def test_resolves_interpolation_and_returns_plain_dict():
    cfg = OmegaConf.create({"base": 4, "derived": "${base}", "nested": {"x": "${base}"}})

    result = resolve_to_plain_dict(cfg)

    assert isinstance(result, dict)
    assert result == {"base": 4, "derived": 4, "nested": {"x": 4}}


def test_result_tree_is_omegaconf_free():
    cfg = OmegaConf.create({"a": {"b": [1, {"c": 2}]}})

    result = resolve_to_plain_dict(cfg)

    # Would raise if any DictConfig/ListConfig remained.
    check_plain_config(result)
    assert isinstance(result["a"], dict)
    assert isinstance(result["a"]["b"], list)
    assert isinstance(result["a"]["b"][1], dict)


def test_missing_mandatory_value_raises():
    cfg = OmegaConf.create({"required": "???"})

    with pytest.raises(ConfigError, match="Failed to resolve"):
        resolve_to_plain_dict(cfg)


def test_unresolvable_interpolation_raises():
    cfg = OmegaConf.create({"a": "${does_not_exist}"})

    with pytest.raises(ConfigError, match="Failed to resolve"):
        resolve_to_plain_dict(cfg)


def test_non_dictconfig_input_raises():
    with pytest.raises(ConfigError, match="expects a DictConfig"):
        resolve_to_plain_dict({"already": "a plain dict"})


#############################################
# check_plain_config
#############################################


def test_check_plain_config_passes_on_plain_tree():
    check_plain_config({"a": [1, 2, {"b": "c"}], "d": None})


def test_check_plain_config_detects_nested_omegaconf():
    leaky = {"a": {"b": OmegaConf.create({"leaked": 1})}}

    with pytest.raises(ConfigError, match=r"leaked past the plain-dict boundary at <root>.a.b"):
        check_plain_config(leaky)


def test_check_plain_config_detects_listconfig_in_list():
    leaky = {"items": [OmegaConf.create([1, 2])]}

    with pytest.raises(ConfigError, match=r"items\[0\]"):
        check_plain_config(leaky)
