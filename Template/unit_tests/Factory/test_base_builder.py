"""
Pytest test module for Factory.base.BaseBuilder.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Factory.base import BaseBuilder


#############################################
# Dummy builder
#############################################

class DummyBuilder(BaseBuilder):
    """
    Minimal concrete builder used to test BaseBuilder behavior.
    """

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ) -> dict:
        return {
            "config": config,
            "runtime_context": runtime_context,
            "args": args,
            "kwargs": kwargs,
        }


#############################################
# Test data
#############################################

DEFAULT_CONFIG = {
    "name": "default",
    "value": 1,
}

USER_CONFIG = {
    "name": "custom",
}


#############################################
# Construction tests
#############################################

def test_builder_copies_default_config() -> None:
    """
    Test that the default config is copied at builder construction.
    """
    default_config = dict(DEFAULT_CONFIG)

    builder = DummyBuilder(default_config=default_config)
    default_config["name"] = "mutated"

    assert builder.default_config["name"] == "default"


def test_builder_rejects_non_dict_default_config() -> None:
    """
    Test that the builder default config must be a dictionary.
    """
    with pytest.raises(RuntimeError):
        DummyBuilder(default_config=["invalid"])


#############################################
# Call tests
#############################################

def test_builder_call_returns_build_result() -> None:
    """
    Test that calling the builder delegates to build_from_config.
    """
    builder = DummyBuilder(default_config=DEFAULT_CONFIG)

    result = builder(
        config=USER_CONFIG,
        runtime_context=None,
    )

    assert result["config"] == USER_CONFIG


def test_builder_call_does_not_mutate_user_config() -> None:
    """
    Test that calling the builder does not mutate the user config.
    """
    builder = DummyBuilder(default_config=DEFAULT_CONFIG)
    user_config = dict(USER_CONFIG)

    result = builder(
        config=user_config,
        runtime_context=None,
    )

    result["config"]["name"] = "mutated"

    assert user_config["name"] == "custom"


def test_builder_propagates_runtime_context() -> None:
    """
    Test that runtime_context is propagated to build_from_config.
    """
    builder = DummyBuilder(default_config=DEFAULT_CONFIG)
    runtime_context = {"paths": {"run_dir": "runs/toy"}}

    result = builder(
        config=USER_CONFIG,
        runtime_context=runtime_context,
    )

    assert result["runtime_context"] == runtime_context


def test_builder_propagates_args_and_kwargs() -> None:
    """
    Test that extra positional and keyword arguments are propagated.
    """
    builder = DummyBuilder(default_config=DEFAULT_CONFIG)

    result = builder(
        USER_CONFIG,
        None,
        "extra_arg",
        extra_kwarg="extra_value",
    )

    assert result["args"] == ("extra_arg",)
    assert result["kwargs"] == {"extra_kwarg": "extra_value"}


#############################################
# Validation tests
#############################################

def test_builder_strict_unknown_key_raises() -> None:
    """
    Test that strict mode rejects keys absent from the default config.
    """
    builder = DummyBuilder(
        default_config=DEFAULT_CONFIG,
        strict=True,
    )

    with pytest.raises(RuntimeError):
        builder(
            config={"unknown": 1},
            runtime_context=None,
        )


def test_builder_non_strict_unknown_key_returns_none() -> None:
    """
    Test that non-strict mode does not build invalid configs.
    """
    builder = DummyBuilder(
        default_config=DEFAULT_CONFIG,
        strict=False,
    )

    result = builder(
        config={"unknown": 1},
        runtime_context=None,
    )

    assert result is None


def test_builder_check_default_keys_false_allows_unknown_key() -> None:
    """
    Test that disabling default-key checking allows unknown config keys.
    """
    builder = DummyBuilder(
        default_config=DEFAULT_CONFIG,
        strict=True,
        check_default_keys=False,
    )

    result = builder(
        config={"unknown": 1},
        runtime_context=None,
    )

    assert result["config"] == {"unknown": 1}


def test_builder_rejects_non_dict_user_config() -> None:
    """
    Test that the user config must be a dictionary.
    """
    builder = DummyBuilder(default_config=DEFAULT_CONFIG)

    with pytest.raises(RuntimeError):
        builder(
            config=["invalid"],
            runtime_context=None,
        )