"""
Pytest test module for Factory.base.BaseBuilderDispatcher.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Factory.base import (
    BaseBuilder,
    BaseBuilderDispatcher,
)


#############################################
# Dummy builders
#############################################

class DummyBuilder(BaseBuilder):
    """
    Minimal child builder used to test dispatcher behavior.
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

DEFAULT_OBJECT_CONFIG = {
    "value": 1,
}

DEFAULT_DISPATCHER_CONFIG = {
    "object_a": DEFAULT_OBJECT_CONFIG,
    "object_b": DEFAULT_OBJECT_CONFIG,
}

BUILDER_REGISTRY = {
    "object_a": DummyBuilder(default_config=DEFAULT_OBJECT_CONFIG),
    "object_b": DummyBuilder(default_config=DEFAULT_OBJECT_CONFIG),
}


#############################################
# Registry validation tests
#############################################

def test_dispatcher_rejects_non_dict_registry() -> None:
    """
    Test that the builder registry must be a dictionary.
    """
    with pytest.raises(RuntimeError):
        BaseBuilderDispatcher(
            default_config=DEFAULT_DISPATCHER_CONFIG,
            builder_registry=["invalid"],
        )


def test_dispatcher_rejects_non_string_registry_key() -> None:
    """
    Test that builder registry keys must be strings.
    """
    with pytest.raises(RuntimeError):
        BaseBuilderDispatcher(
            default_config=DEFAULT_DISPATCHER_CONFIG,
            builder_registry={1: DummyBuilder(default_config=DEFAULT_OBJECT_CONFIG)},
        )


def test_dispatcher_rejects_non_callable_registry_value() -> None:
    """
    Test that builder registry values must be callable.
    """
    with pytest.raises(RuntimeError):
        BaseBuilderDispatcher(
            default_config=DEFAULT_DISPATCHER_CONFIG,
            builder_registry={"object_a": "not_callable"},
        )


#############################################
# Dispatch validation tests
#############################################

def test_dispatcher_rejects_unknown_object_name() -> None:
    """
    Test that strict dispatch rejects unregistered object names.
    """
    dispatcher = BaseBuilderDispatcher(
        default_config=DEFAULT_DISPATCHER_CONFIG,
        builder_registry=BUILDER_REGISTRY,
        strict=True,
        check_default_keys=False,
    )

    with pytest.raises(RuntimeError):
        dispatcher(
            config={"unknown_object": {"value": 1}},
            runtime_context=None,
        )


def test_dispatcher_rejects_non_dict_object_config() -> None:
    """
    Test that each object config must be a dictionary.
    """
    dispatcher = BaseBuilderDispatcher(
        default_config=DEFAULT_DISPATCHER_CONFIG,
        builder_registry=BUILDER_REGISTRY,
    )

    with pytest.raises(RuntimeError):
        dispatcher(
            config={"object_a": ["invalid"]},
            runtime_context=None,
        )


def test_dispatcher_strict_default_key_check_rejects_unknown_config_key() -> None:
    """
    Test that dispatcher-level default-key checking rejects unknown keys.
    """
    dispatcher = BaseBuilderDispatcher(
        default_config=DEFAULT_DISPATCHER_CONFIG,
        builder_registry=BUILDER_REGISTRY,
        strict=True,
        check_default_keys=True,
    )

    with pytest.raises(RuntimeError):
        dispatcher(
            config={"unknown_object": {"value": 1}},
            runtime_context=None,
        )


#############################################
# Dispatch build tests
#############################################

def test_dispatcher_builds_registered_objects() -> None:
    """
    Test that registered objects are built through their corresponding builders.
    """
    dispatcher = BaseBuilderDispatcher(
        default_config=DEFAULT_DISPATCHER_CONFIG,
        builder_registry=BUILDER_REGISTRY,
    )

    result = dispatcher(
        config={
            "object_a": {"value": 10},
            "object_b": {"value": 20},
        },
        runtime_context=None,
    )

    assert set(result.keys()) == {"object_a", "object_b"}
    assert result["object_a"]["config"] == {"value": 10}
    assert result["object_b"]["config"] == {"value": 20}


def test_dispatcher_propagates_runtime_context() -> None:
    """
    Test that runtime_context is propagated to child builders.
    """
    dispatcher = BaseBuilderDispatcher(
        default_config=DEFAULT_DISPATCHER_CONFIG,
        builder_registry=BUILDER_REGISTRY,
    )

    runtime_context = {"paths": {"run_dir": "runs/toy"}}

    result = dispatcher(
        config={"object_a": {"value": 10}},
        runtime_context=runtime_context,
    )

    assert result["object_a"]["runtime_context"] == runtime_context


def test_dispatcher_propagates_args_and_kwargs() -> None:
    """
    Test that extra arguments are propagated to child builders.
    """
    dispatcher = BaseBuilderDispatcher(
        default_config=DEFAULT_DISPATCHER_CONFIG,
        builder_registry=BUILDER_REGISTRY,
    )

    result = dispatcher(
        {"object_a": {"value": 10}},
        None,
        "extra_arg",
        extra_kwarg="extra_value",
    )

    assert result["object_a"]["args"] == ("extra_arg",)
    assert result["object_a"]["kwargs"] == {"extra_kwarg": "extra_value"}


def test_dispatcher_non_strict_invalid_config_returns_none() -> None:
    """
    Test that non-strict invalid dispatcher configs do not build objects.
    """
    dispatcher = BaseBuilderDispatcher(
        default_config=DEFAULT_DISPATCHER_CONFIG,
        builder_registry=BUILDER_REGISTRY,
        strict=False,
        check_default_keys=True,
    )

    result = dispatcher(
        config={"unknown_object": {"value": 1}},
        runtime_context=None,
    )

    assert result is None