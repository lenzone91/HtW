"""
Unit tests for Workflow.Factory.registry.

Covers strict registration, lookup, default-config isolation, and the strict
error surface (RegistryError on every contract violation).
"""

import pytest

from eb_jepa_cleaned.Workflow.Factory.errors import (
    FactoryError,
    RegistryError,
)
from eb_jepa_cleaned.Workflow.Factory.registry import (
    FieldResolution,
    Registry,
    RegistryEntry,
    SubBuildDeclaration,
)


class Dummy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class OtherDummy:
    pass


#############################################
# Registration
#############################################


def test_add_entry_returns_entry_and_registers():
    registry = Registry("widget")

    entry = registry.add_entry(
        name="dummy",
        object_cls=Dummy,
        default_config={"a": 1},
    )

    assert isinstance(entry, RegistryEntry)
    assert registry.has("dummy")
    assert registry.get_entry("dummy") is entry
    assert registry.available_names() == ["dummy"]


def test_register_class_decorator_registers_and_returns_class():
    registry = Registry("widget")

    @registry.register_class(name="dummy", default_config={})
    class Decorated:
        pass

    assert Decorated.__name__ == "Decorated"
    assert registry.has("dummy")
    assert registry.get_entry("dummy").object_cls is Decorated


def test_available_names_is_sorted():
    registry = Registry("widget")
    registry.add_entry(name="b", object_cls=Dummy, default_config={})
    registry.add_entry(name="a", object_cls=Dummy, default_config={})

    assert registry.available_names() == ["a", "b"]


#############################################
# Default-config isolation
#############################################


def test_default_config_is_deep_copied_on_entry():
    source_config = {"nested": {"x": 1}}
    registry = Registry("widget")

    registry.add_entry(name="dummy", object_cls=Dummy, default_config=source_config)
    source_config["nested"]["x"] = 999

    assert registry.get_entry("dummy").default_config == {"nested": {"x": 1}}


def test_get_default_config_returns_independent_copy():
    registry = Registry("widget")
    registry.add_entry(name="dummy", object_cls=Dummy, default_config={"x": 1})

    first = registry.get_default_config("dummy")
    first["x"] = 999

    assert registry.get_default_config("dummy") == {"x": 1}


#############################################
# Strict error surface
#############################################


def test_duplicate_registration_raises():
    registry = Registry("widget")
    registry.add_entry(name="dummy", object_cls=Dummy, default_config={})

    with pytest.raises(RegistryError, match="Duplicate"):
        registry.add_entry(name="dummy", object_cls=OtherDummy, default_config={})


def test_unknown_name_raises_with_available_names():
    registry = Registry("widget")
    registry.add_entry(name="dummy", object_cls=Dummy, default_config={})

    with pytest.raises(RegistryError, match="Unknown widget 'missing'"):
        registry.get_entry("missing")


def test_registry_error_is_factory_error_subclass():
    registry = Registry("widget")

    with pytest.raises(FactoryError):
        registry.get_entry("missing")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": 123, "object_cls": Dummy, "default_config": {}}, "must be a string"),
        ({"name": "", "object_cls": Dummy, "default_config": {}}, "cannot be empty"),
        ({"name": "d", "object_cls": Dummy(), "default_config": {}}, "must be a class"),
        ({"name": "d", "object_cls": Dummy, "default_config": []}, "must be a dictionary"),
        (
            {"name": "d", "object_cls": Dummy, "default_config": {}, "sub_builds": []},
            "sub_builds must be a tuple",
        ),
        (
            {"name": "d", "object_cls": Dummy, "default_config": {}, "sub_builds": ("x",)},
            "must contain SubBuildDeclaration",
        ),
        (
            {
                "name": "d",
                "object_cls": Dummy,
                "default_config": {},
                "field_resolutions": [],
            },
            "field_resolutions must be a tuple",
        ),
        (
            {
                "name": "d",
                "object_cls": Dummy,
                "default_config": {},
                "field_resolutions": ("x",),
            },
            "must contain FieldResolution",
        ),
        (
            {"name": "d", "object_cls": Dummy, "default_config": {}, "metadata": []},
            "metadata must be a dictionary",
        ),
    ],
)
def test_invalid_entry_arguments_raise(kwargs, match):
    registry = Registry("widget")

    with pytest.raises(RegistryError, match=match):
        registry.add_entry(**kwargs)


#############################################
# Declaration containers
#############################################


def test_sub_build_declaration_is_frozen():
    declaration = SubBuildDeclaration(
        source_key="s", target_key="t", builder=object()
    )

    with pytest.raises(Exception):
        declaration.source_key = "other"


def test_field_resolution_defaults():
    resolution = FieldResolution()

    assert resolution.target_key is None
    assert resolution.resolver is None
    assert resolution.remove_source_keys == ()
