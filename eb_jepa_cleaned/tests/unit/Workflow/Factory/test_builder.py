"""
Unit tests for Workflow.Factory.builder.

Covers single/named builds, name resolution, field resolutions, sub-builds,
type-field routing, config isolation, and the strict error surface.
"""

import pytest

from src.Workflow.Factory.builder import (
    RegistryBuilder,
)
from src.Workflow.Factory.errors import (
    BuilderError,
    FactoryError,
    RegistryError,
)
from src.Workflow.Factory.registry import (
    FieldResolution,
    Registry,
    SubBuildDeclaration,
)


#############################################
# Test doubles
#############################################


class Widget:
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class Block:
    def __init__(self, size=1):
        self.size = size


class Net:
    def __init__(self, block=None, name="net"):
        self.block = block
        self.name = name


def make_widget_registry():
    registry = Registry("widget")
    registry.add_entry(name="widget", object_cls=Widget, default_config={"a": None, "b": None})
    return registry


#############################################
# build_one
#############################################


def test_build_one_by_explicit_name():
    builder = RegistryBuilder(make_widget_registry())

    widget = builder.build_one({"a": 1, "b": 2}, name="widget")

    assert isinstance(widget, Widget)
    assert (widget.a, widget.b) == (1, 2)


def test_build_one_by_type_field():
    registry = make_widget_registry()
    builder = RegistryBuilder(registry, type_field="kind")

    widget = builder.build_one({"kind": "widget", "a": 5})

    assert isinstance(widget, Widget)
    assert widget.a == 5
    # The routing type field must not reach the constructor.
    assert not hasattr(widget, "kind")


def test_build_one_does_not_mutate_input_config():
    builder = RegistryBuilder(make_widget_registry())
    config = {"a": 1, "b": 2}

    builder.build_one(config, name="widget")

    assert config == {"a": 1, "b": 2}


def test_build_one_missing_type_field_value_raises():
    builder = RegistryBuilder(make_widget_registry(), type_field="kind")

    with pytest.raises(BuilderError, match="Missing type field 'kind'"):
        builder.build_one({"a": 1})


def test_build_one_no_name_no_type_field_raises():
    builder = RegistryBuilder(make_widget_registry())

    with pytest.raises(BuilderError, match="explicit name or a type_field"):
        builder.build_one({"a": 1})


def test_build_one_unknown_name_raises():
    builder = RegistryBuilder(make_widget_registry())

    with pytest.raises(RegistryError, match="Unknown widget 'missing'"):
        builder.build_one({"a": 1}, name="missing")


def test_build_one_non_dict_config_raises():
    builder = RegistryBuilder(make_widget_registry())

    with pytest.raises(BuilderError, match="must be a dictionary"):
        builder.build_one(["not", "a", "dict"], name="widget")


def test_build_one_invalid_key_raises():
    builder = RegistryBuilder(make_widget_registry())

    with pytest.raises(BuilderError, match="Invalid config keys"):
        builder.build_one({"a": 1, "unknown": 2}, name="widget")


def test_check_default_keys_false_allows_keys_outside_default():
    # 'b' is a real Widget kwarg but is absent from the declared default_config.
    # With check_default_keys disabled, the builder accepts it instead of
    # treating the default_config as an exhaustive allow-list.
    registry = Registry("widget")
    registry.add_entry(name="widget", object_cls=Widget, default_config={"a": None})
    lenient = RegistryBuilder(registry, check_default_keys=False)

    widget = lenient.build_one({"a": 1, "b": 2}, name="widget")

    assert (widget.a, widget.b) == (1, 2)


#############################################
# build_named
#############################################


def test_build_named_uses_outer_keys_as_names():
    builder = RegistryBuilder(make_widget_registry())

    objects = builder.build_named({"widget": {"a": 1}})

    assert set(objects) == {"widget"}
    assert isinstance(objects["widget"], Widget)
    assert objects["widget"].a == 1


def test_build_named_routes_by_type_field():
    builder = RegistryBuilder(make_widget_registry(), type_field="kind")

    objects = builder.build_named({"left": {"kind": "widget", "a": 1}})

    assert set(objects) == {"left"}
    assert objects["left"].a == 1


def test_build_named_non_dict_inner_raises():
    builder = RegistryBuilder(make_widget_registry())

    with pytest.raises(BuilderError, match="must be a dictionary"):
        builder.build_named({"widget": ["bad"]})


#############################################
# Field resolutions
#############################################


def test_field_resolution_sets_target_and_removes_source():
    registry = Registry("widget")
    registry.add_entry(
        name="widget",
        object_cls=Widget,
        default_config={"raw": None, "a": None},
        field_resolutions=(
            FieldResolution(
                target_key="a",
                resolver=lambda config, runtime_context, **kwargs: config["raw"] * 10,
                remove_source_keys=("raw",),
            ),
        ),
    )
    builder = RegistryBuilder(registry)

    widget = builder.build_one({"raw": 3}, name="widget")

    assert widget.a == 30


def test_field_resolution_receives_runtime_context():
    seen = {}

    def resolver(config, runtime_context, **kwargs):
        seen["ctx"] = runtime_context
        return 1

    registry = Registry("widget")
    registry.add_entry(
        name="widget",
        object_cls=Widget,
        default_config={"a": None},
        field_resolutions=(FieldResolution(target_key="a", resolver=resolver),),
    )
    builder = RegistryBuilder(registry)

    builder.build_one({"a": None}, runtime_context={"device": "cpu"}, name="widget")

    assert seen["ctx"] == {"device": "cpu"}


#############################################
# Sub-builds
#############################################


def make_block_registry():
    registry = Registry("block")
    registry.add_entry(name="block", object_cls=Block, default_config={"size": 1})
    return registry


def make_net_builder():
    block_builder = RegistryBuilder(make_block_registry())

    net_registry = Registry("net")
    net_registry.add_entry(
        name="net",
        object_cls=Net,
        default_config={"block": None, "name": "net"},
        sub_builds=(
            SubBuildDeclaration(
                source_key="block",
                target_key="block",
                builder=block_builder,
                build_method="one",
                type_name="block",
            ),
        ),
    )
    return RegistryBuilder(net_registry)


def test_sub_build_one():
    builder = make_net_builder()

    net = builder.build_one({"block": {"size": 7}, "name": "n"}, name="net")

    assert isinstance(net.block, Block)
    assert net.block.size == 7
    assert net.name == "n"


def test_sub_build_missing_source_raises():
    builder = make_net_builder()

    with pytest.raises(BuilderError, match="Missing sub-build config key 'block'"):
        builder.build_one({"name": "n"}, name="net")


def test_sub_build_named():
    block_builder = RegistryBuilder(make_block_registry())

    class Bank:
        def __init__(self, blocks=None):
            self.blocks = blocks

    registry = Registry("bank")
    registry.add_entry(
        name="bank",
        object_cls=Bank,
        default_config={"blocks": None},
        sub_builds=(
            SubBuildDeclaration(
                source_key="blocks",
                target_key="blocks",
                builder=block_builder,
                build_method="named",
            ),
        ),
    )
    builder = RegistryBuilder(registry)

    bank = builder.build_one(
        {"blocks": {"block": {"size": 2}}}, name="bank"
    )

    assert isinstance(bank.blocks["block"], Block)
    assert bank.blocks["block"].size == 2


def test_sub_build_phase_single_named():
    block_builder = RegistryBuilder(make_block_registry())

    class Phased:
        def __init__(self, phases=None):
            self.phases = phases

    registry = Registry("phased")
    registry.add_entry(
        name="phased",
        object_cls=Phased,
        default_config={"phases": None},
        sub_builds=(
            SubBuildDeclaration(
                source_key="phases",
                target_key="phases",
                builder=block_builder,
                build_method="phase_single_named",
            ),
        ),
    )
    builder = RegistryBuilder(registry)

    phased = builder.build_one(
        {
            "phases": {
                "train": {"block": {"size": 1}},
                "val": None,
            }
        },
        name="phased",
    )

    assert isinstance(phased.phases["train"], Block)
    assert phased.phases["val"] is None


def test_phase_single_named_rejects_multiple_per_phase():
    # Two valid registered names so the build reaches the "exactly one" guard
    # rather than failing earlier on an unknown name.
    multi_registry = Registry("block")
    multi_registry.add_entry(name="a", object_cls=Block, default_config={"size": 1})
    multi_registry.add_entry(name="b", object_cls=Block, default_config={"size": 1})
    block_builder = RegistryBuilder(multi_registry)

    class Phased:
        def __init__(self, phases=None):
            self.phases = phases

    registry = Registry("phased")
    registry.add_entry(
        name="phased",
        object_cls=Phased,
        default_config={"phases": None},
        sub_builds=(
            SubBuildDeclaration(
                source_key="phases",
                target_key="phases",
                builder=block_builder,
                build_method="phase_single_named",
            ),
        ),
    )
    builder = RegistryBuilder(registry)

    with pytest.raises(BuilderError, match="exactly one named sub-object"):
        builder.build_one(
            {"phases": {"train": {"a": {"size": 1}, "b": {"size": 2}}}},
            name="phased",
        )


def test_unknown_sub_build_method_raises():
    block_builder = RegistryBuilder(make_block_registry())

    registry = Registry("net")
    registry.add_entry(
        name="net",
        object_cls=Net,
        default_config={"block": None, "name": "net"},
        sub_builds=(
            SubBuildDeclaration(
                source_key="block",
                target_key="block",
                builder=block_builder,
                build_method="bogus",
            ),
        ),
    )
    builder = RegistryBuilder(registry)

    with pytest.raises(BuilderError, match="Unknown sub-build method 'bogus'"):
        builder.build_one({"block": {"size": 1}}, name="net")


#############################################
# Constructor checks
#############################################


def test_builder_requires_registry():
    with pytest.raises(BuilderError, match="must be a Registry"):
        RegistryBuilder({"not": "a registry"})


def test_builder_error_is_factory_error_subclass():
    builder = RegistryBuilder(make_widget_registry())

    with pytest.raises(FactoryError):
        builder.build_one({"a": 1})
