import pytest

from Octave.src.Workflow.Factory import (
    FieldResolution,
    Registry,
    RegistryBuilder,
    SubBuildDeclaration,
)
from Octave.src.Workflow.Utils import handle_error


class ToyObject:
    def __init__(self, value: int = 1, child=None, **kwargs) -> None:
        self.value = value
        self.child = child
        self.kwargs = kwargs


class StrictToyObject:
    def __init__(self, value: int = 1) -> None:
        self.value = value


def make_toy_registry() -> Registry:
    registry = Registry(object_family="toy")
    registry.add_entry(
        name="toy",
        object_cls=ToyObject,
        default_config={
            "toy_type": "toy",
            "value": 1,
        },
        type_field="toy_type",
    )
    return registry


def test_handle_error_raises() -> None:
    with pytest.raises(RuntimeError, match="bad"):
        handle_error("bad")


def test_registry_registers_class_with_decorator() -> None:
    registry = Registry(object_family="toy")

    @registry.register_class(
        name="toy",
        default_config={"value": 1},
    )
    class RegisteredToy(ToyObject):
        pass

    assert registry.has("toy")
    assert registry.get_entry("toy").object_cls is RegisteredToy
    assert registry.available_names() == ["toy"]


def test_registry_rejects_duplicate_registration() -> None:
    registry = make_toy_registry()

    with pytest.raises(RuntimeError, match="Duplicate toy registration"):
        registry.add_entry(
            name="toy",
            object_cls=ToyObject,
            default_config={"value": 1},
        )


def test_registry_returns_copied_default_config() -> None:
    registry = make_toy_registry()

    config = registry.get_default_config("toy")
    config["value"] = 99

    assert registry.get_default_config("toy")["value"] == 1


def test_builder_build_one_from_explicit_name() -> None:
    registry = make_toy_registry()
    builder = RegistryBuilder(registry=registry)

    obj = builder.build_one(config={"value": 3}, name="toy")

    assert isinstance(obj, ToyObject)
    assert obj.value == 3


def test_builder_build_one_from_type_field() -> None:
    registry = make_toy_registry()
    builder = RegistryBuilder(registry=registry, type_field="toy_type")

    obj = builder.build_one(config={"toy_type": "toy", "value": 3})

    assert isinstance(obj, ToyObject)
    assert obj.value == 3


def test_builder_build_named_uses_outer_name() -> None:
    registry = make_toy_registry()
    builder = RegistryBuilder(registry=registry)

    objects = builder.build_named(configs={"toy": {"value": 4}})

    assert set(objects) == {"toy"}
    assert objects["toy"].value == 4


def test_builder_does_not_mutate_input_config() -> None:
    registry = make_toy_registry()
    builder = RegistryBuilder(registry=registry, type_field="toy_type")
    config = {"toy_type": "toy", "value": 3}

    builder.build_one(config=config)

    assert config == {"toy_type": "toy", "value": 3}


def test_builder_rejects_unknown_config_key() -> None:
    registry = make_toy_registry()
    builder = RegistryBuilder(registry=registry)

    with pytest.raises(RuntimeError, match="Invalid config keys"):
        builder.build_one(config={"value": 3, "unknown": 4}, name="toy")


def test_builder_can_skip_default_key_validation() -> None:
    registry = make_toy_registry()
    builder = RegistryBuilder(registry=registry, check_default_keys=False)

    obj = builder.build_one(config={"value": 3, "unknown": 4}, name="toy")

    assert obj.value == 3
    assert obj.kwargs["unknown"] == 4


def test_builder_resolves_field_before_construction() -> None:
    def resolve_value(config, runtime_context=None):
        return config["raw"] + runtime_context["offset"]

    registry = Registry(object_family="toy")
    registry.add_entry(
        name="toy",
        object_cls=ToyObject,
        default_config={"raw": 1},
        field_resolutions=(
            FieldResolution(
                target_key="value",
                resolver=resolve_value,
                remove_source_keys=("raw",),
            ),
        ),
    )
    builder = RegistryBuilder(registry=registry)

    obj = builder.build_one(
        config={"raw": 3},
        name="toy",
        runtime_context={"offset": 4},
    )

    assert obj.value == 7
    assert "raw" not in obj.kwargs


def test_builder_resolves_one_sub_build() -> None:
    child_registry = make_toy_registry()
    child_builder = RegistryBuilder(registry=child_registry, type_field="toy_type")
    parent_registry = Registry(object_family="parent")
    parent_registry.add_entry(
        name="parent",
        object_cls=ToyObject,
        default_config={"child_config": {}},
        sub_builds=(
            SubBuildDeclaration(
                source_key="child_config",
                target_key="child",
                builder=child_builder,
                build_method="one",
                type_field="toy_type",
            ),
        ),
    )
    parent_builder = RegistryBuilder(registry=parent_registry)

    obj = parent_builder.build_one(
        config={"child_config": {"toy_type": "toy", "value": 5}},
        name="parent",
    )

    assert isinstance(obj.child, ToyObject)
    assert obj.child.value == 5
    assert "child_config" not in obj.kwargs


def test_builder_does_not_pass_build_context_kwargs_to_constructors() -> None:
    child_registry = Registry(object_family="child")
    child_registry.add_entry(
        name="child",
        object_cls=StrictToyObject,
        default_config={"value": 1},
    )
    child_builder = RegistryBuilder(registry=child_registry)
    parent_registry = Registry(object_family="parent")
    parent_registry.add_entry(
        name="parent",
        object_cls=ToyObject,
        default_config={"child_config": {}},
        sub_builds=(
            SubBuildDeclaration(
                source_key="child_config",
                target_key="child",
                builder=child_builder,
                build_method="one",
                type_name="child",
                forwarded_kwargs=(),
            ),
        ),
    )
    parent_builder = RegistryBuilder(registry=parent_registry)

    obj = parent_builder.build_one(
        config={"child_config": {"value": 5}},
        name="parent",
        parent_only="not forwarded",
    )

    assert isinstance(obj.child, StrictToyObject)
    assert obj.child.value == 5
    assert "parent_only" not in obj.kwargs


def test_builder_can_limit_forwarded_kwargs_for_sub_builds() -> None:
    def resolve_value(
        config,
        runtime_context=None,
        parent_only: int | None = None,
    ):
        return config["value"] + parent_only

    child_registry = Registry(object_family="child")
    child_registry.add_entry(
        name="child",
        object_cls=StrictToyObject,
        default_config={"value": 1},
        field_resolutions=(
            FieldResolution(
                target_key="value",
                resolver=resolve_value,
            ),
        ),
    )
    child_builder = RegistryBuilder(registry=child_registry)
    parent_registry = Registry(object_family="parent")
    parent_registry.add_entry(
        name="parent",
        object_cls=ToyObject,
        default_config={"child_config": {}},
        sub_builds=(
            SubBuildDeclaration(
                source_key="child_config",
                target_key="child",
                builder=child_builder,
                build_method="one",
                type_name="child",
                forwarded_kwargs=("parent_only",),
            ),
        ),
    )
    parent_builder = RegistryBuilder(registry=parent_registry)

    obj = parent_builder.build_one(
        config={"child_config": {"value": 5}},
        name="parent",
        parent_only=2,
        ignored_context=10,
    )

    assert isinstance(obj.child, StrictToyObject)
    assert obj.child.value == 7
    assert "ignored_context" not in obj.kwargs


def test_builder_resolves_phase_single_named_sub_build() -> None:
    child_registry = make_toy_registry()
    child_builder = RegistryBuilder(registry=child_registry)
    parent_registry = Registry(object_family="parent")
    parent_registry.add_entry(
        name="parent",
        object_cls=ToyObject,
        default_config={"children": {}},
        sub_builds=(
            SubBuildDeclaration(
                source_key="children",
                target_key="child",
                builder=child_builder,
                build_method="phase_single_named",
            ),
        ),
    )
    parent_builder = RegistryBuilder(registry=parent_registry)

    obj = parent_builder.build_one(
        config={
            "children": {
                "train": {"toy": {"value": 1}},
                "val": None,
            },
        },
        name="parent",
    )

    assert obj.child["train"].value == 1
    assert obj.child["val"] is None
