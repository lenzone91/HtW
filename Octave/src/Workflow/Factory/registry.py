from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

from ..Utils.error import handle_error


@dataclass(frozen=True)
class SubBuildDeclaration:
    """
    Declarative description of one dependency built before an object.
    """

    source_key: str
    target_key: str
    builder: object
    build_method: str = "one"
    type_name: str | None = None
    type_field: str | None = None
    remove_source: bool = True
    forwarded_kwargs: tuple[str, ...] | None = None


@dataclass(frozen=True)
class FieldResolution:
    """
    Declarative description of one config-to-constructor conversion.
    """

    target_key: str | None = None
    resolver: Callable | None = None
    remove_source_keys: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class RegistryEntry:
    """
    Metadata describing one buildable object.
    """

    name: str
    object_cls: type
    default_config: dict
    type_field: str | None = None
    sub_builds: tuple[SubBuildDeclaration, ...] = field(default_factory=tuple)
    field_resolutions: tuple[FieldResolution, ...] = field(default_factory=tuple)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.default_config = deepcopy(self.default_config)
        self.sub_builds = tuple(self.sub_builds)
        self.field_resolutions = tuple(self.field_resolutions)
        self.metadata = deepcopy(self.metadata)


class Registry:
    """
    Registry of buildable objects for one object family.
    """

    def __init__(self, object_family: str) -> None:
        self.object_family = object_family
        self.entries = {}

    def register_class(
        self,
        name: str,
        default_config: dict,
        type_field: str | None = None,
        sub_builds: tuple[SubBuildDeclaration, ...] = (),
        field_resolutions: tuple[FieldResolution, ...] = (),
        metadata: dict | None = None,
    ) -> Callable:
        """
        Decorator registering a class as buildable.
        """
        def decorator(object_cls: type) -> type:
            self.add_entry(
                name=name,
                object_cls=object_cls,
                default_config=default_config,
                type_field=type_field,
                sub_builds=sub_builds,
                field_resolutions=field_resolutions,
                metadata=metadata,
            )
            return object_cls

        return decorator

    def add_entry(
        self,
        name: str,
        object_cls: type,
        default_config: dict,
        type_field: str | None = None,
        sub_builds: tuple[SubBuildDeclaration, ...] = (),
        field_resolutions: tuple[FieldResolution, ...] = (),
        metadata: dict | None = None,
    ) -> RegistryEntry | None:
        if metadata is None:
            metadata = {}

        if not self.check_entry_arguments(
            name=name,
            object_cls=object_cls,
            default_config=default_config,
            sub_builds=sub_builds,
            field_resolutions=field_resolutions,
            metadata=metadata,
        ):
            return None

        if self.has(name):
            self.handle_error(f"Duplicate {self.object_family} registration: {name}.")
            return None

        entry = RegistryEntry(
            name=name,
            object_cls=object_cls,
            default_config=default_config,
            type_field=type_field,
            sub_builds=sub_builds,
            field_resolutions=field_resolutions,
            metadata=metadata,
        )
        self.entries[name] = entry
        return entry

    def has(self, name: str) -> bool:
        return name in self.entries

    def get_entry(self, name: str) -> RegistryEntry | None:
        if self.has(name):
            return self.entries[name]

        self.handle_unknown_name(name)
        return None

    def get_default_config(self, name: str) -> dict | None:
        entry = self.get_entry(name)
        if entry is None:
            return None

        return deepcopy(entry.default_config)

    def available_names(self) -> list[str]:
        return sorted(self.entries)

    def check_entry_arguments(
        self,
        name: str,
        object_cls: type,
        default_config: dict,
        sub_builds: tuple[SubBuildDeclaration, ...],
        field_resolutions: tuple[FieldResolution, ...],
        metadata: dict,
    ) -> bool:
        if not isinstance(name, str) or name == "":
            self.handle_error("Registry entry name must be a non-empty string.")
            return False

        if not isinstance(object_cls, type):
            self.handle_error(
                f"Registry entry object_cls must be a class, got {type(object_cls)}."
            )
            return False

        if not isinstance(default_config, dict):
            self.handle_error(
                "Registry entry default_config must be a dictionary, "
                f"got {type(default_config)}."
            )
            return False

        if not isinstance(sub_builds, tuple):
            self.handle_error(
                f"Registry entry sub_builds must be a tuple, got {type(sub_builds)}."
            )
            return False

        for sub_build in sub_builds:
            if not isinstance(sub_build, SubBuildDeclaration):
                self.handle_error(
                    "Registry entry sub_builds must contain "
                    f"SubBuildDeclaration objects, got {type(sub_build)}."
                )
                return False

        if not isinstance(field_resolutions, tuple):
            self.handle_error(
                "Registry entry field_resolutions must be a tuple, "
                f"got {type(field_resolutions)}."
            )
            return False

        for field_resolution in field_resolutions:
            if not isinstance(field_resolution, FieldResolution):
                self.handle_error(
                    "Registry entry field_resolutions must contain "
                    f"FieldResolution objects, got {type(field_resolution)}."
                )
                return False

        if not isinstance(metadata, dict):
            self.handle_error(
                f"Registry entry metadata must be a dictionary, got {type(metadata)}."
            )
            return False

        return True

    def handle_unknown_name(self, name: str) -> None:
        self.handle_error(
            f"Unknown {self.object_family} '{name}'. "
            f"Available {self.object_family}s are: {self.available_names()}."
        )

    def handle_error(self, message: str) -> None:
        handle_error(message)
