from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, NoReturn

from .errors import RegistryError


#############################################
# Sub-build declaration
#############################################


# dataclass(frozen=True) gives an immutable metadata container with a generated
# __init__ and __repr__. These declarations are pure data; they hold no logic.
@dataclass(frozen=True)
class SubBuildDeclaration:
    """
    Declarative description of one dependency built before an object.

    The builder uses this metadata to turn a sub-config into a
    constructor-ready object.
    """

    source_key: str
    target_key: str
    builder: object
    build_method: str = "one"
    type_name: str | None = None
    type_field: str | None = None
    remove_source: bool = True


#############################################
# Field resolution declaration
#############################################


@dataclass(frozen=True)
class FieldResolution:
    """
    Declarative description of one config field resolution.

    Used for config values that must be transformed before construction, for
    example serializable dtype names or runtime-resolved values.
    """

    target_key: str | None = None
    resolver: Callable | None = None
    remove_source_keys: tuple[str, ...] = field(default_factory=tuple)


#############################################
# Registry entry
#############################################


@dataclass
class RegistryEntry:
    """
    Metadata describing one buildable object.
    """

    name: str
    object_cls: type
    default_config: dict
    type_field: str | None = None
    # field(default_factory=...) avoids sharing one mutable default between
    # several entries.
    sub_builds: tuple[SubBuildDeclaration, ...] = field(default_factory=tuple)
    field_resolutions: tuple[FieldResolution, ...] = field(default_factory=tuple)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.default_config = deepcopy(self.default_config)
        self.metadata = deepcopy(self.metadata)
        self.sub_builds = tuple(self.sub_builds)
        self.field_resolutions = tuple(self.field_resolutions)


#############################################
# Registry
#############################################


class Registry:
    """
    Registry of buildable objects for one project object family.

    The registry is declarative: it stores object metadata and exposes
    decorator-based registration. It does not build objects.

    The registry is strict. Every contract violation raises a RegistryError
    immediately. There is no permissive mode.
    """

    def __init__(self, object_family: str) -> None:
        self.object_family = object_family
        self.entries: dict[str, RegistryEntry] = {}

    #############################################
    # Registration
    #############################################

    def register_class(
        self,
        name: str,
        default_config: dict,
        type_field: str | None = None,
        sub_builds: tuple[SubBuildDeclaration, ...] = (),
        field_resolutions: tuple[FieldResolution, ...] = (),
        metadata: dict | None = None,
    ) -> Callable[[type], type]:
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
    ) -> RegistryEntry:
        """
        Add one registry entry. Returns the created entry.
        """
        if metadata is None:
            metadata = {}

        self.check_entry_arguments(
            name=name,
            object_cls=object_cls,
            default_config=default_config,
            sub_builds=sub_builds,
            field_resolutions=field_resolutions,
            metadata=metadata,
        )

        if self.has(name):
            raise RegistryError(
                f"Duplicate {self.object_family} registration for '{name}'."
            )

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

    #############################################
    # Lookup
    #############################################

    def has(self, name: str) -> bool:
        """
        Return whether name is registered.
        """
        return name in self.entries

    def get_entry(self, name: str) -> RegistryEntry:
        """
        Return the registry entry associated with name.
        """
        if not self.has(name):
            self.raise_unknown_name(name)

        return self.entries[name]

    def get_default_config(self, name: str) -> dict:
        """
        Return a copy of the default config associated with name.
        """
        return deepcopy(self.get_entry(name).default_config)

    def available_names(self) -> list[str]:
        """
        Return sorted registered names.
        """
        return sorted(self.entries.keys())

    #############################################
    # Validation
    #############################################

    def check_entry_arguments(
        self,
        name: str,
        object_cls: type,
        default_config: dict,
        sub_builds: tuple[SubBuildDeclaration, ...],
        field_resolutions: tuple[FieldResolution, ...],
        metadata: dict,
    ) -> None:
        """
        Validate entry metadata before registration. Raises on any violation.
        """
        if not isinstance(name, str):
            raise RegistryError(
                f"Registry entry name must be a string, got {type(name)}."
            )

        if not name:
            raise RegistryError("Registry entry name cannot be empty.")

        if not isinstance(object_cls, type):
            raise RegistryError(
                "Registry entry object_cls must be a class, "
                f"got {type(object_cls)}."
            )

        if not isinstance(default_config, dict):
            raise RegistryError(
                "Registry entry default_config must be a dictionary, "
                f"got {type(default_config)}."
            )

        if not isinstance(sub_builds, tuple):
            raise RegistryError(
                f"Registry entry sub_builds must be a tuple, got {type(sub_builds)}."
            )

        for sub_build in sub_builds:
            if not isinstance(sub_build, SubBuildDeclaration):
                raise RegistryError(
                    "Registry entry sub_builds must contain SubBuildDeclaration "
                    f"objects, got {type(sub_build)}."
                )

        if not isinstance(field_resolutions, tuple):
            raise RegistryError(
                "Registry entry field_resolutions must be a tuple, "
                f"got {type(field_resolutions)}."
            )

        for field_resolution in field_resolutions:
            if not isinstance(field_resolution, FieldResolution):
                raise RegistryError(
                    "Registry entry field_resolutions must contain "
                    f"FieldResolution objects, got {type(field_resolution)}."
                )

        if not isinstance(metadata, dict):
            raise RegistryError(
                f"Registry entry metadata must be a dictionary, got {type(metadata)}."
            )

    #############################################
    # Error helpers
    #############################################

    def raise_unknown_name(self, name: str) -> NoReturn:
        """
        Raise a semantic error for an unknown registered name.
        """
        raise RegistryError(
            f"Unknown {self.object_family} '{name}'. "
            f"Available {self.object_family}s are: {self.available_names()}."
        )
