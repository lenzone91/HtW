from copy import deepcopy

from .errors import BuilderError
from .registry import (
    FieldResolution,
    Registry,
    RegistryEntry,
    SubBuildDeclaration,
)


SUB_BUILD_METHODS = ("one", "named", "phase_single_named")


#############################################
# Registry-aware builder
#############################################


class RegistryBuilder:
    """
    Generic builder for objects declared in a Registry.

    The registry says what can be built. This builder validates configs,
    resolves declared field resolutions and sub-builds, and calls the
    registered class constructor.

    The builder is strict. Every contract violation raises a BuilderError
    immediately. There is no permissive mode.
    """

    def __init__(
        self,
        registry: Registry,
        check_default_keys: bool = True,
        type_field: str | None = None,
    ) -> None:
        # check_default_keys is a structural declaration, not an error-escape:
        # it states whether an entry's default_config is an exhaustive allow-list
        # of accepted keys. When True, unknown config keys raise.
        self.registry = registry
        self.check_default_keys = check_default_keys
        self.type_field = type_field

        self.check_registry()

    #############################################
    # Public build API
    #############################################

    def build_one(
        self,
        config: dict,
        runtime_context: dict | None = None,
        name: str | None = None,
        type_field: str | None = None,
        *args,
        **kwargs,
    ):
        """
        Build one object.

        The registry name is provided explicitly with name, or read from a type
        field in the config.
        """
        config = self.copy_config(config)
        self.check_config_is_dict(config)

        resolved_name = self.resolve_one_name(
            config=config,
            name=name,
            type_field=type_field,
        )

        return self.build_entry(
            resolved_name,
            config,
            runtime_context,
            self.resolve_type_field(type_field),
            *args,
            **kwargs,
        )

    def build_named(
        self,
        configs: dict,
        runtime_context: dict | None = None,
        type_field: str | None = None,
        *args,
        **kwargs,
    ) -> dict:
        """
        Build several named objects.

        Outer config keys are instance names. The registry name is read from
        type_field when present, otherwise the outer key is used.
        """
        configs = self.copy_config(configs)
        self.check_config_is_dict(configs)

        objects = {}

        for instance_name, object_config in configs.items():
            self.check_config_is_dict(object_config)

            registered_name = self.resolve_named_entry_name(
                instance_name=instance_name,
                config=object_config,
                type_field=type_field,
            )

            objects[instance_name] = self.build_entry(
                registered_name,
                object_config,
                runtime_context,
                self.resolve_type_field(type_field),
                *args,
                **kwargs,
            )

        return objects

    #############################################
    # Entry build path
    #############################################

    def build_entry(
        self,
        name: str,
        config: dict,
        runtime_context: dict | None = None,
        routing_type_field: str | None = None,
        *args,
        **kwargs,
    ):
        """
        Build one object from one resolved registry entry.
        """
        if not self.registry.has(name):
            self.registry.raise_unknown_name(name)

        entry = self.registry.get_entry(name)

        config = self.prepare_entry_config(
            entry=entry,
            config=config,
            runtime_context=runtime_context,
            routing_type_field=routing_type_field,
            **kwargs,
        )

        constructor_kwargs = dict(config)
        constructor_kwargs.update(kwargs)

        return entry.object_cls(*args, **constructor_kwargs)

    def prepare_entry_config(
        self,
        entry: RegistryEntry,
        config: dict,
        runtime_context: dict | None = None,
        routing_type_field: str | None = None,
        **kwargs,
    ) -> dict:
        """
        Validate and resolve config before object construction.
        """
        # Strip builder-level routing/type fields first: they are routing
        # metadata, not constructor arguments, so they must not be subject to the
        # default-config key allow-list nor reach the constructor.
        self.remove_routing_type_field(
            config=config,
            routing_type_field=routing_type_field,
            entry=entry,
        )

        self.check_entry_config(entry, config)

        config = self.resolve_field_resolutions(
            entry=entry,
            config=config,
            runtime_context=runtime_context,
            **kwargs,
        )

        config = self.resolve_sub_builds(
            entry=entry,
            config=config,
            runtime_context=runtime_context,
            **kwargs,
        )

        return config

    def resolve_field_resolutions(
        self,
        entry: RegistryEntry,
        config: dict,
        runtime_context: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Resolve declared config fields before sub-builds and construction.
        """
        for field_resolution in entry.field_resolutions:
            config = self.resolve_one_field(
                field_resolution=field_resolution,
                config=config,
                runtime_context=runtime_context,
                **kwargs,
            )

        return config

    def resolve_one_field(
        self,
        field_resolution: FieldResolution,
        config: dict,
        runtime_context: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Resolve one declared field into constructor-ready config.

        Resolver contract: resolver(config, runtime_context, **kwargs) -> value.
        """
        if field_resolution.resolver is not None:
            value = field_resolution.resolver(
                config=config,
                runtime_context=runtime_context,
                **kwargs,
            )

            if field_resolution.target_key is not None:
                config[field_resolution.target_key] = value

        for source_key in field_resolution.remove_source_keys:
            config.pop(source_key, None)

        return config

    def remove_routing_type_field(
        self,
        config: dict,
        routing_type_field: str | None,
        entry: RegistryEntry,
    ) -> None:
        """
        Remove route-only type fields before constructor invocation.
        """
        if routing_type_field is not None:
            config.pop(routing_type_field, None)

        if entry.type_field is not None:
            config.pop(entry.type_field, None)

    #############################################
    # Name resolution
    #############################################

    def resolve_one_name(
        self,
        config: dict,
        name: str | None = None,
        type_field: str | None = None,
    ) -> str:
        """
        Resolve the registry name for a single-object build.
        """
        if name is not None:
            return name

        type_field = self.resolve_type_field(type_field)

        if type_field is None:
            raise BuilderError(
                "build_one requires either an explicit name or a type_field."
            )

        if type_field not in config:
            raise BuilderError(f"Missing type field '{type_field}' in config.")

        return config[type_field]

    def resolve_named_entry_name(
        self,
        instance_name: str,
        config: dict,
        type_field: str | None = None,
    ) -> str:
        """
        Resolve the registry name for one item in a named build.
        """
        type_field = self.resolve_type_field(type_field)

        if type_field is not None and type_field in config:
            return config[type_field]

        return instance_name

    def resolve_type_field(
        self,
        type_field: str | None = None,
    ) -> str | None:
        """
        Resolve the method-level type field against the builder default.
        """
        if type_field is not None:
            return type_field

        return self.type_field

    #############################################
    # Sub-build resolution
    #############################################

    def resolve_sub_builds(
        self,
        entry: RegistryEntry,
        config: dict,
        runtime_context: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Resolve declared sub-builds into constructor-ready values.
        """
        for sub_build in entry.sub_builds:
            self.check_sub_build_source_exists(sub_build, config)

            built_object = self.build_sub_object(
                sub_build,
                config[sub_build.source_key],
                runtime_context,
                **kwargs,
            )

            config[sub_build.target_key] = built_object

            if (
                sub_build.remove_source
                and sub_build.source_key != sub_build.target_key
            ):
                config.pop(sub_build.source_key, None)

        return config

    def build_sub_object(
        self,
        sub_build: SubBuildDeclaration,
        config: dict,
        runtime_context: dict | None = None,
        **kwargs,
    ):
        """
        Build one declared dependency using its linked builder.
        """
        if sub_build.build_method == "one":
            return sub_build.builder.build_one(
                config,
                runtime_context,
                sub_build.type_name,
                sub_build.type_field,
                **kwargs,
            )

        if sub_build.build_method == "named":
            return sub_build.builder.build_named(
                config,
                runtime_context,
                sub_build.type_field,
                **kwargs,
            )

        if sub_build.build_method == "phase_single_named":
            return self.build_phase_single_named_sub_objects(
                sub_build=sub_build,
                config=config,
                runtime_context=runtime_context,
                **kwargs,
            )

        raise BuilderError(
            f"Unknown sub-build method '{sub_build.build_method}'. "
            f"Expected one of {SUB_BUILD_METHODS}."
        )

    def build_phase_single_named_sub_objects(
        self,
        sub_build: SubBuildDeclaration,
        config: dict,
        runtime_context: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Build one named sub-object per phase.
        """
        phase_objects = {}

        for phase, phase_config in config.items():
            if phase_config is None:
                phase_objects[phase] = None
                continue

            built_objects = sub_build.builder.build_named(
                phase_config,
                runtime_context,
                sub_build.type_field,
                **kwargs,
            )

            if len(built_objects) != 1:
                raise BuilderError(
                    "Each phase must define exactly one named sub-object, "
                    f"got {len(built_objects)} for phase '{phase}'."
                )

            phase_objects[phase] = next(iter(built_objects.values()))

        return phase_objects

    #############################################
    # Validation
    #############################################

    def check_registry(self) -> None:
        """
        Check that the builder receives a valid registry.
        """
        if not isinstance(self.registry, Registry):
            raise BuilderError(
                f"registry must be a Registry, got {type(self.registry)}."
            )

    def check_entry_config(
        self,
        entry: RegistryEntry,
        config: dict,
    ) -> None:
        """
        Check config against the registry entry default config.
        """
        self.check_config_is_dict(config)

        if self.check_default_keys:
            self.check_keys_are_in_default_config(
                config=config,
                default_config=entry.default_config,
            )

    def check_config_is_dict(self, config) -> None:
        """
        Check that a config object is a dictionary.
        """
        if not isinstance(config, dict):
            raise BuilderError(
                f"Builder config must be a dictionary, got {type(config)}."
            )

    def check_keys_are_in_default_config(
        self,
        config: dict,
        default_config: dict,
    ) -> None:
        """
        Check that user-provided keys are allowed by the default config.
        """
        invalid_keys = set(config.keys()) - set(default_config.keys())

        if invalid_keys:
            raise BuilderError(
                f"Invalid config keys: {sorted(invalid_keys)}. "
                f"Allowed keys are: {sorted(default_config.keys())}."
            )

    def check_sub_build_source_exists(
        self,
        sub_build: SubBuildDeclaration,
        config: dict,
    ) -> None:
        """
        Check that a declared sub-build source key exists.
        """
        if sub_build.source_key not in config:
            raise BuilderError(
                f"Missing sub-build config key '{sub_build.source_key}'."
            )

    #############################################
    # Helpers
    #############################################

    def copy_config(self, config):
        """
        Copy configs to avoid mutating user-provided dictionaries.
        """
        return deepcopy(config)
