from copy import deepcopy

from ..Utils.error import handle_error
from .registry import FieldResolution, Registry, RegistryEntry, SubBuildDeclaration


class RegistryBuilder:
    """
    Generic builder for objects declared in a Registry.
    """

    def __init__(
        self,
        registry: Registry,
        strict: bool = True,
        check_default_keys: bool = True,
        type_field: str | None = None,
    ) -> None:
        self.registry = registry
        self.strict = strict
        self.check_default_keys = check_default_keys
        self.type_field = type_field
        self.check_registry()

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
        """
        config = self.copy_config(config)
        if not self.check_config_is_dict(config):
            return None

        resolved_name = self.resolve_one_name(
            config=config,
            name=name,
            type_field=type_field,
        )
        if resolved_name is None:
            return None

        return self.build_entry(
            name=resolved_name,
            config=config,
            runtime_context=runtime_context,
            routing_type_field=self.resolve_type_field(type_field),
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
    ) -> dict | None:
        """
        Build several named objects.
        """
        configs = self.copy_config(configs)
        if not self.check_config_is_dict(configs):
            return None

        objects = {}
        for instance_name, object_config in configs.items():
            if not self.check_config_is_dict(object_config):
                return None

            registered_name = self.resolve_named_entry_name(
                instance_name=instance_name,
                config=object_config,
                type_field=type_field,
            )
            built_object = self.build_entry(
                name=registered_name,
                config=object_config,
                runtime_context=runtime_context,
                routing_type_field=self.resolve_type_field(type_field),
                *args,
                **kwargs,
            )
            if built_object is None:
                return None

            objects[instance_name] = built_object

        return objects

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
            self.handle_unknown_name(name)
            return None

        entry = self.registry.get_entry(name)
        config = self.prepare_entry_config(
            entry=entry,
            config=config,
            runtime_context=runtime_context,
            routing_type_field=routing_type_field,
            **kwargs,
        )
        if config is None:
            return None

        return entry.object_cls(*args, **config)

    def prepare_entry_config(
        self,
        entry: RegistryEntry,
        config: dict,
        runtime_context: dict | None = None,
        routing_type_field: str | None = None,
        *args,
        **kwargs,
    ) -> dict | None:
        """
        Validate and resolve config before object construction.
        """
        if not self.check_entry_config(entry, config):
            return None

        self.remove_routing_type_field(
            config=config,
            routing_type_field=routing_type_field,
            entry=entry,
        )

        config = self.resolve_field_resolutions(
            entry=entry,
            config=config,
            runtime_context=runtime_context,
            **kwargs,
        )
        if config is None:
            return None

        return self.resolve_sub_builds(
            entry=entry,
            config=config,
            runtime_context=runtime_context,
            **kwargs,
        )

    def resolve_field_resolutions(
        self,
        entry: RegistryEntry,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ) -> dict | None:
        for field_resolution in entry.field_resolutions:
            config = self.resolve_one_field(
                field_resolution=field_resolution,
                config=config,
                runtime_context=runtime_context,
                *args,
                **kwargs,
            )
            if config is None:
                return None

        return config

    def resolve_one_field(
        self,
        field_resolution: FieldResolution,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ) -> dict | None:
        if field_resolution.resolver is not None:
            value = field_resolution.resolver(
                config=config,
                runtime_context=runtime_context,
                strict=self.strict,
                *args,
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
        if routing_type_field is not None:
            config.pop(routing_type_field, None)

        if entry.type_field is not None:
            config.pop(entry.type_field, None)

    def resolve_one_name(
        self,
        config: dict,
        name: str | None = None,
        type_field: str | None = None,
    ) -> str | None:
        if name is not None:
            return name

        type_field = self.resolve_type_field(type_field)
        if type_field is None:
            self.handle_error("build_one requires either name or type_field.")
            return None

        if type_field not in config:
            self.handle_error(f"Missing type field '{type_field}' in config.")
            return None

        return config[type_field]

    def resolve_named_entry_name(
        self,
        instance_name: str,
        config: dict,
        type_field: str | None = None,
    ) -> str:
        type_field = self.resolve_type_field(type_field)
        if type_field is not None and type_field in config:
            return config[type_field]

        return instance_name

    def resolve_type_field(self, type_field: str | None = None) -> str | None:
        if type_field is not None:
            return type_field

        return self.type_field

    def resolve_sub_builds(
        self,
        entry: RegistryEntry,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ) -> dict | None:
        for sub_build in entry.sub_builds:
            if not self.check_sub_build_source_exists(sub_build, config):
                return None

            built_object = self.build_sub_object(
                sub_build=sub_build,
                config=config[sub_build.source_key],
                runtime_context=runtime_context,
                *args,
                **kwargs,
            )
            if built_object is None:
                return None

            config[sub_build.target_key] = built_object
            if sub_build.remove_source and sub_build.source_key != sub_build.target_key:
                config.pop(sub_build.source_key, None)

        return config

    def build_sub_object(
        self,
        sub_build: SubBuildDeclaration,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ):
        forwarded_kwargs = self.resolve_sub_build_kwargs(
            sub_build=sub_build,
            kwargs=kwargs,
        )
        builder = self.resolve_sub_builder(sub_build.builder)

        if sub_build.build_method == "one":
            return builder.build_one(
                config,
                runtime_context,
                sub_build.type_name,
                sub_build.type_field,
                *args,
                **forwarded_kwargs,
            )

        if sub_build.build_method == "named":
            return builder.build_named(
                config,
                runtime_context,
                sub_build.type_field,
                *args,
                **forwarded_kwargs,
            )

        if sub_build.build_method == "phase_single_named":
            return self.build_phase_single_named_sub_objects(
                sub_build=sub_build,
                config=config,
                runtime_context=runtime_context,
                *args,
                **forwarded_kwargs,
            )

        self.handle_error(
            f"Unknown sub-build method '{sub_build.build_method}'."
        )
        return None

    def resolve_sub_builder(self, builder):
        if not isinstance(builder, RegistryBuilder):
            return builder

        return RegistryBuilder(
            registry=builder.registry,
            strict=self.strict,
            check_default_keys=builder.check_default_keys,
            type_field=builder.type_field,
        )

    def resolve_sub_build_kwargs(
        self,
        sub_build: SubBuildDeclaration,
        kwargs: dict,
    ) -> dict:
        if sub_build.forwarded_kwargs is None:
            return dict(kwargs)

        return {
            key: kwargs[key]
            for key in sub_build.forwarded_kwargs
            if key in kwargs
        }

    def build_phase_single_named_sub_objects(
        self,
        sub_build: SubBuildDeclaration,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ) -> dict | None:
        builder = self.resolve_sub_builder(sub_build.builder)
        phase_objects = {}
        for phase, phase_config in config.items():
            if phase_config is None:
                phase_objects[phase] = None
                continue

            built_objects = builder.build_named(
                phase_config,
                runtime_context,
                sub_build.type_field,
                *args,
                **kwargs,
            )
            if built_objects is None:
                return None

            if len(built_objects) != 1:
                self.handle_error(
                    "Each phase must define exactly one named sub-object, "
                    f"got {len(built_objects)} for phase '{phase}'."
                )
                return None

            phase_objects[phase] = next(iter(built_objects.values()))

        return phase_objects

    def check_registry(self) -> bool:
        if isinstance(self.registry, Registry):
            return True

        self.handle_error(f"registry must be a Registry, got {type(self.registry)}.")
        return False

    def check_entry_config(self, entry: RegistryEntry, config: dict) -> bool:
        if not self.check_config_is_dict(config):
            return False

        if self.check_default_keys:
            return self.check_keys_are_in_default_config(
                config=config,
                default_config=entry.default_config,
            )

        return True

    def check_config_is_dict(self, config) -> bool:
        if isinstance(config, dict):
            return True

        self.handle_error(f"Builder config must be a dictionary, got {type(config)}.")
        return False

    def check_keys_are_in_default_config(
        self,
        config: dict,
        default_config: dict,
    ) -> bool:
        invalid_keys = set(config) - set(default_config)
        if not invalid_keys:
            return True

        self.handle_error(
            f"Invalid config keys: {sorted(invalid_keys)}. "
            f"Allowed keys are: {sorted(default_config)}."
        )
        return False

    def check_sub_build_source_exists(
        self,
        sub_build: SubBuildDeclaration,
        config: dict,
    ) -> bool:
        if sub_build.source_key in config:
            return True

        self.handle_error(f"Missing sub-build config key '{sub_build.source_key}'.")
        return False

    def copy_config(self, config: dict) -> dict:
        return deepcopy(config)

    def handle_error(self, message: str) -> None:
        handle_error(message, strict=self.strict)

    def handle_unknown_name(self, name: str) -> None:
        self.handle_error(
            f"Unknown {self.registry.object_family} '{name}'. "
            f"Available {self.registry.object_family}s are: "
            f"{self.registry.available_names()}."
        )
