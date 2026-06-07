from abc import ABC, abstractmethod
from copy import deepcopy

from ..Utils.error import handle_error


###############################################################
# Base builder
###############################################################

# Base builder parent class that all other project-specific factory builders
# inherit from.
#
# Centralizes common tests, helpers and config-driven construction logic.


class BaseBuilder(ABC):
    """
    Base class for config-based object builders.

    Responsibilities:
    - validate common config structure;
    - optionally reject keys absent from the default config;
    - centralize strict / non-strict error handling;
    - propagate runtime_context to child builders;
    - delegate object-specific construction to build_from_config.
    """

    def __init__(
        self,
        default_config: dict,
        strict: bool = True,
        check_default_keys: bool = True,
    ):
        self.default_config = self.copy_config(default_config)
        self.strict = strict
        self.check_default_keys = check_default_keys

        self.check_config_is_dict(self.default_config)

    def __call__(
        self,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ):
        config = self.prepare_config(config)

        if not self.check_config(
            config,
            runtime_context,
            *args,
            **kwargs,
        ):
            return None

        return self.build_from_config(
            config,
            runtime_context,
            *args,
            **kwargs,
        )

    def prepare_config(self, config: dict) -> dict:
        """
        Prepare config before validation and construction.

        Default behavior:
        - copy config to avoid mutating user-provided dictionaries.

        Child builders may override this for lightweight normalization.
        """
        return self.copy_config(config)

    def check_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ) -> bool:
        """
        Run common config validation.
        """
        if not self.check_config_is_dict(config):
            return False

        if self.check_default_keys:
            if not self.check_keys_are_in_default_config(config):
                return False

        return True

    @abstractmethod
    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ):
        """
        Build the object from a prepared and validated config.

        runtime_context is optional and may be ignored by builders that do not
        require runtime facts from Setup/.
        """
        raise NotImplementedError

    def copy_config(self, config: dict) -> dict:
        """
        Copy config to avoid mutating user-provided dictionaries.
        """
        return deepcopy(config)

    def check_config_is_dict(self, config) -> bool:
        if isinstance(config, dict):
            return True

        self.handle_error(
            f"Builder config must be a dictionary, got {type(config)}."
        )
        return False

    def check_keys_are_in_default_config(self, config: dict) -> bool:
        """
        Check that user-provided keys are allowed by the default config.

        This checks inclusion, not equality:
            user keys ⊆ default config keys.
        """
        invalid_keys = set(config.keys()) - set(self.default_config.keys())

        if len(invalid_keys) == 0:
            return True

        self.handle_error(
            "Invalid config keys: "
            f"{sorted(invalid_keys)}. "
            f"Allowed keys are: {sorted(self.default_config.keys())}."
        )
        return False

    def handle_error(self, message: str) -> None:
        """
        Raise in strict mode, warn otherwise.
        """
        handle_error(message, self.strict)


###############################################################
# Base builder dispatcher
###############################################################

# Base builder dispatcher parent class that all other project-specific factory
# builder dispatchers inherit from.
#
# Centralizes common tests, helpers and dispatch logic.


class BaseBuilderDispatcher(BaseBuilder):
    """
    Base class for dispatching named configs to object-specific builders.

    Expected input:
        configs[object_name] = object_config

    The builder registry defines which objects are buildable:
        builder_registry[object_name] = builder
    """

    def __init__(
        self,
        default_config: dict,
        builder_registry: dict,
        strict: bool = True,
        check_default_keys: bool = True,
        check_registered_names: bool = True
    ):
        super().__init__(
            default_config=default_config,
            strict=strict,
            check_default_keys=check_default_keys,
        )

        self.builder_registry = builder_registry
        self.check_registered_names = check_registered_names

        self.check_builder_registry()
        self.builder_registry = dict(builder_registry)

    def check_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ) -> bool:
        """
        Run dispatcher-level config validation.
        """
        if not super().check_config(
            config,
            runtime_context,
            *args,
            **kwargs,
        ):
            return False

        for object_name, object_config in config.items():
            if not self.check_object_config(object_name, object_config):
                return False

            if self.check_registered_names:
                if not self.check_object_name_is_registered(object_name):
                    return False

        return True

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ) -> dict:
        """
        Build all objects from validated named configs.
        """
        objects = {}

        for object_name, object_config in config.items():
            objects[object_name] = self.build_one(
                object_name,
                object_config,
                runtime_context,
                *args,
                **kwargs,
            )

        return objects

    def build_one(
        self,
        object_name: str,
        object_config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ):
        """
        Build one object using its registered builder.
        """
        builder = self.builder_registry[object_name]

        return builder(
            object_config,
            runtime_context,
            *args,
            **kwargs,
        )

    def check_builder_registry(self) -> bool:
        """
        Check that the builder registry is valid.
        """
        if not isinstance(self.builder_registry, dict):
            self.handle_error(
                f"builder_registry must be a dictionary, got {type(self.builder_registry)}."
            )
            return False

        for object_name, builder in self.builder_registry.items():
            if not isinstance(object_name, str):
                self.handle_error(
                    f"Builder registry object names must be strings, got {type(object_name)}."
                )
                return False

            if not callable(builder):
                self.handle_error(
                    f"Builder for object '{object_name}' must be callable, got {type(builder)}."
                )
                return False

        return True

    def handle_unknown_object(self, object_name: str) -> None:
        """
        Handle the case where no builder is registered for object_name.

        Can be overridden by child dispatchers to customize the error message.
        """
        self.handle_error(
            f"Unknown object '{object_name}'. "
            f"Available objects are: {sorted(self.builder_registry.keys())}."
        )

    def check_object_config(
        self,
        object_name: str,
        object_config: dict,
    ) -> bool:
        """
        Check one named object config before building.

        Default behavior only checks that the object config is a dictionary.
        Name/registry checks are separated so child dispatchers can reuse this
        method when object_name is not the registry key.
        """
        return self.check_config_is_dict(object_config)

    def check_object_name_is_registered(self, object_name: str) -> bool:
        """
        Check that object_name is registered as a buildable object.
        """
        if object_name not in self.builder_registry:
            self.handle_unknown_object(object_name)
            return False

        return True