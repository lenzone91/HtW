#############################################
# Factory error types
#############################################

# The factory is strict. There is no warn-or-continue mode.
# Every contract violation raises one of these semantic errors immediately.


class FactoryError(RuntimeError):
    """Base class for all registry and builder errors."""


class RegistryError(FactoryError):
    """
    Raised for registry-level contract violations.

    Examples: malformed registry entry, duplicate registration, unknown name.
    """


class BuilderError(FactoryError):
    """
    Raised for builder-level contract violations.

    Examples: non-dict config, unknown config key, missing required key,
    invalid sub-build declaration, wrong constructed object type.
    """
