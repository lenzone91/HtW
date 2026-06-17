#############################################
# Config error type
#############################################

# The config layer is strict. Composition and resolution failures raise this
# semantic error immediately. There is no permissive mode.


class ConfigError(RuntimeError):
    """
    Raised for config-layer contract violations.

    Examples: unsupported file type, non-dict config, unresolved interpolation,
    missing mandatory value, OmegaConf object leaking past the plain-dict
    boundary.
    """
