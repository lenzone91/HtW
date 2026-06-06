import importlib


#############################################
# Environment setup
#############################################

# This file centralizes general environment checks.
#
# It does not install packages.
# It does not configure hardware.
# It does not configure reproducibility.
# It only checks that the core Python environment is usable.


def setup_environment(environment_config: dict | None = None) -> dict:
    """
    Validate general environment requirements and return environment context.
    """
    if environment_config is None:
        environment_config = {}

    required_imports = environment_config.get("required_imports", ())

    check_required_imports(required_imports)

    return {
        "required_imports": tuple(required_imports),
    }


#############################################
# Import checks
#############################################

# Optional dependencies should usually be checked where they are used.
# This avoids maintaining a fragile global optional / required distinction.


def check_required_imports(required_imports: list[str] | tuple[str, ...]) -> None:
    """
    Check that all required import names are importable.
    """
    for import_name in required_imports:
        check_required_import(import_name)


def check_required_import(import_name: str) -> None:
    """
    Check that one required import name is importable.
    """
    try:
        importlib.import_module(import_name)

    except ImportError as error:
        raise ImportError(
            f"Required import {import_name} is not available."
        ) from error