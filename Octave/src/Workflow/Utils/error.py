import warnings


def handle_error(message: str, strict: bool = True) -> None:
    """
    Raise in strict mode, warn otherwise.
    """
    if strict:
        raise RuntimeError(message)

    warnings.warn(message, stacklevel=2)
