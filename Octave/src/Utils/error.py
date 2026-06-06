import warnings


#############################################
# Error handling
#############################################

# Hardware checks can either fail strictly or warn depending on config.


def handle_error(
    msg: str,
    strict: bool = True,
) -> None:
    """
    Raise an error in strict mode, otherwise emit a warning.
    """
    if strict:
        raise RuntimeError(msg)

    warnings.warn(msg)