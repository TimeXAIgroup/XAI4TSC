"""Validation of enum-like string options against their ``Literal`` type."""

from typing import get_args


def _check_option[T](name: str, value: T, options: object) -> T:
    """
    Validate an enum-like option, returning it unchanged.

    Called from constructors, so a typo fails when the object is built rather
    than when the option is first read.

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : T
        The value given.
    options : object
        The ``Literal[...]`` type listing the allowed values.

    Returns
    -------
    T
        *value*.

    Raises
    ------
    ValueError
        If *value* is not one of the allowed values.
    """
    allowed = get_args(options)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}.")
    return value
