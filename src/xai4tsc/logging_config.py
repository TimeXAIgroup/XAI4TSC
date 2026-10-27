"""
Public logging controls for the xai4tsc package.

By default the package is silent: :mod:`xai4tsc` installs only a
:class:`logging.NullHandler` on the ``'xai4tsc'`` logger. Call
:func:`enable_logging` to attach a console handler (and, optionally, a file
handler) and :func:`disable_logging` to return the package to its silent
default. Both target the ``'xai4tsc'`` named logger only and never touch the
root logger, so enabling logging here does not affect the host application's
own logging configuration. More sinks are plain :mod:`logging` handlers added
to ``logging.getLogger("xai4tsc")``.
"""

import logging
import sys
from pathlib import Path

_LOGGER_NAME = "xai4tsc"
# Marker attribute set on handlers this module creates, so it replaces or
# removes only its own handlers and never the library's NullHandler.
_MANAGED_ATTR = "_xai4tsc_managed"


def _make_formatter(level: int) -> logging.Formatter:
    """
    Return a log formatter for the given verbosity level.

    Parameters
    ----------
    level : int
        Numeric logging level. When at or below :data:`logging.DEBUG`, the
        logger name is included in the format string.

    Returns
    -------
    logging.Formatter
        Configured formatter instance.
    """
    fmt = (
        "%(asctime)s, %(name)s, %(levelname)s: %(message)s"
        if level <= logging.DEBUG
        else "%(asctime)s, %(levelname)s: %(message)s"
    )
    return logging.Formatter(fmt)


def _coerce_level(level: int | str) -> int:
    """
    Coerce a logging level given as a name or number to its numeric value.

    Parameters
    ----------
    level : int or str
        A level name (e.g. ``"DEBUG"``, case-insensitive) or a numeric level
        (e.g. :data:`logging.WARNING`).

    Returns
    -------
    int
        The numeric logging level.
    """
    if isinstance(level, str):
        return logging.getLevelName(level.upper())
    return int(level)


def _add_managed(logger: logging.Logger, handler: logging.Handler, level: int) -> None:
    """Configure *handler* at *level*, mark it as ours and attach it to *logger*."""
    handler.setLevel(level)
    handler.setFormatter(_make_formatter(level))
    setattr(handler, _MANAGED_ATTR, True)
    logger.addHandler(handler)


def enable_logging(
    destination: str | Path | None = None,
    level: int | str = "INFO",
) -> logging.Logger:
    """
    Enable logging for the xai4tsc package.

    Attaches one console (stdout) handler to the ``'xai4tsc'`` logger and, when
    ``destination`` is given, one file handler. Each call replaces the handlers
    a previous call attached, so repeated calls never duplicate output; handlers
    added by other code are left alone. Only the ``'xai4tsc'`` logger is
    configured; the root logger is left untouched.

    Parameters
    ----------
    destination : str or pathlib.Path, optional
        Log file to write to as well. Missing parent directories are created and
        an existing file is overwritten. If ``None`` (default), only console
        logging is enabled.
    level : int or str, default="INFO"
        Logging level as a name (e.g. ``"DEBUG"``) or a numeric value (e.g.
        :data:`logging.WARNING`). At ``DEBUG`` the logger name is included in
        the log format.

    Returns
    -------
    logging.Logger
        The configured ``'xai4tsc'`` logger.
    """
    disable_logging()
    logger = logging.getLogger(_LOGGER_NAME)
    lvl = _coerce_level(level)
    logger.setLevel(lvl)
    logger.propagate = False  # don't bubble up to the root logger

    _add_managed(logger, logging.StreamHandler(sys.stdout), lvl)
    if destination is not None:
        log_file = Path(destination)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _add_managed(logger, logging.FileHandler(log_file, mode="w"), lvl)

    # Silence noisy third-party loggers.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.colorbar").setLevel(logging.ERROR)

    return logger


def disable_logging() -> None:
    """
    Disable xai4tsc logging and restore the silent-by-default state.

    Removes every handler installed by :func:`enable_logging` (leaving the
    library's :class:`logging.NullHandler` in place) and resets the
    ``'xai4tsc'`` logger's level and propagation to their defaults.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_ATTR, False):
            logger.removeHandler(handler)
            handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
