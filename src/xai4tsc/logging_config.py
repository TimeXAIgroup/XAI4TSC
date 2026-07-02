"""
Public logging controls for the xai4tsc package.

By default the package is silent: :mod:`xai4tsc` installs only a
:class:`logging.NullHandler` on the ``'xai4tsc'`` logger. Call
:func:`enable_logging` to attach a console handler (and, optionally, a file
handler) and :func:`disable_logging` to return the package to its silent
default. Additional file sinks can be attached and detached independently with
:func:`add_file_log` / :func:`remove_file_log` (or the :func:`file_log` context
manager) without disturbing the console or other file sinks. All functions
target the ``'xai4tsc'`` named logger only and never touch the root logger, so
enabling logging here does not affect the host application's own logging
configuration.
"""

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_LOGGER_NAME = "xai4tsc"
_DEFAULT_LOG_FILENAME = "xai4tsc.log"
# Marker attribute set on handlers this module creates, so disable_logging()
# removes only our handlers and never the library's NullHandler.
_MANAGED_ATTR = "_xai4tsc_managed"
# Kind marker distinguishing our console handler from file-sink handlers, so the
# console can be found and updated in place without disturbing file sinks.
_KIND_ATTR = "_xai4tsc_kind"
_KIND_CONSOLE = "console"
_KIND_FILE = "file"


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


def _resolve_log_file(destination: str | Path) -> Path:
    """
    Resolve a log destination to a concrete file path.

    A destination that names an existing directory, or that has no file
    suffix, is treated as a directory and the default log filename is
    appended; otherwise the destination is used as the log file path itself.

    Parameters
    ----------
    destination : str or pathlib.Path
        Directory to place ``xai4tsc.log`` in, or an explicit log file path.

    Returns
    -------
    pathlib.Path
        The concrete log file path to write to.
    """
    path = Path(destination)
    if path.is_dir() or path.suffix == "":
        return path / _DEFAULT_LOG_FILENAME
    return path


def _find_managed(logger: logging.Logger, kind: str) -> list[logging.Handler]:
    """
    Return this module's managed handlers of a given kind on ``logger``.

    Parameters
    ----------
    logger : logging.Logger
        The logger whose handlers to inspect.
    kind : str
        Handler kind to match: ``"console"`` or ``"file"``.

    Returns
    -------
    list of logging.Handler
        The managed handlers of the requested kind, in attachment order.
    """
    return [
        h
        for h in logger.handlers
        if getattr(h, _MANAGED_ATTR, False) and getattr(h, _KIND_ATTR, None) == kind
    ]


def enable_logging(
    destination: str | Path | None = None,
    level: int | str = "INFO",
) -> logging.Logger:
    """
    Enable logging for the xai4tsc package.

    Ensures a single console (stdout) handler on the ``'xai4tsc'`` logger and,
    when ``destination`` is given, also attaches a file handler via
    :func:`add_file_log`. This is additive and idempotent: calling it repeatedly
    updates the existing console handler in place rather than duplicating it, and
    it never removes file sinks attached by :func:`add_file_log`. Only the
    ``'xai4tsc'`` logger is configured; the root logger is left untouched.

    Parameters
    ----------
    destination : str or pathlib.Path, optional
        If provided, also log to a file. When it names an existing directory or
        has no file suffix, logs are written to ``<destination>/xai4tsc.log``;
        otherwise it is used as the log file path directly. Missing parent
        directories are created. If ``None`` (default), only console logging is
        enabled.
    level : int or str, default="INFO"
        Logging level as a name (e.g. ``"DEBUG"``) or a numeric value (e.g.
        :data:`logging.WARNING`). At ``DEBUG`` the logger name is included in
        the log format.

    Returns
    -------
    logging.Logger
        The configured ``'xai4tsc'`` logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    lvl = _coerce_level(level)
    logger.setLevel(lvl)
    logger.propagate = False  # don't bubble up to the root logger

    # Ensure exactly one managed console handler, updating it in place so
    # repeat calls neither duplicate it nor disturb existing file sinks.
    consoles = _find_managed(logger, _KIND_CONSOLE)
    if consoles:
        console = consoles[0]
    else:
        console = logging.StreamHandler(sys.stdout)
        setattr(console, _MANAGED_ATTR, True)
        setattr(console, _KIND_ATTR, _KIND_CONSOLE)
        logger.addHandler(console)
    console.setLevel(lvl)
    console.setFormatter(_make_formatter(lvl))

    # Silence noisy third-party loggers.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.colorbar").setLevel(logging.ERROR)

    if destination is not None:
        add_file_log(destination, level=lvl)

    return logger


def add_file_log(
    destination: str | Path,
    level: int | str | None = None,
) -> logging.Handler:
    """
    Attach a file handler to the xai4tsc logger without disturbing others.

    Unlike :func:`enable_logging`, this is purely additive: multiple file sinks
    can coexist (e.g. a global log plus one per model). Attaching the same
    resolved path twice returns the existing handler rather than duplicating it.

    Parameters
    ----------
    destination : str or pathlib.Path
        Where to write. When it names an existing directory or has no file
        suffix, logs go to ``<destination>/xai4tsc.log``; otherwise it is used
        as the log file path directly. Missing parent directories are created.
    level : int or str, optional
        Logging level for this sink as a name or numeric value. If ``None``
        (default), the logger's current level is inherited.

    Returns
    -------
    logging.Handler
        The attached (or already-existing) file handler. Pass it to
        :func:`remove_file_log` to detach this sink.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    log_file = _resolve_log_file(destination)
    # Match logging.FileHandler.baseFilename, which is os.path.abspath (no
    # symlink resolution — matters on macOS where /tmp -> /private/tmp).
    base = os.path.abspath(log_file)

    # Deduplicate: reuse an existing managed file handler for the same path.
    for handler in _find_managed(logger, _KIND_FILE):
        if getattr(handler, "baseFilename", None) == base:
            return handler

    lvl = _coerce_level(level) if level is not None else (logger.level or logging.INFO)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(lvl)
    file_handler.setFormatter(_make_formatter(lvl))
    setattr(file_handler, _MANAGED_ATTR, True)
    setattr(file_handler, _KIND_ATTR, _KIND_FILE)
    logger.addHandler(file_handler)
    logger.info("Now also logging to file: %s", log_file)
    return file_handler


def remove_file_log(target: logging.Handler | str | Path | None) -> None:
    """
    Detach a file sink previously attached by :func:`add_file_log`.

    Parameters
    ----------
    target : logging.Handler or str or pathlib.Path or None
        The handler returned by :func:`add_file_log`, or the path it was
        attached for (resolved the same directory-aware way). ``None`` is a
        no-op, so a ``finally`` block can call this unconditionally.
    """
    if target is None:
        return
    logger = logging.getLogger(_LOGGER_NAME)

    if isinstance(target, logging.Handler):
        if getattr(target, _MANAGED_ATTR, False):
            logger.removeHandler(target)
            target.close()
        return

    base = os.path.abspath(_resolve_log_file(target))
    for handler in _find_managed(logger, _KIND_FILE):
        if getattr(handler, "baseFilename", None) == base:
            logger.removeHandler(handler)
            handler.close()


@contextmanager
def file_log(
    destination: str | Path,
    level: int | str | None = None,
) -> Iterator[logging.Handler]:
    """
    Context manager that attaches a file sink for the duration of a block.

    Attaches via :func:`add_file_log` on entry and detaches via
    :func:`remove_file_log` on exit, even if the block raises.

    Parameters
    ----------
    destination : str or pathlib.Path
        File or directory to log to (see :func:`add_file_log`).
    level : int or str, optional
        Level for this sink; inherits the logger's level when ``None``.

    Yields
    ------
    logging.Handler
        The attached file handler.
    """
    handler = add_file_log(destination, level=level)
    try:
        yield handler
    finally:
        remove_file_log(handler)


def disable_logging() -> None:
    """
    Disable xai4tsc logging and restore the silent-by-default state.

    Removes every handler installed by :func:`enable_logging` and
    :func:`add_file_log` (leaving the library's :class:`logging.NullHandler` in
    place) and resets the ``'xai4tsc'`` logger's level and propagation to their
    defaults.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_ATTR, False):
            logger.removeHandler(handler)
            handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
