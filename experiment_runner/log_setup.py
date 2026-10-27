"""
Logging configuration for the xai4tsc standalone runner.

The console comes from the package's :func:`xai4tsc.enable_logging`; the runner
adds colour to it and attaches its own file sinks: one ``experiment.log`` per run
and one log per model, each detached when its part of the run ends. Everything
targets the ``'xai4tsc'`` named logger only - never the root logger - so the
package remains silent when imported outside the runner.

Console output tints whole lines above ``INFO`` when the stream is a terminal,
following the ``NO_COLOR`` and ``FORCE_COLOR`` conventions, so piped or redirected
output stays plain. ``INFO`` lines are left uncoloured, since they are the bulk of
the output. File sinks are never coloured, so a log file holds no escape codes.
"""

import logging
import os
from pathlib import Path

from xai4tsc import enable_logging

_LOGGER_NAME = "xai4tsc"

# ANSI SGR codes for the levels worth noticing. INFO is deliberately absent: it is
# the bulk of the output, so colouring it would leave nothing standing out.
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """
    Formatter that tints a whole line, for console output only.

    Records at ``INFO`` are left plain, so the lines carrying colour are exactly
    the ones that want attention.  Everything else is wrapped end to end -
    timestamp, level and message - which is both the most visible option and works
    with grep: the escape codes sit only at the line's edges, so a pattern spanning
    the timestamp, level and message (``"WARNING: Config"``) still matches. Colouring
    the parts separately would put a reset between the level name and the colon that
    follows it, and break exactly that.

    The record is never mutated, so other handlers - a file sink in particular -
    are unaffected and receive no escape codes.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render *record*, wrapping the whole line in its level's colour."""
        text = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno)
        return f"{color}{text}{_RESET}" if color else text


def _supports_color(stream: object) -> bool:
    """
    Return whether *stream* should receive ANSI colour.

    Honours the ``NO_COLOR`` and ``FORCE_COLOR`` conventions, and otherwise
    colours only a real terminal, so piped or redirected output stays plain.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _make_formatter(level: int, color: bool = False) -> logging.Formatter:
    """
    Return a log formatter for the given verbosity level.

    Parameters
    ----------
    level : int
        Numeric logging level. When at or below :data:`logging.DEBUG`, the
        logger name is included in the format string.
    color : bool
        Tint lines with ANSI codes. Only ever true for the console handler; a
        file sink must stay free of escape codes.

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
    return _ColorFormatter(fmt) if color else logging.Formatter(fmt)


def _setup_console_logging(debug: bool) -> None:
    """
    Configure console (stdout) logging for the xai4tsc standalone runner.

    Parameters
    ----------
    debug : bool
        If ``True``, set log level to DEBUG and include logger name in output.
    """
    level = logging.DEBUG if debug else logging.INFO
    logger = enable_logging(level=level)
    for handler in logger.handlers:
        # The package's console handler; a FileHandler is a StreamHandler subclass.
        if type(handler) is logging.StreamHandler:
            color = _supports_color(handler.stream)
            handler.setFormatter(_make_formatter(level, color=color))


def add_file_log(log_path: Path, level: int | None = None) -> logging.Handler:
    """
    Attach a file sink to the xai4tsc logger.

    Parameters
    ----------
    log_path : Path
        Log file to write. Missing parent directories are created and an
        existing file is overwritten.
    level : int, optional
        Level for this sink. If ``None`` (default), the logger's effective level.

    Returns
    -------
    logging.Handler
        The attached handler; pass it to :func:`remove_file_log` to detach it.
        Without that, a later run or model in the same process would also write
        into this log.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    lvl = level if level is not None else logger.getEffectiveLevel()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(lvl)
    handler.setFormatter(_make_formatter(lvl))
    logger.addHandler(handler)
    logger.info("Now also logging to file: %s", log_path)
    return handler


def remove_file_log(handler: logging.Handler | None) -> None:
    """
    Detach and close a file sink attached by :func:`add_file_log`.

    Parameters
    ----------
    handler : logging.Handler or None
        The handler to detach. ``None`` is a no-op, so a ``finally`` block can
        call this unconditionally.
    """
    if handler is None:
        return
    logging.getLogger(_LOGGER_NAME).removeHandler(handler)
    handler.close()
