"""
Logging configuration for the xai4tsc standalone runner.

Targets the ``'xai4tsc'`` named logger only — never the root logger — so the
package remains silent when imported outside the runner context.
"""

import logging
import sys
from pathlib import Path


def _make_formatter(debug: bool) -> logging.Formatter:
    """
    Return a log formatter for the given verbosity level.

    Parameters
    ----------
    debug : bool
        If ``True``, include the logger name in the format string.

    Returns
    -------
    logging.Formatter
        Configured formatter instance.
    """
    fmt = (
        "%(asctime)s, %(name)s, %(levelname)s: %(message)s"
        if debug
        else "%(asctime)s, %(levelname)s: %(message)s"
    )
    return logging.Formatter(fmt)


def _setup_console_logging(debug: bool) -> None:
    """
    Configure console (stdout) logging for the xai4tsc standalone runner.

    Targets only the ``'xai4tsc'`` logger — does not touch the root logger.

    Parameters
    ----------
    debug : bool
        If ``True``, set log level to DEBUG and include logger name in output.
    """
    loglevel = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(loglevel)
    handler.setFormatter(_make_formatter(debug))

    app_logger = logging.getLogger("xai4tsc")
    app_logger.addHandler(handler)
    app_logger.setLevel(loglevel)
    app_logger.propagate = False  # don't bubble up to root logger

    # Silence noisy third-party loggers
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.colorbar").setLevel(logging.ERROR)


def _add_file_logging(debug: bool, log_path: Path) -> None:
    """
    Add a file handler to the xai4tsc logger once the results directory is available.

    Parameters
    ----------
    debug : bool
        If ``True``, set log level to DEBUG.
    log_path : Path
        Path to the log file to write.
    """
    loglevel = logging.DEBUG if debug else logging.INFO
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(loglevel)
    handler.setFormatter(_make_formatter(debug))

    logging.getLogger("xai4tsc").addHandler(handler)
    logging.getLogger("xai4tsc").info("Now also logging to file: %s", str(log_path))
