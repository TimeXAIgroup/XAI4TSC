"""
Logging configuration for the xai4tsc standalone runner.

Thin adapters over the package's public logging API
(:func:`xai4tsc.enable_logging`) so handler, formatter, and level setup live in
a single place. Targets the ``'xai4tsc'`` named logger only — never the root
logger — so the package remains silent when imported outside the runner.
"""

import logging
from pathlib import Path

from xai4tsc.logging_config import add_file_log, enable_logging


def _setup_console_logging(debug: bool) -> None:
    """
    Configure console (stdout) logging for the xai4tsc standalone runner.

    Parameters
    ----------
    debug : bool
        If ``True``, set log level to DEBUG and include logger name in output.
    """
    enable_logging(level=logging.DEBUG if debug else logging.INFO)


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
    add_file_log(log_path, level=logging.DEBUG if debug else logging.INFO)
