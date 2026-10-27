"""
Unit tests for the public logging API (:mod:`xai4tsc.logging_config`).

Covers console and file handler installation, level coercion, that repeated
calls replace rather than accumulate handlers, and that disable_logging()
restores the silent-by-default state (NullHandler preserved).
"""

import logging

import pytest

import xai4tsc

_LOGGER_NAME = "xai4tsc"


def _managed_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Return the handlers this package installed on ``logger``."""
    return [h for h in logger.handlers if getattr(h, "_xai4tsc_managed", False)]


@pytest.fixture(autouse=True)
def _restore_logging():
    """Reset xai4tsc logging state before and after each test."""
    xai4tsc.disable_logging()
    yield
    xai4tsc.disable_logging()


@pytest.mark.unit
def test_enable_adds_console_handler():
    logger = xai4tsc.enable_logging()
    console = _managed_handlers(logger)
    assert len(console) == 1
    assert isinstance(console[0], logging.StreamHandler)
    assert logger.level == logging.INFO
    assert logger.propagate is False


@pytest.mark.unit
def test_disable_removes_managed_but_keeps_nullhandler():
    logger = logging.getLogger(_LOGGER_NAME)
    xai4tsc.enable_logging()
    xai4tsc.disable_logging()
    assert _managed_handlers(logger) == []
    # The library NullHandler installed at import time must survive.
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
    assert logger.propagate is True
    assert logger.level == logging.NOTSET


@pytest.mark.unit
def test_file_destination_is_the_log_file(tmp_path):
    target = tmp_path / "logs" / "run.log"
    logger = xai4tsc.enable_logging(target)
    logger.info("hello file")
    for handler in _managed_handlers(logger):
        handler.flush()
    assert "hello file" in target.read_text()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("warning", logging.WARNING),
        (logging.ERROR, logging.ERROR),
    ],
)
def test_level_coercion(level, expected):
    logger = xai4tsc.enable_logging(level=level)
    assert logger.level == expected


@pytest.mark.unit
def test_repeat_calls_replace_their_handlers(tmp_path):
    logger = xai4tsc.enable_logging(tmp_path / "a.log")
    xai4tsc.enable_logging(tmp_path / "a.log")
    # One console + one file handler, not doubled.
    assert len(_managed_handlers(logger)) == 2
    # A call without a destination drops the earlier file handler.
    xai4tsc.enable_logging()
    assert len(_managed_handlers(logger)) == 1


@pytest.mark.unit
def test_enable_logging_leaves_foreign_handlers_alone(tmp_path):
    logger = logging.getLogger(_LOGGER_NAME)
    foreign = logging.FileHandler(tmp_path / "own.log")
    logger.addHandler(foreign)
    try:
        xai4tsc.enable_logging()
        xai4tsc.disable_logging()
        assert foreign in logger.handlers
    finally:
        logger.removeHandler(foreign)
        foreign.close()
