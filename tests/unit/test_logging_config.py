"""
Unit tests for the public logging API (:mod:`xai4tsc.logging_config`).

Covers console/file handler installation, directory-aware destination
resolution, level coercion, idempotency, additive multi-sink behaviour
(add_file_log / remove_file_log / file_log), and that disable_logging()
restores the silent-by-default state (NullHandler preserved).
"""

import logging

import pytest

import xai4tsc
from xai4tsc.logging_config import _resolve_log_file

_LOGGER_NAME = "xai4tsc"


def _managed_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Return the handlers this package installed on ``logger``."""
    return [h for h in logger.handlers if getattr(h, "_xai4tsc_managed", False)]


def _file_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Return the managed file-sink handlers on ``logger``."""
    return [
        h
        for h in _managed_handlers(logger)
        if getattr(h, "_xai4tsc_kind", None) == "file"
    ]


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
def test_file_destination_directory(tmp_path):
    logger = xai4tsc.enable_logging(tmp_path)
    log_file = tmp_path / "xai4tsc.log"
    assert log_file.exists()
    logger.info("hello file")
    for handler in _managed_handlers(logger):
        handler.flush()
    assert "hello file" in log_file.read_text()


@pytest.mark.unit
def test_file_destination_explicit_path(tmp_path):
    target = tmp_path / "run.log"
    xai4tsc.enable_logging(target)
    assert target.exists()
    assert not (tmp_path / "xai4tsc.log").exists()


@pytest.mark.unit
def test_resolve_log_file_directory_vs_file(tmp_path):
    # Existing directory -> default filename appended.
    assert _resolve_log_file(tmp_path) == tmp_path / "xai4tsc.log"
    # Suffix-less, non-existent path -> treated as a directory.
    assert _resolve_log_file(tmp_path / "logs") == tmp_path / "logs" / "xai4tsc.log"
    # Path with a suffix -> used as-is.
    assert _resolve_log_file(tmp_path / "run.log") == tmp_path / "run.log"


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
def test_idempotent_no_handler_accumulation(tmp_path):
    logger = xai4tsc.enable_logging(tmp_path)
    xai4tsc.enable_logging(tmp_path)
    # One console + one file handler, not doubled.
    assert len(_managed_handlers(logger)) == 2


@pytest.mark.unit
def test_add_file_log_accumulates(tmp_path):
    logger = xai4tsc.enable_logging()
    xai4tsc.add_file_log(tmp_path / "a.log")
    xai4tsc.add_file_log(tmp_path / "b.log")
    # Two distinct file sinks, plus the console handler still present.
    assert len(_file_handlers(logger)) == 2
    assert (tmp_path / "a.log").exists()
    assert (tmp_path / "b.log").exists()


@pytest.mark.unit
def test_enable_logging_is_non_destructive_to_file_sinks(tmp_path):
    logger = xai4tsc.enable_logging()
    handler = xai4tsc.add_file_log(tmp_path / "keep.log")
    # A second enable_logging() must not sweep the file sink or duplicate console.
    xai4tsc.enable_logging()
    assert handler in logger.handlers
    assert len(_file_handlers(logger)) == 1
    consoles = [
        h
        for h in _managed_handlers(logger)
        if getattr(h, "_xai4tsc_kind", None) == "console"
    ]
    assert len(consoles) == 1


@pytest.mark.unit
def test_add_file_log_dedups_same_path(tmp_path):
    xai4tsc.enable_logging()
    first = xai4tsc.add_file_log(tmp_path / "x.log")
    second = xai4tsc.add_file_log(tmp_path / "x.log")
    assert first is second
    assert len(_file_handlers(logging.getLogger(_LOGGER_NAME))) == 1


@pytest.mark.unit
def test_remove_file_log_by_handle_and_path(tmp_path):
    logger = xai4tsc.enable_logging()
    handler = xai4tsc.add_file_log(tmp_path / "byhandle.log")
    xai4tsc.remove_file_log(handler)
    assert _file_handlers(logger) == []

    xai4tsc.add_file_log(tmp_path / "bypath.log")
    xai4tsc.remove_file_log(tmp_path / "bypath.log")
    assert _file_handlers(logger) == []


@pytest.mark.unit
def test_remove_file_log_none_is_noop(tmp_path):
    logger = xai4tsc.enable_logging()
    xai4tsc.add_file_log(tmp_path / "keep.log")
    xai4tsc.remove_file_log(None)
    assert len(_file_handlers(logger)) == 1


@pytest.mark.unit
def test_file_log_context_manager_detaches(tmp_path):
    logger = xai4tsc.enable_logging()
    with xai4tsc.file_log(tmp_path / "scoped.log") as handler:
        assert handler in logger.handlers
    assert _file_handlers(logger) == []


@pytest.mark.unit
def test_file_log_context_manager_detaches_on_error(tmp_path):
    logger = xai4tsc.enable_logging()
    with pytest.raises(ValueError, match="boom"), xai4tsc.file_log(tmp_path / "e.log"):
        raise ValueError("boom")
    assert _file_handlers(logger) == []
