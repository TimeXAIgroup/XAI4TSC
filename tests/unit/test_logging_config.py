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
from xai4tsc.logging_config import (
    _LEVEL_COLORS,
    _RESET,
    _make_formatter,
    _resolve_log_file,
    _supports_color,
)

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


# ── console colouring ─────────────────────────────────────────────────────────

_ESC = "\033"


def _render(level: int, color: bool) -> str:
    """Format one record of *level* through the console formatter."""
    formatter = _make_formatter(logging.DEBUG, color=color)
    record = logging.LogRecord(
        name="xai4tsc.test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg="the message",
        args=(),
        exc_info=None,
    )
    return formatter.format(record)


@pytest.mark.unit
def test_info_lines_are_never_coloured():
    """INFO is the bulk of the output, so colour there would signal nothing."""
    assert _ESC not in _render(logging.INFO, color=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "level", [logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL]
)
def test_other_levels_colour_the_whole_line(level):
    rendered = _render(level, color=True)
    color = _LEVEL_COLORS[level]

    # One span, wrapping the line end to end.
    assert rendered.count(color) == 1, rendered
    assert rendered.startswith(color)
    assert rendered.endswith(f"the message{_RESET}")


@pytest.mark.unit
@pytest.mark.parametrize("level", [logging.WARNING, logging.ERROR, logging.CRITICAL])
def test_coloured_lines_stay_greppable(level):
    """
    Escape codes only at the edges, so a pattern spanning fields still matches.

    Colouring the timestamp and level name separately put a reset between the
    level and its colon, which broke ``grep "WARNING:"``.
    """
    rendered = _render(level, color=True)

    assert f"{logging.getLevelName(level)}: the message" in rendered
    assert _ESC not in rendered[len(_LEVEL_COLORS[level]) : -len(_RESET)]


@pytest.mark.unit
@pytest.mark.parametrize("level", [logging.INFO, logging.WARNING, logging.ERROR])
def test_plain_formatter_emits_no_escape_codes(level):
    """What a file sink uses: never coloured, whatever the level."""
    assert _ESC not in _render(level, color=False)


@pytest.mark.unit
def test_colouring_a_record_does_not_leak_into_other_handlers():
    """The record is shared, so a file sink formatting it next must see it plain."""
    record = logging.LogRecord(
        name="xai4tsc.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="shared",
        args=(),
        exc_info=None,
    )
    _make_formatter(logging.INFO, color=True).format(record)

    assert record.levelname == "WARNING"
    assert _ESC not in _make_formatter(logging.INFO, color=False).format(record)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env", "isatty", "expected"),
    [
        ({"NO_COLOR": "1"}, True, False),
        ({"NO_COLOR": "1", "FORCE_COLOR": "1"}, True, False),
        ({"FORCE_COLOR": "1"}, False, True),
        ({}, True, True),
        ({}, False, False),
    ],
)
def test_colour_support_honours_the_conventions(env, isatty, expected, monkeypatch):
    for var in ("NO_COLOR", "FORCE_COLOR"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    stream = type("S", (), {"isatty": lambda self: isatty})()
    assert _supports_color(stream) is expected
