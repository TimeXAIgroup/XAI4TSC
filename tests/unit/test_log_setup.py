"""
Unit tests for experiment_runner/log_setup.py.

Covers the runner's file sinks (attached per run and per model, detached
cleanly) and the console colouring.
"""

import logging

import pytest
from experiment_runner.log_setup import (
    _LEVEL_COLORS,
    _RESET,
    _ColorFormatter,
    _make_formatter,
    _setup_console_logging,
    _supports_color,
    add_file_log,
    remove_file_log,
)

import xai4tsc


@pytest.fixture(autouse=True)
def _restore_logging():
    """Reset xai4tsc logging state before and after each test."""
    xai4tsc.disable_logging()
    yield
    xai4tsc.disable_logging()


@pytest.mark.unit
def test_file_sink_writes_and_detaches(tmp_path):
    logger = xai4tsc.enable_logging(level="DEBUG")
    model_log = tmp_path / "GunPoint" / "FCN" / "GunPoint_FCN.log"

    handler = add_file_log(model_log)
    assert handler.level == logging.DEBUG  # inherited from the logger
    logging.getLogger("xai4tsc.runner").info("inside the model")
    remove_file_log(handler)
    logging.getLogger("xai4tsc.runner").info("after the model")

    assert handler not in logger.handlers
    text = model_log.read_text()
    assert "inside the model" in text
    assert "after the model" not in text
    remove_file_log(None)  # no-op, so a finally block can call it unconditionally


@pytest.mark.unit
def test_console_is_coloured_and_file_sink_is_not(tmp_path, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    _setup_console_logging(debug=False)
    handler = add_file_log(tmp_path / "run.log")
    try:
        formatters = {
            type(h): type(h.formatter)
            for h in logging.getLogger("xai4tsc").handlers
            if not isinstance(h, logging.NullHandler)
        }
        assert formatters[logging.StreamHandler] is _ColorFormatter
        assert formatters[logging.FileHandler] is logging.Formatter
    finally:
        remove_file_log(handler)


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
