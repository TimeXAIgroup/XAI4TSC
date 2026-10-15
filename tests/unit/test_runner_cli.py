"""Unit tests for the runner's command-line entry point - no training, no I/O."""

import inspect
import sys

import experiment_runner.main as runner_main
import pytest
from experiment_runner.main import cli


@pytest.mark.unit
def test_cli_takes_no_arguments():
    """`__main__.py` has nothing to pass, so `cli` must need nothing."""
    required = [
        name
        for name, param in inspect.signature(cli).parameters.items()
        if param.default is inspect.Parameter.empty
    ]
    assert not required


@pytest.mark.unit
def test_main_module_delegates_to_cli():
    """`python -m experiment_runner` must reach the same callable."""
    from experiment_runner.__main__ import cli as delegated

    assert delegated is cli


@pytest.mark.unit
@pytest.mark.parametrize(("failures", "status"), [(0, 0), (1, 1), (7, 1)])
def test_cli_exit_status_reports_whether_anything_failed(monkeypatch, failures, status):
    """The failure count collapses to a shell exit status, not a literal count."""
    monkeypatch.setattr(runner_main, "main", lambda *args, **kwargs: failures)
    monkeypatch.setattr(sys, "argv", ["python -m experiment_runner"])

    with pytest.raises(SystemExit) as exit_info:
        cli()

    assert exit_info.value.code == status


@pytest.mark.unit
def test_cli_forwards_the_config_path_and_debug_flag(monkeypatch):
    seen = {}

    def _record(config_path, debug=False):
        seen["config_path"] = config_path
        seen["debug"] = debug
        return 0

    monkeypatch.setattr(runner_main, "main", _record)
    monkeypatch.setattr(
        sys, "argv", ["python -m experiment_runner", "--conf", "my.yaml", "--debug"]
    )

    with pytest.raises(SystemExit):
        cli()

    assert seen == {"config_path": "my.yaml", "debug": True}


@pytest.mark.unit
def test_cli_defaults_to_the_shipped_example_config(monkeypatch):
    """Running with no arguments must pick up example.yaml, which ships."""
    seen = {}

    def _record(config_path, debug=False):
        seen["config_path"] = config_path
        return 0

    monkeypatch.setattr(runner_main, "main", _record)
    monkeypatch.setattr(sys, "argv", ["python -m experiment_runner"])

    with pytest.raises(SystemExit):
        cli()

    config_path = runner_main.Path(seen["config_path"])
    assert config_path.name == "example.yaml"
    assert config_path.is_file()


@pytest.mark.unit
def test_help_names_an_invocation_that_exists(capsys, monkeypatch):
    """
    The usage line must be typeable.

    It read `xai4tsc` before R21, which is not a console script this project
    installs, so `--help` taught a command that could only fail.
    """
    monkeypatch.setattr(sys, "argv", ["whatever", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        cli()

    assert exit_info.value.code == 0
    assert "usage: python -m experiment_runner" in capsys.readouterr().out
