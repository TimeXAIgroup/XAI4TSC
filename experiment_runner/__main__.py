"""
Module entry point, so the runner is executable by its package name.

::

    python -m experiment_runner --conf path/to/config.yaml

Delegates to :func:`experiment_runner.main.cli`, which owns the argument parsing
and the exit status. ``python -m experiment_runner.main`` runs the same code.
"""

from .main import cli

if __name__ == "__main__":
    cli()
