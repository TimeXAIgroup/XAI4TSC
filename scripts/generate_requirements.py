#!/usr/bin/env python3
"""Generate ``requirements.txt`` from ``[project.dependencies]`` in pyproject.toml.

``requirements.txt`` is a convenience for users who install with plain pip
(``pip install -r requirements.txt``). It mirrors the **direct** runtime
dependencies declared in ``pyproject.toml`` as version *ranges* — it is not a
pinned lockfile export (use ``poetry install`` / ``poetry.lock`` for an exact,
reproducible environment).

``pyproject.toml`` is the single source of truth. Rather than hand-editing both
files, regenerate ``requirements.txt`` from pyproject whenever the dependencies
change::

    python scripts/generate_requirements.py

The ``Tests`` workflow's ``deps`` job verifies the two stay in sync, so a stale
``requirements.txt`` fails CI.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

HEADER = """\
# Runtime dependencies for xai4tsc.
#
# GENERATED FILE — do not edit by hand.
# Regenerate after changing dependencies in pyproject.toml:
#     python scripts/generate_requirements.py
#
# This mirrors the direct runtime dependencies in [project.dependencies] as
# version ranges, for users who prefer plain pip (pip install -r requirements.txt).
# It is NOT a pinned lockfile export — for an exact, fully reproducible install
# use Poetry (poetry install) with poetry.lock.
"""


def _normalise(specifier: str) -> str:
    """Normalise a PEP 508 specifier to bare requirements.txt form.

    Poetry writes dependencies as ``name (>=x,<y)``; requirements.txt uses
    ``name>=x,<y``. Both are valid PEP 508 — strip the cosmetic spaces and
    parentheses so the two representations are identical.

    Parameters
    ----------
    specifier : str
        A dependency string from ``[project.dependencies]``.

    Returns
    -------
    str
        The specifier with spaces and parentheses removed.
    """
    return specifier.replace(" ", "").replace("(", "").replace(")", "").strip()


def generate() -> str:
    """Render the ``requirements.txt`` contents from ``[project.dependencies]``.

    Returns
    -------
    str
        The full file contents, including the generated-file header.
    """
    with PYPROJECT.open("rb") as fh:
        dependencies = tomllib.load(fh)["project"]["dependencies"]

    lines = [HEADER.rstrip("\n"), ""]
    lines.extend(_normalise(dep) for dep in dependencies)
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the generated requirements to ``requirements.txt``."""
    REQUIREMENTS.write_text(generate(), encoding="utf-8")
    print(f"Wrote {REQUIREMENTS.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
