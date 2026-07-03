#!/usr/bin/env python3
"""
Generate ``requirements.txt`` from ``[project.dependencies]`` in pyproject.toml.

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

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
REQUIREMENTS_DOCS = REPO_ROOT / "docs" / "requirements.txt"

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
DOCS_HEADER = """\
# Documentation build dependencies (used by .github/workflows/docs.yml).
#
# GENERATED FILE — do not edit by hand.
# Regenerate after changing the docs dependencies in pyproject.toml:
#
#     python scripts/generate_requirements.py"""


_CARET_RE = re.compile(r"\^(\d+(?:\.\d+){0,2})")


def _caret_to_range(version: str) -> str:
    """
    Expand a Poetry caret version into an equivalent PEP 508 range.

    The version is the part after ``^`` (e.g. ``1.2.3``), expanded into a
    ``>=lower,<upper`` range.

    Caret semantics: allow changes that do not modify the leftmost
    non-zero component of the version.
        ^1.2.3 -> >=1.2.3,<2.0.0
        ^1.2   -> >=1.2.0,<2.0.0
        ^0.2.3 -> >=0.2.3,<0.3.0
        ^0.0.3 -> >=0.0.3,<0.0.4
        ^0.0   -> >=0.0.0,<0.1.0
        ^0     -> >=0.0.0,<1.0.0
    """
    parts = [int(p) for p in version.split(".")]

    lower_parts = parts + [0] * (3 - len(parts))
    lower = ".".join(str(p) for p in lower_parts)

    upper_parts = list(parts)
    for i, p in enumerate(upper_parts):
        if p != 0:
            upper_parts[i] += 1
            upper_parts = upper_parts[: i + 1]
            break
    else:
        # every given component is zero — bump the last one
        upper_parts[-1] += 1
    upper_parts += [0] * (3 - len(upper_parts))
    upper = ".".join(str(p) for p in upper_parts)

    return f">={lower},<{upper}"


def _normalise(specifier: str) -> str:
    """
    Normalise a PEP 508 specifier to bare requirements.txt form.

    Poetry writes dependencies as ``name (>=x,<y)``; requirements.txt uses
    ``name>=x,<y``. Both are valid PEP 508 — strip the cosmetic spaces and
    parentheses so the two representations are identical.

    Poetry also supports the caret operator (``^x.y.z``), which has no
    direct PEP 508 equivalent, so it is expanded into an explicit
    ``>=x.y.z,<X.0.0`` range.

    Parameters
    ----------
    specifier : str
        A dependency string from ``[project.dependencies]``.

    Returns
    -------
    str
        The specifier with spaces and parentheses removed, and any caret
        constraints expanded into explicit ranges.
    """
    cleaned = specifier.replace(" ", "").replace("(", "").replace(")", "").strip()
    return _CARET_RE.sub(lambda m: _caret_to_range(m.group(1)), cleaned)


def generate_reqs() -> str:
    """
    Render the ``requirements.txt`` contents from ``[project.dependencies]``.

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


def generate_docs_reqs() -> str:
    """
    Render the ``requirements.txt`` contents from ``[project.dependencies]``.

    Returns
    -------
    str
        The full file contents, including the generated-file header.
    """
    with PYPROJECT.open("rb") as fh:
        dependencies = tomllib.load(fh)["tool"]["poetry"]["group"]["docs"][
            "dependencies"
        ]

    lines = [DOCS_HEADER.rstrip("\n"), ""]
    dependencies = [k + v for k, v in dependencies.items()]
    lines.extend(_normalise(dep) for dep in dependencies)
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the generated requirements to ``requirements.txt``."""
    REQUIREMENTS.write_text(generate_reqs(), encoding="utf-8")
    print(f"Wrote {REQUIREMENTS.relative_to(REPO_ROOT)}")
    REQUIREMENTS_DOCS.write_text(generate_docs_reqs(), encoding="utf-8")
    print(f"Wrote {REQUIREMENTS_DOCS.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
