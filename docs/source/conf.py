# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Add project root to path
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "XAI4TSC"
copyright = "2026, TimeXAI Research Group"
author = "TimeXAI Research Group"

# Pull the release version from the installed package so docs never diverge
# from pyproject.toml. Falls back gracefully if the package isn't installed.
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("xai4tsc")
except Exception:  # noqa: BLE001 — docs build should not fail over version lookup
    release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
    # "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "autoapi.extension",
    "myst_parser",  # render Markdown (.md) sources alongside reStructuredText
]
# Guido on autoapi
# https://bylr.info/articles/2022/05/10/api-doc-with-sphinx-autoapi/#improving-the-default-templates

# autodoc_default_options = {
#     "imported-members": False,
# }

autoapi_dirs = ["../../src/xai4tsc", "../../experiment_runner"]  # path to your package
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # "imported-members",
]
autoapi_keep_files = False

# Choose docstring format
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_favicon = "_static/xai4tsc_html_icon.png"
html_static_path = ["_static"]
