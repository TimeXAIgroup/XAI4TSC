"""
Explainer base classes, built-in Captum methods, and the ``EXPLAINERS`` registry.

Submodules:

- :mod:`~xai4tsc.xai.explain` — the :func:`generate_explanation` entry point and
  the ``EXPLAINERS`` registry.
- :mod:`~xai4tsc.xai.feature_attribution` — Captum gradient / perturbation
  methods and TSHAP.
- :mod:`~xai4tsc.xai.freqrise`, :mod:`~xai4tsc.xai.explanation_domains`,
  :mod:`~xai4tsc.xai.random_baseline` — frequency and time-frequency explainers.
- :mod:`~xai4tsc.xai.wrappers` — the SIGN wrapper.
- :mod:`~xai4tsc.xai._types` — the :class:`Explanation` dataclass and the
  :class:`Domain` / :class:`DataType` enums.
- :mod:`~xai4tsc.xai.base` — the explainer base classes.

Use :func:`generate_explanation` as the main entry point. Subclass
:class:`~xai4tsc.xai.base.ExplainerBase` (or one of its mid-level
specialisations) to add a custom method.
"""

from ._types import DataType, Domain, Explanation
from .base import (
    ExplainerBase,
    GradientExplainer,
    PerturbationExplainer,
    SurrogateExplainer,
    WrapperExplainer,
)
from .explain import (
    EXPLAINERS,
    build_explainer,
    generate_explanation,
    register_explainer,
)
from .wrappers import SignExplainer

__all__ = [
    "EXPLAINERS",
    "DataType",
    "Domain",
    "ExplainerBase",
    "Explanation",
    "GradientExplainer",
    "PerturbationExplainer",
    "SignExplainer",
    "SurrogateExplainer",
    "WrapperExplainer",
    "build_explainer",
    "generate_explanation",
    "register_explainer",
]
