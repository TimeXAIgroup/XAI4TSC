"""
XAI evaluation metrics wrapping the Quantus library.

Submodules:

- :mod:`~xai4tsc.evaluation.evaluate` — the :func:`evaluate` entry point and the
  ``METRICS`` catalog.
- :mod:`~xai4tsc.evaluation.base` — the :class:`EvaluatorBase` /
  :class:`QuantusEvaluator` wrappers.
- :mod:`~xai4tsc.evaluation.quantus_groups` — what each Quantus metric consumes
  (``QUANTUS_GROUPS``) and the explanation domains that follow from it.
- :mod:`~xai4tsc.evaluation._ground_truth` — the ground-truth region schema
  (:class:`GroundTruthRegion`) and the masks the localization metrics score
  against. Datasets store plain mappings; this is where they are parsed.
- :mod:`~xai4tsc.evaluation.frequency_evaluate`,
  :mod:`~xai4tsc.evaluation.timefrequency_perturbation`,
  :mod:`~xai4tsc.evaluation.timefrequency_auc` — native frequency and
  time-frequency metrics.

Use :func:`evaluate` as the main entry point; it takes an :class:`EvaluatorBase`
instance, such as a native evaluator or ``QuantusEvaluator(quantus.Complexity)``. The
``METRICS`` catalog maps metric names to callables that produce one (native
subclasses, or the single :class:`QuantusEvaluator` bound to a class from
``QUANTUS_METRICS``); it is for discovery, no package function reads it.
"""

from .base import EvaluatorBase, QuantusEvaluator
from .evaluate import (
    METRICS,
    QUANTUS_METRICS,
    blocked_by_backend,
    evaluate,
    ground_truth_missing,
    required_domains_for,
)
from .quantus_groups import QUANTUS_GROUPS, QuantusGroup

__all__ = [
    "METRICS",
    "QUANTUS_GROUPS",
    "QUANTUS_METRICS",
    "EvaluatorBase",
    "QuantusEvaluator",
    "QuantusGroup",
    "blocked_by_backend",
    "evaluate",
    "ground_truth_missing",
    "required_domains_for",
]
