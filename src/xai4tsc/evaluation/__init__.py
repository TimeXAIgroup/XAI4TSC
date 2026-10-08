"""
XAI evaluation metrics wrapping the Quantus library.

Submodules:

- :mod:`~xai4tsc.evaluation.evaluate` — the :func:`evaluate` entry point and the
  ``METRICS`` registry.
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

Use :func:`evaluate` as the main entry point. The ``METRICS`` registry maps metric names
to callables that produce an :class:`EvaluatorBase` (native subclasses, or the single
:class:`QuantusEvaluator` bound to a name from ``QUANTUS_METRICS``); use
:func:`register_metric` to add custom ones.
"""

from .base import EvaluatorBase, QuantusEvaluator
from .evaluate import (
    METRICS,
    QUANTUS_METRICS,
    evaluate,
    ground_truth_missing,
    register_metric,
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
    "evaluate",
    "ground_truth_missing",
    "register_metric",
    "required_domains_for",
]
