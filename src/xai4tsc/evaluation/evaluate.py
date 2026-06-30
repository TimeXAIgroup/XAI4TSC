"""``evaluate()`` entry point and the ``METRICS`` / ``QUANTUS_METRICS`` registries."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import quantus as q

from ..utils.utils import dict_to_args
from .base import EvaluatorBase, QuantusEvaluator
from .frequency_evaluate import FrequencyEvaluator
from .timefrequency_auc import TimeFrequencyAUCEvaluator
from .timefrequency_perturbation import (
    TimeFrequencyEvaluator,
    TimeFrequencyEvaluatorGaussian,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from torch import nn

    from ..xai.explain import Explanation

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    metric: str,
    explanation: Explanation,
    data: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    metric_class_params: dict | None = None,
    metric_call_params: dict | None = None,
    device: str = "cpu",
) -> float | np.ndarray | None:
    """
    Evaluate an explanation using the named metric from the METRICS registry.

    Parameters
    ----------
    model :
        A :class:`~torch.nn.Module` (typically a :class:`~xai4tsc.ModelBase`
        instance).
    metric : str
        Registry key for the metric to run.
    explanation : Explanation
        Explanation dataclass containing ``exp_values``.
    data : np.ndarray, optional
        Input samples.
    labels : np.ndarray, optional
        Ground-truth labels.
    metric_class_params : dict, optional
        Keyword arguments for the metric constructor.
    metric_call_params : dict, optional
        Keyword arguments for the metric ``__call__``.
    device : str
        Compute device.

    Returns
    -------
    float or np.ndarray or None
        Metric score, or ``None`` if the metric is not registered.
    """
    if metric not in METRICS:
        logger.warning("Metric %s is not registered, it will be skipped!", metric)
        return None

    model.eval()
    # Every METRICS value is a callable -> EvaluatorBase (a native subclass, or
    # partial(QuantusEvaluator, name) for Quantus metrics), so dispatch is uniform.
    # Filter construction params to the factory's signature so shared Quantus-only
    # params (e.g. `normalise`) don't break native evaluators; QuantusEvaluator
    # exposes **kwargs, so its params pass through and are filtered per-metric later.
    factory = METRICS[metric]
    init_params = dict_to_args(metric_class_params or {}, factory)
    evaluator = factory(**(init_params or {}))
    result = evaluator.evaluate(
        model=model,
        explanation=explanation,
        data=data,
        labels=labels,
        device=device,
        **(metric_call_params or {}),
    )
    logger.info("%s: %s", metric, result)
    return result


def register_metric(name: str, metric_class: type) -> None:
    """
    Register a custom metric class.

    Mirrors :func:`~xai4tsc.register_model` / :func:`~xai4tsc.register_explainer`:
    *metric_class* must be an :class:`~xai4tsc.evaluation.base.EvaluatorBase` subclass
    (instantiated with its constructor params at evaluation time). The fixed set of
    Quantus-library metrics is wired in package-side (via ``QUANTUS_METRICS`` and the
    single :class:`~xai4tsc.evaluation.base.QuantusEvaluator`), so users only ever
    register their own ``EvaluatorBase`` subclasses.

    Parameters
    ----------
    name:
        Key used to reference the metric in experiment configs.
    metric_class:
        The metric class to register.

    Raises
    ------
    TypeError
        If *metric_class* is not an :class:`~xai4tsc.evaluation.base.EvaluatorBase`
        subclass.
    """
    if not (isinstance(metric_class, type) and issubclass(metric_class, EvaluatorBase)):
        raise TypeError(
            f"Metric '{name}' must be an EvaluatorBase subclass, got {metric_class!r}."
        )
    METRICS[name] = metric_class


# Lookup table of external Quantus metric classes, keyed by display name. Mapped into
# the unified METRICS registry below via the single QuantusEvaluator adapter.
QUANTUS_METRICS = {
    "AUC": q.AUC,
    "Attribution Localisation": q.AttributionLocalisation,
    "Avg-Sensitivity": q.AvgSensitivity,
    "Completeness": q.Completeness,
    "Complexity": q.Complexity,
    "Consistency": q.Consistency,
    "Continuity Test": q.Continuity,
    "Effective Complexity": q.EffectiveComplexity,
    "Efficient MPRT": q.EfficientMPRT,
    "Faithfulness Correlation": q.FaithfulnessCorrelation,
    "Faithfulness Estimate": q.FaithfulnessEstimate,
    "Focus": q.Focus,
    "IROF": q.IROF,
    "Infidelity": q.Infidelity,
    "InputInvariance": q.InputInvariance,
    "Local Lipschitz Estimate": q.LocalLipschitzEstimate,
    "MPRT": q.MPRT,
    "Max-Sensitivity": q.MaxSensitivity,
    "Monotonicity-Arya": q.Monotonicity,
    "Monotonicity-Nguyen": q.MonotonicityCorrelation,
    "NonSensitivity": q.NonSensitivity,
    "Pixel-Flipping": q.PixelFlipping,
    "Pointing Game": q.PointingGame,
    "ROAD": q.ROAD,
    "Random Logit": q.RandomLogit,
    "Region Segmentation": q.RegionPerturbation,
    "Relative Input Stability": q.RelativeInputStability,
    "Relative Output Stability": q.RelativeOutputStability,
    "Relative Representation Stability": q.RelativeRepresentationStability,
    "Relevance Mass Accuracy": q.RelevanceMassAccuracy,
    "Relevance Rank Accuracy": q.RelevanceRankAccuracy,
    "Selectivity": q.Selectivity,
    "SensitivityN": q.SensitivityN,
    "Smooth MPRT": q.SmoothMPRT,
    "Sparseness": q.Sparseness,
    "Sufficiency": q.Sufficiency,
    "Top-K Intersection": q.TopKIntersection,
}


# Unified metric registry. Every value is a callable -> EvaluatorBase: native metrics
# are EvaluatorBase subclasses; each Quantus metric is mapped in via the single
# QuantusEvaluator, bound to its name with functools.partial.
METRICS: dict[str, type | Callable] = {
    # Frequency / time-frequency perturbation metrics (xai4tsc).
    "Frequency Perturbation": FrequencyEvaluator,
    "Time-Frequency Perturbation": TimeFrequencyEvaluator,
    "Time-Frequency Perturbation Gaussian": TimeFrequencyEvaluatorGaussian,
    # Ground-truth localization metric (needs per-sample GT metadata).
    "Time-Frequency AUC": TimeFrequencyAUCEvaluator,
    # Quantus-library metrics, adapted by the single QuantusEvaluator.
    **{name: partial(QuantusEvaluator, name) for name in QUANTUS_METRICS},
}
