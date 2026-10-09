"""``evaluate()`` entry point and the ``METRICS`` / ``QUANTUS_METRICS`` registries."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import quantus as q

from ..utils.utils import dict_to_args
from ..xai._types import Domain
from .base import EvaluatorBase, QuantusEvaluator
from .frequency_evaluate import FrequencyEvaluator
from .quantus_groups import BLOCKED_BY_BACKEND, quantus_requires_native_domain
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
    seed: int | None = None,
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
    seed : int or None
        Seed for a metric that randomises something of its own, such as the MPRT
        family's weight randomisation. Forwarded to the evaluator, which offers it
        only to a backend that declares a ``seed``; the rest ignore it. A seed in
        *metric_class_params* takes precedence.

    Returns
    -------
    float or np.ndarray or None
        Metric score, or ``None`` if the metric is not registered.
    """
    if metric not in METRICS:
        logger.warning("Metric %s is not registered, it will be skipped!", metric)
        return None

    # A metric that declares required_domains cannot score an explanation from
    # another domain; refuse here rather than letting the backend raise.
    required = required_domains_for(metric, explanation.data_domain)
    if required and explanation.explanation_domain not in required:
        logger.warning(
            "Metric %s requires an explanation in domain(s) %s but the explanation "
            "is %s, it will be skipped!",
            metric,
            sorted(d.value for d in required),
            explanation.explanation_domain.value,
        )
        return None

    # Localisation metrics require datasets with ground truth.
    missing = ground_truth_missing(metric, explanation)
    if missing:
        logger.warning(
            "Metric %s cannot score this explanation: %s. It will be skipped!",
            metric,
            missing,
        )
        return None

    # A handful of Quantus metrics assume 2-D image input deep in their own code,
    # so no setting on this side makes them work on a time series.
    if metric in BLOCKED_BY_BACKEND:
        logger.warning(
            "Metric %s cannot score a time series: %s. It will be skipped!",
            metric,
            BLOCKED_BY_BACKEND[metric],
        )
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
    # The seed goes in below the user's call params, so a configured one wins and
    # no duplicate keyword can be raised. Evaluators that cannot use it ignore it.
    call_params = {"seed": seed, **(metric_call_params or {})}
    result = evaluator.evaluate(
        model=model,
        explanation=explanation,
        data=data,
        labels=labels,
        device=device,
        **call_params,
    )
    logger.info("%s: %s", metric, result)
    return result


def ground_truth_missing(metric: str, explanation: Explanation) -> str | None:
    """
    Return why *metric* cannot score *explanation*, or ``None`` if it can.

    The localisation metrics compare an attribution against the regions a dataset
    declares as discriminative. Most datasets declare none — the discriminative
    region of a real recording is not known — so this is an ordinary unmet
    requirement rather than a failure, and the caller is expected to skip and say
    so.

    Parameters
    ----------
    metric : str
        A key of the ``METRICS`` registry.
    explanation : Explanation
        The explanation to score; its ``metadata`` carries the regions.

    Returns
    -------
    str or None
        A reason the metric cannot be scored, or ``None`` when it can — including
        for every metric that does not need ground truth at all.
    """
    from ._ground_truth import regions_for_sample
    from .quantus_groups import quantus_requires_s_batch

    if not quantus_requires_s_batch(metric):
        return None
    if not explanation.metadata:
        return (
            "it scores the attribution against the dataset's ground-truth "
            "regions, and this dataset declares none"
        )
    # A ValueError from here is deliberately left to propagate: malformed metadata
    # is a defect, not an unmet requirement, and reporting it as a skip would hide it.
    if any(regions_for_sample(sample) for sample in explanation.metadata):
        return None
    return (
        "the dataset declares ground-truth regions, but none of the explained "
        "samples has one"
    )


def required_domains_for(metric: str, data_domain: Domain = Domain.TIME) -> set[Domain]:
    """
    Return the explanation domains a registered metric can score.

    Reads the declaration off the registry *value*, so it covers native evaluator
    classes, user-registered subclasses, and the Quantus entries alike (those carry
    their per-name declaration on the factory; see :func:`_quantus_entry`). A metric
    that requires the model-native domain resolves to ``{data_domain}``.

    Parameters
    ----------
    metric : str
        Registry key for the metric.
    data_domain : Domain
        The domain the model's inputs live in, i.e.
        :attr:`xai4tsc.Explanation.data_domain`. The default reflects that every
        shipped loader produces time-domain data; pass the explanation's own value
        where it is known.

    Returns
    -------
    set of Domain
        The domains the metric applies to; an empty set means any domain, which is
        also what an unregistered name returns.
    """
    entry = METRICS.get(metric)
    if getattr(entry, "requires_native_domain", False):
        return {data_domain}
    return getattr(entry, "required_domains", None) or set()


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


def _quantus_entry(name: str) -> partial:
    """
    Build the ``METRICS`` entry for one Quantus metric.

    A bare ``functools.partial`` hides the wrapped class's declarations, and
    :func:`required_domains_for` reads them off the registry value to decide whether a
    metric can score an explanation. They are therefore attached to the factory, per
    metric name (see :mod:`~xai4tsc.evaluation.quantus_groups`).

    Parameters
    ----------
    name : str
        A key of ``QUANTUS_METRICS``.

    Returns
    -------
    functools.partial
        A callable producing a :class:`~xai4tsc.evaluation.base.QuantusEvaluator`.
    """
    entry = partial(QuantusEvaluator, name)
    entry.requires_native_domain = quantus_requires_native_domain(name)
    entry.data_applicability = QuantusEvaluator.data_applicability
    return entry


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
    **{name: _quantus_entry(name) for name in QUANTUS_METRICS},
}
