"""``evaluate()`` entry point and the ``METRICS`` / ``QUANTUS_METRICS`` registries."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import quantus as q

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
    evaluator: EvaluatorBase,
    model: nn.Module,
    explanation: Explanation,
    data: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    call_params: dict | None = None,
    device: str = "cpu",
    seed: int | None = None,
) -> float | np.ndarray | None:
    """
    Score an explanation with an evaluator.

    Parameters
    ----------
    evaluator : EvaluatorBase
        The configured metric to run, e.g. ``FrequencyEvaluator()`` or
        ``QuantusEvaluator(quantus.Complexity, disable_warnings=True)``.
    model :
        A :class:`~torch.nn.Module` (typically a :class:`~xai4tsc.ModelBase`
        instance).
    explanation : Explanation
        Explanation dataclass containing ``exp_values``.
    data : np.ndarray, optional
        Input samples.
    labels : np.ndarray, optional
        Ground-truth labels.
    call_params : dict, optional
        Keyword arguments for the evaluator's ``evaluate()``; for a
        :class:`~xai4tsc.evaluation.base.QuantusEvaluator` they reach the Quantus
        metric's ``__call__``.
    device : str
        Compute device.
    seed : int or None
        Seed for a metric that randomises something of its own, such as the MPRT
        family's weight randomisation. Forwarded to the evaluator, which offers it
        only to a backend that declares a ``seed``; the rest ignore it. A seed in
        the evaluator's constructor params or in *call_params* takes precedence.

    Returns
    -------
    float or np.ndarray or None
        Metric score, or ``None`` if the evaluator cannot score this explanation
        (wrong domain, no ground truth, or a backend that cannot handle a series).

    Raises
    ------
    TypeError
        If *evaluator* is not an :class:`~xai4tsc.evaluation.base.EvaluatorBase`
        instance, e.g. a metric's display name.
    """
    if not isinstance(evaluator, EvaluatorBase):
        raise TypeError(
            "evaluate() takes an EvaluatorBase instance, e.g. FrequencyEvaluator() "
            f"or QuantusEvaluator(quantus.Complexity), got {evaluator!r}."
        )
    metric = evaluator.metric_name or type(evaluator).__name__

    # A metric that declares required_domains cannot score an explanation from
    # another domain; refuse here rather than letting the backend raise.
    required = required_domains_for(evaluator, explanation.data_domain)
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
    missing = ground_truth_missing(evaluator, explanation)
    if missing:
        logger.warning(
            "Metric %s cannot score this explanation: %s. It will be skipped!",
            metric,
            missing,
        )
        return None

    # A handful of Quantus metrics assume 2-D image input deep in their own code,
    # so no setting on this side makes them work on a time series.
    blocked = blocked_by_backend(evaluator)
    if blocked:
        logger.warning(
            "Metric %s cannot score a time series: %s. It will be skipped!",
            metric,
            blocked,
        )
        return None

    model.eval()
    # The seed goes in below the user's call params, so a configured one wins and
    # no duplicate keyword can be raised. Evaluators that cannot use it ignore it.
    result = evaluator.evaluate(
        model=model,
        explanation=explanation,
        data=data,
        labels=labels,
        device=device,
        **{"seed": seed, **(call_params or {})},
    )
    logger.info("%s: %s", metric, result)
    return result


def blocked_by_backend(evaluator: EvaluatorBase) -> str | None:
    """
    Return why *evaluator*'s backend cannot score a time series, or ``None``.

    Parameters
    ----------
    evaluator : EvaluatorBase
        The evaluator to check. Only a
        :class:`~xai4tsc.evaluation.base.QuantusEvaluator` can be blocked, by an
        entry for its Quantus class in
        :data:`~xai4tsc.evaluation.quantus_groups.BLOCKED_BY_BACKEND`.

    Returns
    -------
    str or None
        The reason, or ``None`` when the evaluator can run.
    """
    if not isinstance(evaluator, QuantusEvaluator):
        return None
    return BLOCKED_BY_BACKEND.get(evaluator.quantus_class)


def ground_truth_missing(
    evaluator: EvaluatorBase, explanation: Explanation
) -> str | None:
    """
    Return why *evaluator* cannot score *explanation*, or ``None`` if it can.

    The localisation metrics compare an attribution against the regions a dataset
    declares as discriminative. Most datasets declare none — the discriminative
    region of a real recording is not known — so this is an ordinary unmet
    requirement rather than a failure, and the caller is expected to skip and say
    so.

    Parameters
    ----------
    evaluator : EvaluatorBase
        The evaluator to check; only a
        :class:`~xai4tsc.evaluation.base.QuantusEvaluator` for a localisation
        metric needs ground truth here.
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

    if not (
        isinstance(evaluator, QuantusEvaluator)
        and quantus_requires_s_batch(evaluator.quantus_class)
    ):
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


def required_domains_for(
    evaluator: EvaluatorBase | Callable[..., EvaluatorBase] | None,
    data_domain: Domain = Domain.TIME,
) -> set[Domain]:
    """
    Return the explanation domains an evaluator can score.

    Reads the declarations off *evaluator*, so it covers native evaluators,
    user subclasses and :class:`~xai4tsc.evaluation.base.QuantusEvaluator`
    alike. It also accepts a ``METRICS`` value, which carries the same
    declarations (the Quantus entries on the factory, see :func:`_quantus_entry`),
    so a caller can check a metric before building it. A metric that requires the
    model-native domain resolves to ``{data_domain}``.

    Parameters
    ----------
    evaluator : EvaluatorBase, METRICS value or None
        The evaluator, or the class or factory that builds it.
    data_domain : Domain
        The domain the model's inputs live in, i.e.
        :attr:`xai4tsc.Explanation.data_domain`. The default reflects that every
        shipped loader produces time-domain data; pass the explanation's own value
        where it is known.

    Returns
    -------
    set of Domain
        The domains the metric applies to; an empty set means any domain, which is
        also what ``None`` returns.
    """
    if getattr(evaluator, "requires_native_domain", False):
        return {data_domain}
    return getattr(evaluator, "required_domains", None) or set()


# Lookup table of external Quantus metric classes, keyed by display name. Mapped into
# the unified METRICS registry below via the single QuantusEvaluator adapter.
QUANTUS_METRICS: dict[str, type[q.Metric]] = {
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


def _quantus_entry(quantus_class: type[q.Metric]) -> partial:
    """
    Build the ``METRICS`` entry for one Quantus metric.

    A bare ``functools.partial`` hides the wrapped class's declarations, and
    :func:`required_domains_for` reads them off the registry value to decide whether a
    metric can score an explanation. They are therefore attached to the factory, per
    metric class (see :mod:`~xai4tsc.evaluation.quantus_groups`).

    Parameters
    ----------
    quantus_class : type[quantus.Metric]
        A value of ``QUANTUS_METRICS``.

    Returns
    -------
    functools.partial
        A callable producing a :class:`~xai4tsc.evaluation.base.QuantusEvaluator`.
    """
    entry = partial(QuantusEvaluator, quantus_class)
    entry.requires_native_domain = quantus_requires_native_domain(quantus_class)
    entry.data_applicability = QuantusEvaluator.data_applicability
    return entry


# Unified metric registry. Every value is a callable -> EvaluatorBase: native metrics
# are EvaluatorBase subclasses; each Quantus metric is mapped in via the single
# QuantusEvaluator, bound to its class with functools.partial.
METRICS: dict[str, type | Callable] = {
    # Frequency / time-frequency perturbation metrics (xai4tsc).
    "Frequency Perturbation": FrequencyEvaluator,
    "Time-Frequency Perturbation": TimeFrequencyEvaluator,
    "Time-Frequency Perturbation Gaussian": TimeFrequencyEvaluatorGaussian,
    # Ground-truth localization metric (needs per-sample GT metadata).
    "Time-Frequency AUC": TimeFrequencyAUCEvaluator,
    # Quantus-library metrics, adapted by the single QuantusEvaluator.
    **{name: _quantus_entry(cls) for name, cls in QUANTUS_METRICS.items()},
}
