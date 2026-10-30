"""
What each Quantus metric consumes, and the explanation domains that follow from it.

Quantus metrics do not all need the same inputs. A metric that reads only the
attribution measures a property of the attribution itself and is therefore
domain-agnostic; one that perturbs the input and re-queries the model needs the
explanation in the **model-native domain**, the domain the model's inputs live in
(``Explanation.data_domain``), because that is the space it can perturb and predict
in. For a model trained on raw series that is the time domain; for one trained on
spectra it is the frequency domain, and these metrics apply there instead. Quantus
itself draws no such distinction - every metric inherits one ``__call__`` that
asserts the attribution and the input correspond - so it is recorded here.

The tables here are keyed by the Quantus metric class. The four groups are disjoint
and cover every class in ``QUANTUS_METRICS``:

=======================  ===========================  ===  ===========================
group                    its own code reads           n    explanation domain
=======================  ===========================  ===  ===========================
``ATTRIBUTION_ONLY``     ``a_batch``                  3    any
``MODEL_PERTURBATION``   ``x_batch`` + ``model``      16   model-native
``LOCALISATION``         ``x_batch`` + ``s_batch``    6    model-native (see below)
``RE_EXPLAIN``           ``+ explain_func``           12   model-native
=======================  ===========================  ===  ===========================

Derived by walking each metric's own ``evaluate_batch`` / ``evaluate_instance`` /
``custom_*_preprocess`` bodies with :mod:`ast`, excluding the inherited
``Metric.__call__`` that every subclass re-declares with the full parameter list.
``sandbox/fixtures/group_metrics.py`` re-derives the table from the installed
``quantus`` so it can be re-checked after a dependency bump.

``LOCALISATION`` is model-native for the time being because it is blocked on
``s_batch`` (ground-truth masks), which nothing supplies yet - not because of the
domain. Once masks are derived, those six become domain-agnostic too:
:class:`~xai4tsc.evaluation.timefrequency_auc.TimeFrequencyAUCEvaluator` already
builds ground-truth masks on the coefficient grid.
"""

from __future__ import annotations

from enum import Enum

import quantus as q


class QuantusGroup(Enum):
    """Which inputs a Quantus metric's own code reads; see the module docstring."""

    ATTRIBUTION_ONLY = "attribution-only"
    """Reads the attribution alone, so it applies in any explanation domain."""

    MODEL_PERTURBATION = "model-perturbation"
    """Perturbs the input and re-queries the model."""

    LOCALISATION = "localisation"
    """Scores the attribution against a ground-truth mask (``s_batch``)."""

    RE_EXPLAIN = "re-explain"
    """Re-explains perturbed or randomised inputs via ``explain_func``."""


QUANTUS_GROUPS: dict[type[q.Metric], QuantusGroup] = {
    # Attribution-only: domain-agnostic.
    q.Complexity: QuantusGroup.ATTRIBUTION_ONLY,
    q.EffectiveComplexity: QuantusGroup.ATTRIBUTION_ONLY,
    q.Sparseness: QuantusGroup.ATTRIBUTION_ONLY,
    # Model perturbation.
    q.Completeness: QuantusGroup.MODEL_PERTURBATION,
    q.Consistency: QuantusGroup.MODEL_PERTURBATION,
    q.FaithfulnessCorrelation: QuantusGroup.MODEL_PERTURBATION,
    q.FaithfulnessEstimate: QuantusGroup.MODEL_PERTURBATION,
    q.Focus: QuantusGroup.MODEL_PERTURBATION,
    q.IROF: QuantusGroup.MODEL_PERTURBATION,
    q.Infidelity: QuantusGroup.MODEL_PERTURBATION,
    q.Monotonicity: QuantusGroup.MODEL_PERTURBATION,
    q.MonotonicityCorrelation: QuantusGroup.MODEL_PERTURBATION,
    q.NonSensitivity: QuantusGroup.MODEL_PERTURBATION,
    q.PixelFlipping: QuantusGroup.MODEL_PERTURBATION,
    q.ROAD: QuantusGroup.MODEL_PERTURBATION,
    q.RegionPerturbation: QuantusGroup.MODEL_PERTURBATION,
    q.Selectivity: QuantusGroup.MODEL_PERTURBATION,
    q.SensitivityN: QuantusGroup.MODEL_PERTURBATION,
    q.Sufficiency: QuantusGroup.MODEL_PERTURBATION,
    # Localisation: needs ground-truth masks.
    q.AUC: QuantusGroup.LOCALISATION,
    q.AttributionLocalisation: QuantusGroup.LOCALISATION,
    q.PointingGame: QuantusGroup.LOCALISATION,
    q.RelevanceMassAccuracy: QuantusGroup.LOCALISATION,
    q.RelevanceRankAccuracy: QuantusGroup.LOCALISATION,
    q.TopKIntersection: QuantusGroup.LOCALISATION,
    # Re-explain: needs an explain_func.
    q.AvgSensitivity: QuantusGroup.RE_EXPLAIN,
    q.Continuity: QuantusGroup.RE_EXPLAIN,
    q.EfficientMPRT: QuantusGroup.RE_EXPLAIN,
    q.InputInvariance: QuantusGroup.RE_EXPLAIN,
    q.LocalLipschitzEstimate: QuantusGroup.RE_EXPLAIN,
    q.MPRT: QuantusGroup.RE_EXPLAIN,
    q.MaxSensitivity: QuantusGroup.RE_EXPLAIN,
    q.RandomLogit: QuantusGroup.RE_EXPLAIN,
    q.RelativeInputStability: QuantusGroup.RE_EXPLAIN,
    q.RelativeOutputStability: QuantusGroup.RE_EXPLAIN,
    q.RelativeRepresentationStability: QuantusGroup.RE_EXPLAIN,
    q.SmoothMPRT: QuantusGroup.RE_EXPLAIN,
}
"""Group of every metric class in ``QUANTUS_METRICS``."""


_NATIVE_DOMAIN_GROUPS = frozenset(
    {
        QuantusGroup.MODEL_PERTURBATION,
        QuantusGroup.LOCALISATION,
        QuantusGroup.RE_EXPLAIN,
    }
)
"""Groups whose metrics need the explanation in the model-native domain."""


def quantus_requires_native_domain(quantus_class: type[q.Metric]) -> bool:
    """
    Return whether a Quantus metric class needs the model-native domain.

    Parameters
    ----------
    quantus_class : type[quantus.Metric]
        The Quantus metric class, e.g. ``quantus.PixelFlipping``.

    Returns
    -------
    bool
        ``True`` if the explanation must be in the same domain as the model's
        inputs, ``False`` if the metric applies in any domain. An unclassified
        class is treated as needing the native domain, the conservative choice.
    """
    group = QUANTUS_GROUPS.get(quantus_class)
    if group is None:
        return True
    return group in _NATIVE_DOMAIN_GROUPS


def quantus_requires_explain_func(quantus_class: type[q.Metric]) -> bool:
    """
    Return whether a Quantus metric class re-explains and so needs an explain func.

    Parameters
    ----------
    quantus_class : type[quantus.Metric]
        The Quantus metric class, e.g. ``quantus.PixelFlipping``.

    Returns
    -------
    bool
        ``True`` for the ``RE_EXPLAIN`` group only. An unclassified class is treated
        as not requiring one: supplying the callable to a metric that ignores it is
        harmless, whereas demanding one that is never used would refuse a metric
        that works.
    """
    return QUANTUS_GROUPS.get(quantus_class) is QuantusGroup.RE_EXPLAIN


def quantus_requires_s_batch(quantus_class: type[q.Metric]) -> bool:
    """
    Return whether a Quantus metric class scores against ground-truth masks.

    Parameters
    ----------
    quantus_class : type[quantus.Metric]
        The Quantus metric class, e.g. ``quantus.PixelFlipping``.

    Returns
    -------
    bool
        ``True`` for the ``LOCALISATION`` group only. These metrics compare the
        attribution against the region a dataset declares as discriminative, so
        they are unusable without one — unlike the other groups, which need
        nothing but the model and the attribution.
    """
    return QUANTUS_GROUPS.get(quantus_class) is QuantusGroup.LOCALISATION


BLOCKED_BY_BACKEND: dict[type[q.Metric], str] = {
    q.Continuity: (
        "it reshapes a (B, C, T) batch to (B, C, 1, T) and feeds that to the model, "
        "which a 1-D convolutional model cannot accept"
    ),
    q.Focus: (
        "it splits the attribution into four quadrants, which needs a second "
        "spatial axis that a (B, C, T) series does not have"
    ),
    q.IROF: (
        "it segments the input into super-pixels, which needs a 2-D image "
        "rather than a series"
    ),
    q.Infidelity: (
        "it pads over two spatial axes, so the pad width never matches the "
        "axes of a (B, C, T) series"
    ),
    q.ROAD: (
        "its imputation resolves pixel neighbours from a second spatial axis "
        "that a (B, C, T) series does not have"
    ),
    q.SmoothMPRT: (
        "its implementation calls the library's own explain function directly "
        "instead of the one it was given, so it would silently score a vanilla "
        "gradient rather than the explainer under evaluation"
    ),
    q.InputInvariance: (
        "it reads 'out_channels' off the model's second named module, which is a "
        "container rather than a conv layer in every model shipped here"
    ),
}
"""Metrics whose Quantus implementation cannot serve a time series correctly.

Four of them index a second spatial axis that a ``(B, C, T)`` series does not
have - super-pixels, quadrants, 2-D padding, pixel neighbourhoods - and raise
somewhere inside the library. Refusing them here turns an obscure ``IndexError``
or shape assertion into a stated reason, and keeps them out of the error column
where they look like something worth fixing.

``SmoothMPRT`` is the dangerous one and the reason this table is not just about
shapes: it *returns a number*. It calls the library's own explain function
instead of the one it is given, so without this entry it would report a
plausible score for an explainer it never ran.
"""
