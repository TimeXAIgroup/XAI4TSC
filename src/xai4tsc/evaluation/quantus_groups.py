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

The four groups are disjoint and cover every key of ``QUANTUS_METRICS``:

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


QUANTUS_GROUPS: dict[str, QuantusGroup] = {
    # Attribution-only: domain-agnostic.
    "Complexity": QuantusGroup.ATTRIBUTION_ONLY,
    "Effective Complexity": QuantusGroup.ATTRIBUTION_ONLY,
    "Sparseness": QuantusGroup.ATTRIBUTION_ONLY,
    # Model perturbation.
    "Completeness": QuantusGroup.MODEL_PERTURBATION,
    "Consistency": QuantusGroup.MODEL_PERTURBATION,
    "Faithfulness Correlation": QuantusGroup.MODEL_PERTURBATION,
    "Faithfulness Estimate": QuantusGroup.MODEL_PERTURBATION,
    "Focus": QuantusGroup.MODEL_PERTURBATION,
    "IROF": QuantusGroup.MODEL_PERTURBATION,
    "Infidelity": QuantusGroup.MODEL_PERTURBATION,
    "Monotonicity-Arya": QuantusGroup.MODEL_PERTURBATION,
    "Monotonicity-Nguyen": QuantusGroup.MODEL_PERTURBATION,
    "NonSensitivity": QuantusGroup.MODEL_PERTURBATION,
    "Pixel-Flipping": QuantusGroup.MODEL_PERTURBATION,
    "ROAD": QuantusGroup.MODEL_PERTURBATION,
    "Region Segmentation": QuantusGroup.MODEL_PERTURBATION,
    "Selectivity": QuantusGroup.MODEL_PERTURBATION,
    "SensitivityN": QuantusGroup.MODEL_PERTURBATION,
    "Sufficiency": QuantusGroup.MODEL_PERTURBATION,
    # Localisation: needs ground-truth masks.
    "AUC": QuantusGroup.LOCALISATION,
    "Attribution Localisation": QuantusGroup.LOCALISATION,
    "Pointing Game": QuantusGroup.LOCALISATION,
    "Relevance Mass Accuracy": QuantusGroup.LOCALISATION,
    "Relevance Rank Accuracy": QuantusGroup.LOCALISATION,
    "Top-K Intersection": QuantusGroup.LOCALISATION,
    # Re-explain: needs an explain_func.
    "Avg-Sensitivity": QuantusGroup.RE_EXPLAIN,
    "Continuity Test": QuantusGroup.RE_EXPLAIN,
    "Efficient MPRT": QuantusGroup.RE_EXPLAIN,
    "InputInvariance": QuantusGroup.RE_EXPLAIN,
    "Local Lipschitz Estimate": QuantusGroup.RE_EXPLAIN,
    "MPRT": QuantusGroup.RE_EXPLAIN,
    "Max-Sensitivity": QuantusGroup.RE_EXPLAIN,
    "Random Logit": QuantusGroup.RE_EXPLAIN,
    "Relative Input Stability": QuantusGroup.RE_EXPLAIN,
    "Relative Output Stability": QuantusGroup.RE_EXPLAIN,
    "Relative Representation Stability": QuantusGroup.RE_EXPLAIN,
    "Smooth MPRT": QuantusGroup.RE_EXPLAIN,
}
"""Group of every metric in ``QUANTUS_METRICS``, keyed by the same display name."""


_NATIVE_DOMAIN_GROUPS = frozenset(
    {
        QuantusGroup.MODEL_PERTURBATION,
        QuantusGroup.LOCALISATION,
        QuantusGroup.RE_EXPLAIN,
    }
)
"""Groups whose metrics need the explanation in the model-native domain."""


def quantus_requires_native_domain(name: str) -> bool:
    """
    Return whether the named Quantus metric needs the model-native domain.

    Parameters
    ----------
    name : str
        A key of ``QUANTUS_METRICS`` (the metric's display name).

    Returns
    -------
    bool
        ``True`` if the explanation must be in the same domain as the model's
        inputs, ``False`` if the metric applies in any domain. An unclassified
        name is treated as needing the native domain, the conservative choice.
    """
    group = QUANTUS_GROUPS.get(name)
    if group is None:
        return True
    return group in _NATIVE_DOMAIN_GROUPS


def quantus_requires_explain_func(name: str) -> bool:
    """
    Return whether the named Quantus metric re-explains and so needs an explain func.

    Parameters
    ----------
    name : str
        A key of ``QUANTUS_METRICS`` (the metric's display name).

    Returns
    -------
    bool
        ``True`` for the ``RE_EXPLAIN`` group only. An unclassified name is treated
        as not requiring one: supplying the callable to a metric that ignores it is
        harmless, whereas demanding one that is never used would refuse a metric
        that works.
    """
    return QUANTUS_GROUPS.get(name) is QuantusGroup.RE_EXPLAIN


BLOCKED_BY_BACKEND: dict[str, str] = {
    "Continuity Test": (
        "it reshapes a (B, C, T) batch to (B, C, 1, T) and feeds that to the model, "
        "which a 1-D convolutional model cannot accept"
    ),
    "Focus": (
        "it splits the attribution into four quadrants, which needs a second "
        "spatial axis that a (B, C, T) series does not have"
    ),
    "IROF": (
        "it segments the input into super-pixels, which needs a 2-D image "
        "rather than a series"
    ),
    "Infidelity": (
        "it pads over two spatial axes, so the pad width never matches the "
        "axes of a (B, C, T) series"
    ),
    "ROAD": (
        "its imputation resolves pixel neighbours from a second spatial axis "
        "that a (B, C, T) series does not have"
    ),
    "Smooth MPRT": (
        "its implementation calls the library's own explain function directly "
        "instead of the one it was given, so it would silently score a vanilla "
        "gradient rather than the explainer under evaluation"
    ),
    "InputInvariance": (
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

``Smooth MPRT`` is the dangerous one and the reason this table is not just about
shapes: it *returns a number*. It calls the library's own explain function
instead of the one it is given, so without this entry it would report a
plausible score for an explainer it never ran.
"""
