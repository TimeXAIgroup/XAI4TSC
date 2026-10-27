"""``Explanation`` dataclass; re-exports the ``DataType`` / ``Domain`` vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .._domains import DATA_DOMAINS, DataType, Domain, resolve_data_domain

__all__ = [
    "DATA_DOMAINS",
    "DataType",
    "Domain",
    "Explanation",
    "is_multi_target",
    "resolve_data_domain",
]

if TYPE_CHECKING:
    from xai4tsc.utils.fourier_transforms import DomainTransform


@dataclass
class Explanation:
    """
    A standard container for explanation results.

    Attributes
    ----------
    explainer : str
        Name of the explainer method that produced this explanation.
    explainer_params : dict
        Parameters the explainer was built with, as supplied. Lets a consumer
        rebuild an equivalently configured explainer via
        :func:`~xai4tsc.xai.explain.build_explainer`, which is what the
        re-explaining evaluation metrics need. Empty when none were given.
    explainer_call_params : dict
        Per-call options ``explain()`` was called with, beside the construction
        options in ``explainer_params``. Recorded so the re-explaining metrics
        replay the same call: an attribution scored with one set of call options
        and re-computed with another would make the metric report a configuration
        difference as a property of the explainer. Empty when none were given.
    explanation_type : str
        Category of explanation, e.g. ``"feature_attribution"``.
    exp_values : np.ndarray
        Raw explanation values. Rank depends on ``explanation_domain``: a
        single-target explanation is ``(n_samples, C, T)`` for :attr:`Domain.TIME`
        and :attr:`Domain.FREQUENCY` (``T`` is the coefficient count for the
        latter) and ``(n_samples, C, n_freq, n_time)`` for
        :attr:`Domain.TIME_FREQUENCY`. Explaining all classes (``targets="all"``)
        prepends a leading class axis, so the rank is one higher; use
        :func:`is_multi_target` to detect this rather than inspecting ``ndim``
        alone.
    data : np.ndarray
        Original input samples that were explained, shape ``(N, C, T)``.
    labels : np.ndarray or None
        Class labels for each sample in ``data``; ``None`` when the explanation
        was generated without labels.
    indices : np.ndarray
        Indices into the test set from which ``data`` was drawn.
    encoder : object
        Fitted label encoder used during dataset preparation.
    meta : dict
        Any extra metadata returned by the explainer.
    data_domain : Domain
        Signal domain of ``data``. Defaults to :attr:`Domain.TIME`.
    explanation_domain : Domain
        Signal domain of ``exp_values``. Defaults to :attr:`Domain.TIME`. May
        differ from ``data_domain`` (e.g. a frequency explainer consumes
        time-domain data and emits frequency-domain relevance).
    transform : DomainTransform, optional
        The stateful transform mapping between the time domain and
        ``data_domain`` / ``explanation_domain``. ``None`` for pure time-domain
        explanations. Freq/TF views of ``data`` are *derived* on demand via
        ``transform.forward(data)`` rather than stored.
    explained_targets : np.ndarray, optional
        Class index each attribution explains, one per sample. An attribution is
        always *for* a class, and a ground-truth localization metric has to know
        which one to pick the right discriminative region. ``None`` when the
        explanation holds one attribution per class (``targets="all"``), where the
        consumer splits the stack and sets this per class, or when the producer did
        not record it.
    metadata : list, optional
        Per-sample dataset metadata for the explained samples, aligned to
        ``exp_values`` / ``indices`` (one entry per row). Carries ground-truth
        localization (e.g. ``{"ground_truth": {class: [regions]}, ...}``) from
        datasets that provide it; consumed by ground-truth localization metrics
        such as ``TimeFrequencyAUC``. ``None`` when the dataset has no metadata.
    target_mask : np.ndarray, optional
        Which attributions of a multi-target stack apply, shape
        ``(n_targets, n_samples)`` of ``bool``. ``None`` means every one does,
        which is the single-target case and ``targets="all"``.

        It exists for the per-output (multi-label) formulation, where a
        prediction is a *set* of labels: ``targets="label"`` marks the labels a
        sample truly carries and ``targets="predicted"`` those over the decision
        threshold, so the number of attributions that mean anything varies per
        sample. A ragged array cannot hold that, and flattening it would break
        the invariant that a row of ``exp_values`` is a sample lining up with
        ``indices`` and ``metadata`` - so the stack stays rectangular and this
        says which of its rows mean anything. :meth:`rows_for_target` reads it.
    """

    explainer: str
    explainer_params: dict
    explanation_type: str
    exp_values: np.ndarray
    data: np.ndarray
    labels: np.ndarray | None
    indices: np.ndarray
    encoder: object
    meta: dict
    data_domain: Domain = Domain.TIME
    explanation_domain: Domain = Domain.TIME
    transform: DomainTransform | None = None
    metadata: list | None = None
    explained_targets: np.ndarray | None = None
    explainer_call_params: dict = field(default_factory=dict)
    target_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate the target mask against the attribution stack."""
        self.validate_target_mask()

    def validate_target_mask(self) -> None:
        """
        Check that ``target_mask`` lines up with ``exp_values``.

        Called on every construction of the dataclass, and again once
        ``exp_values`` has been filled in - the container is built with
        ``exp_values=None`` and populated afterwards, so construction alone
        cannot see the shape. A mask that has drifted out of step with the stack
        would otherwise pair one target's attributions with another target's
        samples, silently.

        Raises
        ------
        ValueError
            If the mask is not ``(n_targets, n_samples)`` for this stack.
        """
        if self.target_mask is None or self.exp_values is None:
            return
        mask = np.asarray(self.target_mask)
        expected = self.exp_values.shape[:2]
        if mask.shape != expected:
            raise ValueError(
                f"target_mask has shape {mask.shape}, but the attribution stack "
                f"is {self.exp_values.shape}, so the mask must be {expected} "
                "(one row per target, one column per sample)."
            )

    def rows_for_target(self, target: int) -> np.ndarray:
        """
        Return the samples whose attribution for *target* is meaningful.

        Parameters
        ----------
        target : int
            Index along the stack's leading (target) axis.

        Returns
        -------
        np.ndarray
            Boolean selector over samples. All ``True`` when no
            ``target_mask`` is set, so a caller needs no special case.
        """
        if self.target_mask is None:
            # From `data`, which is always the explained samples; `indices` is
            # optional on a hand-built explanation.
            return np.ones(np.shape(self.data)[0], dtype=bool)
        return np.asarray(self.target_mask[target], dtype=bool)


# Rank of a *single-target* ``exp_values`` array per explanation domain. Frequency
# is a 1-D coefficient axis like time ``(n, C, F)``; time-frequency adds a second
# spectral axis ``(n, C, n_freq, n_time)``. Explaining all classes prepends a
# leading class axis, so the multi-target rank is this value plus one.
_SINGLE_TARGET_NDIM = {
    Domain.TIME: 3,
    Domain.FREQUENCY: 3,
    Domain.TIME_FREQUENCY: 4,
}


def is_multi_target(exp: Explanation) -> bool:
    """
    Report whether an explanation holds one attribution per class.

    A ``targets="all"`` explanation stacks a per-class attribution on a leading
    axis, so its ``exp_values`` rank is one higher than a single-target array in
    the same domain. Because the single-target rank differs by domain (3 for
    time / frequency, 4 for time-frequency), the domain is required to tell a
    multi-target array apart from a single-target time-frequency one (both 4-D).

    Parameters
    ----------
    exp : Explanation
        The explanation to inspect (uses ``exp_values`` and
        ``explanation_domain``).

    Returns
    -------
    bool
        ``True`` when ``exp.exp_values`` carries a leading per-class axis.
    """
    return exp.exp_values.ndim == _SINGLE_TARGET_NDIM[exp.explanation_domain] + 1
