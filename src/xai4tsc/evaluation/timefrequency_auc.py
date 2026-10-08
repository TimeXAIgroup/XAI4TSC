"""
Time-frequency localization AUC metric.

Scores how well a frequency / time-frequency attribution **localizes** onto the
ground-truth discriminative regions of a sample. For each sample a boolean
ground-truth mask is built in the explanation's coefficient grid from the dataset
metadata (see :mod:`xai4tsc.evaluation._ground_truth`), and
the ROC-AUC of the (real) attribution against that mask is computed: 1.0 means the
attribution ranks every discriminative coefficient above every non-discriminative
one, 0.5 is chance.

Unlike the perturbation metrics this is a *ground-truth* metric — it needs the
per-sample regions, which :meth:`TimeFrequencyAUCEvaluator.evaluate` reads as
``metadata`` directly off the explanation. It is the synthetic-data counterpart to the
frequency / time-frequency perturbation metrics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from sklearn.metrics import roc_auc_score

from ..xai._types import DataType, Domain
from ._ground_truth import (
    coefficient_mask,
    regions_for_class,
    regions_for_sample,
)
from .base import EvaluatorBase

if TYPE_CHECKING:
    from torch import nn

    from ..utils.fourier_transforms import DomainTransform
    from ..xai._types import Explanation

logger = logging.getLogger(__name__)


class TimeFrequencyAUCEvaluator(EvaluatorBase):
    """
    Ground-truth localization AUC for frequency / time-frequency attributions.

    For each sample, the ROC-AUC of the (real) attribution against the
    discriminative ground-truth mask is computed; samples with no discriminative
    regions (or a fully-masked grid) are skipped (AUC is undefined there).

    Parameters
    ----------
    transform : DomainTransform, optional
        The explanation's transform; supplies the STFT grid geometry
        (``n_fft`` / ``hop_length``). Read from the explanation when left ``None``.
    metadata : list, optional
        Per-sample dataset metadata aligned to the attribution rows. Read from
        ``Explanation.metadata`` when left ``None``; required — without it (on either)
        the metric returns ``nan`` and warns.
    return_aggregate : bool
        If ``True`` (default) return the mean AUC over valid samples; otherwise
        return the per-sample AUC list (``nan`` for undefined samples).
    abs_attribution : bool
        Rank by attribution magnitude (default ``True``); set ``False`` to rank
        by signed value.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.TIME_SERIES}
    required_domains: ClassVar[set[Domain]] = {
        Domain.FREQUENCY,
        Domain.TIME_FREQUENCY,
    }

    def __init__(
        self,
        transform: DomainTransform | None = None,
        metadata: list | None = None,
        return_aggregate: bool = True,
        abs_attribution: bool = True,
    ) -> None:
        self.transform = transform
        self.metadata = metadata
        self.return_aggregate = return_aggregate
        self.abs_attribution = abs_attribution

    def evaluate(
        self,
        model: nn.Module,
        explanation: Explanation,
        data: np.ndarray,
        labels: np.ndarray,
        device: str = "cpu",
        **kwargs: object,
    ) -> float | list:
        """
        Compute the localization AUC for the batch.

        The ``transform`` and per-sample ground-truth ``metadata`` are read from the
        constructor if set, else from the explanation.

        Parameters
        ----------
        model :
            Unused (this is a ground-truth metric, not a perturbation one).
        explanation : Explanation
            Provides the attribution (``exp_values``), the domain ``transform``, and
            the per-sample ground-truth ``metadata``.
        data : np.ndarray
            Time-domain inputs ``(B, C, T)`` — sets the frequency scale ``T``.
        labels : np.ndarray
            Target class per sample (unused; the mask unions all classes).
        device : str
            Unused.
        **kwargs : object
            Ignored.

        Returns
        -------
        float or list
            Mean AUC over valid samples (``return_aggregate``) or the per-sample
            AUC list. ``nan`` where the metric is undefined / inapplicable.
        """
        transform = (
            self.transform if self.transform is not None else explanation.transform
        )
        metadata = self.metadata if self.metadata is not None else explanation.metadata
        if metadata is None:
            logger.warning(
                "TimeFrequencyAUCEvaluator requires per-sample ground-truth metadata "
                "(Explanation.metadata); none was available — returning nan. "
                "Use a dataset that provides ground-truth localization "
                "(e.g. a SyntheticDataset)."
            )
            return float("nan") if self.return_aggregate else []

        a = self.real_attribution(explanation.exp_values)
        a = np.asarray(a)
        if self.abs_attribution:
            a = np.abs(a)
        signal_length = int(np.asarray(data).shape[-1])

        targets = explanation.explained_targets
        scores: list[float] = []
        for b in range(a.shape[0]):
            sample_meta = metadata[b] if b < len(metadata) else None
            regions = regions_for_sample(sample_meta)
            # Score the attribution against the class it explains, not against
            # every region of the sample (see regions_for_class).
            target = None if targets is None else targets[b]
            regions = regions_for_class(regions, target)
            mask = coefficient_mask(regions, a[b].shape, signal_length, transform)
            if mask.any() and not mask.all():
                scores.append(float(roc_auc_score(mask.ravel(), a[b].ravel())))
            else:
                scores.append(float("nan"))

        scores_arr = np.asarray(scores, dtype=float)
        if self.return_aggregate:
            valid = scores_arr[~np.isnan(scores_arr)]
            return float(np.mean(valid)) if valid.size else float("nan")
        return scores
