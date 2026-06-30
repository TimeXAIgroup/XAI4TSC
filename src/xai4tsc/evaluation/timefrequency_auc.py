"""
Time-frequency localization AUC metric.

Scores how well a frequency / time-frequency attribution **localizes** onto the
ground-truth discriminative regions of a sample. For each sample a boolean
ground-truth mask is built in the explanation's coefficient grid from the dataset
metadata (the ``{class: [regions]}`` produced by e.g. ``FreqShapesDataset``), and
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
from .base import EvaluatorBase

if TYPE_CHECKING:
    from torch import nn

    from ..utils.fourier_transforms import DomainTransform
    from ..xai._types import Explanation

logger = logging.getLogger(__name__)


def _sample_regions(sample_meta: object) -> list[dict]:
    """
    Return the discriminative regions for one sample (union across classes).

    Parameters
    ----------
    sample_meta : object
        A per-sample metadata mapping with a ``"ground_truth"`` key. Two shapes are
        accepted: the per-class dict ``{class_key: [region, ...]}`` (class keys may
        be ``int`` or ``str`` after a JSON round-trip) or a flat ``[region, ...]``
        list (the regions of all classes already unioned, as shipped by the
        pre-split ``freq_shapes`` dataset). Either way the regions are unioned, so
        the two are equivalent here. Anything else yields no regions.

    Returns
    -------
    list of dict
        The flattened discriminative regions; each region is
        ``{"channel", "pos", "len", "freq", "phase"}``.
    """
    if not isinstance(sample_meta, dict):
        return []
    gt = sample_meta.get("ground_truth", {})
    if isinstance(gt, list):  # flat union of regions (per-class keying dropped)
        return [r for r in gt if isinstance(r, dict)]
    if not isinstance(gt, dict):
        return []
    regions: list[dict] = []
    for region_list in gt.values():
        if isinstance(region_list, list):
            regions.extend(r for r in region_list if isinstance(r, dict))
    return regions


def _ground_truth_mask(
    regions: list[dict],
    grid_shape: tuple[int, ...],
    signal_length: int,
    transform: DomainTransform | None,
) -> np.ndarray:
    """
    Build a boolean ground-truth mask in the explanation's coefficient grid.

    Parameters
    ----------
    regions : list of dict
        Discriminative regions for one sample (see :func:`_sample_regions`).
    grid_shape : tuple of int
        Shape of one sample's attribution: ``(C, n_freq)`` for a frequency
        explanation or ``(C, n_freq, n_time)`` for a time-frequency one.
    signal_length : int
        Length ``T`` of the time-domain series (sets the frequency scale).
    transform : DomainTransform or None
        The explanation's transform; ``n_fft`` / ``hop_length`` are read from it
        for the STFT grid geometry when present.

    Returns
    -------
    np.ndarray
        Boolean array of shape *grid_shape*; ``True`` at discriminative cells.
    """
    mask = np.zeros(grid_shape, dtype=bool)
    if not regions:
        return mask

    n_channels = grid_shape[0]
    n_freq = grid_shape[1]
    is_tf = len(grid_shape) == 3
    n_time = grid_shape[2] if is_tf else None
    n_fft = getattr(transform, "n_fft", None)
    hop = getattr(transform, "hop_length", None)

    for region in regions:
        channel = int(region.get("channel", 0))
        if channel < 0 or channel >= n_channels:
            continue
        freq = float(region.get("freq", 0.0))
        # STFT bin scales cycles/sample (freq/T) onto the n_fft grid; rFFT bin over
        # the full series is just the cycles-per-series count == freq.
        scale = n_fft / signal_length if (is_tf and n_fft) else 1.0
        f_bin = min(max(round(freq * scale), 0), n_freq - 1)

        if is_tf:
            pos = int(region.get("pos", 0))
            length = int(region.get("len", signal_length))
            if hop:
                # torch.stft uses center=True: frame f is centred at f*hop.
                f0 = max(0, pos // hop)
                f1 = min(n_time, (pos + length) // hop + 1)
                mask[channel, f_bin, f0:f1] = True
            else:
                mask[channel, f_bin, :] = True
        else:
            mask[channel, f_bin] = True
    return mask


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

        scores: list[float] = []
        for b in range(a.shape[0]):
            sample_meta = metadata[b] if b < len(metadata) else None
            regions = _sample_regions(sample_meta)
            mask = _ground_truth_mask(regions, a[b].shape, signal_length, transform)
            if mask.any() and not mask.all():
                scores.append(float(roc_auc_score(mask.ravel(), a[b].ravel())))
            else:
                scores.append(float("nan"))

        scores_arr = np.asarray(scores, dtype=float)
        if self.return_aggregate:
            valid = scores_arr[~np.isnan(scores_arr)]
            return float(np.mean(valid)) if valid.size else float("nan")
        return scores
