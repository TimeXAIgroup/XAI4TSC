"""
Time-frequency perturbation metrics (STFT pixel-flipping).

Like :class:`~xai4tsc.evaluation.frequency_evaluate.FrequencyEvaluator` but in the
time-frequency (STFT) domain. The plain variant (:class:`TimeFrequencyEvaluator`) zeroes
the most-important coefficients; the Gaussian variant
(:class:`TimeFrequencyEvaluatorGaussian`) smears each removed coefficient with a
Gaussian kernel before blending toward the baseline (a softer, spectrogram-aware
perturbation). Both are
:class:`~xai4tsc.evaluation.frequency_evaluate.PerturbationEvaluator` subclasses.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy.ndimage import gaussian_filter

from ..xai._types import Domain
from .frequency_evaluate import PerturbationEvaluator, _zero_replacement

if TYPE_CHECKING:
    from ..utils.fourier_transforms import DomainTransform


def _gaussian_replacement(baseline: float, sigma: Sequence[float]) -> Callable:
    """
    Return a perturb function that Gaussian-smears each removed TF coefficient.

    For every selected ``(channel, frequency, time)`` cell, a unit impulse is
    smoothed with :func:`scipy.ndimage.gaussian_filter`, normalised to ``[0, 1]``,
    and used to blend that channel's spectrogram toward *baseline*.
    """

    def fn(
        flat: np.ndarray, indices: np.ndarray, per_sample_shape: tuple[int, ...]
    ) -> np.ndarray:
        n_channels, n_freq, n_time = per_sample_shape
        out = flat.copy()
        for b in range(out.shape[0]):
            x = out[b].reshape(n_channels, n_freq, n_time)
            for idx in indices[b]:
                channel = idx // (n_freq * n_time)
                remaining = idx % (n_freq * n_time)
                freq = remaining // n_time
                timestep = remaining % n_time
                mask = np.zeros((n_freq, n_time), dtype=float)
                mask[freq, timestep] = 1.0
                mask = gaussian_filter(mask, sigma=sigma)
                mask = mask / (mask.max() + 1e-12)
                x[channel] = (1 - mask) * x[channel] + mask * baseline
            out[b] = x.reshape(-1)
        return out

    return fn


class TimeFrequencyEvaluator(PerturbationEvaluator):
    """
    Time-frequency perturbation curve (STFT pixel-flipping).

    Like :class:`~xai4tsc.evaluation.frequency_evaluate.FrequencyEvaluator` but in the
    time-frequency (STFT) domain: the most-important STFT coefficients are zeroed. See
    :class:`~xai4tsc.evaluation.frequency_evaluate.PerturbationEvaluator` for the
    constructor parameters.
    """

    required_domains: ClassVar[set[Domain]] = {Domain.TIME_FREQUENCY}
    _default_features_in_step: ClassVar[int] = 5

    def _perturb_func(self) -> Callable:
        return _zero_replacement(self.perturb_baseline)

    def _animation_cls(self) -> type:
        from ..utils.animation import TimeFrequencyPerturbationAnimation

        return TimeFrequencyPerturbationAnimation


class TimeFrequencyEvaluatorGaussian(TimeFrequencyEvaluator):
    """
    Time-frequency perturbation with a Gaussian-smeared baseline replacement.

    Parameters
    ----------
    sigma : sequence of float
        Standard deviations ``(freq, time)`` of the Gaussian kernel applied around
        each removed coefficient.
    transform, features_in_step, perturb_baseline, return_aoc_per_sample :
        See :class:`~xai4tsc.evaluation.frequency_evaluate.PerturbationEvaluator`.
    """

    def __init__(
        self,
        transform: DomainTransform | None = None,
        features_in_step: int | None = None,
        perturb_baseline: float = 0.0,
        return_aoc_per_sample: bool = False,
        sigma: Sequence[float] = (5.0, 1.0),
    ) -> None:
        super().__init__(
            transform=transform,
            features_in_step=features_in_step,
            perturb_baseline=perturb_baseline,
            return_aoc_per_sample=return_aoc_per_sample,
        )
        self.sigma = sigma

    def _perturb_func(self) -> Callable:
        return _gaussian_replacement(self.perturb_baseline, self.sigma)
