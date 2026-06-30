"""
Frequency-domain perturbation metric (frequency pixel-flipping).

Ranks the explanation's frequency coefficients by importance, progressively
replaces the most-important ones with a baseline, inverts back to the time domain,
and tracks the model's predicted probability for the target class. A faithful
explanation makes the prediction drop quickly.

Implemented as :class:`FrequencyEvaluator`, an
:class:`~xai4tsc.evaluation.base.EvaluatorBase` subclass that reads the domain transform
from the explanation. The shared perturbation-curve logic lives in
:class:`PerturbationEvaluator`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from ..xai._types import DataType, Domain
from .base import EvaluatorBase

if TYPE_CHECKING:
    from torch import nn

    from ..utils.fourier_transforms import DomainTransform
    from ..xai._types import Explanation


def _zero_replacement(baseline: float) -> Callable:
    """Return a perturb function that sets the selected flat indices to *baseline*."""

    def fn(
        flat: np.ndarray, indices: np.ndarray, per_sample_shape: tuple[int, ...]
    ) -> np.ndarray:
        out = flat.copy()
        for b in range(out.shape[0]):
            out[b, indices[b]] = baseline
        return out

    return fn


def _perturbation_curve(
    model: object,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch: np.ndarray,
    transform: DomainTransform,
    *,
    features_in_step: int,
    perturb_func: Callable,
    return_aoc: bool,
    collect_frames: bool = False,
    frame_sample: int = 0,
) -> list | tuple[list, dict]:
    """
    Run a domain perturbation curve and return per-sample curves or AOC scores.

    Parameters
    ----------
    model :
        A model exposing ``predict(x) -> (classes, probs)``.
    x_batch : np.ndarray
        Time-domain inputs ``(B, C, T)``.
    y_batch : np.ndarray
        Target class index per sample.
    a_batch : np.ndarray
        Attribution in the transform's coefficient shape. Complex arrays are
        reduced to magnitude for ranking.
    transform : DomainTransform
        The domain transform (forward to coefficients, inverse back to time).
    features_in_step : int
        Number of coefficients perturbed per step.
    perturb_func : callable
        ``(flat_coeffs, indices, per_sample_shape) -> flat_coeffs``.
    return_aoc : bool
        If ``True`` return per-sample AOC (``1 - AUC / n_steps``); else the raw
        per-sample perturbation curves.
    collect_frames : bool
        If ``True`` also record per-step animation frames for a single sample and
        return ``(result, frames)``. Off by default (scoring is unaffected).
    frame_sample : int
        Index of the sample to record frames for when *collect_frames* is set.

    Returns
    -------
    list or tuple of (list, dict)
        Per-sample AOC floats / per-sample curves; plus a frame dict when
        *collect_frames* is set (keys ``time``, ``coeffs``, ``relevance``,
        ``prediction``).
    """
    coeffs = transform.forward(torch.as_tensor(x_batch, dtype=torch.float32))
    coeffs = coeffs.cpu().numpy()
    coeff_shape = coeffs.shape
    batch_size = coeff_shape[0]

    a = np.asarray(a_batch)
    a = np.abs(a) if np.iscomplexobj(a) else a
    if a.shape != coeff_shape:
        a = np.broadcast_to(a, coeff_shape)
    a_flat = a.reshape(batch_size, -1)
    n_features = a_flat.shape[-1]
    # Most-important coefficient first.
    a_indices = np.argsort(-a_flat, axis=1)

    n_steps = math.ceil(n_features / features_in_step)
    per_sample_shape = coeff_shape[1:]
    perturbed = coeffs.reshape(batch_size, -1).copy()

    preds = []
    time_frames: list = []
    coeff_frames: list = []
    for step in range(n_steps + 1):
        if step > 0:
            ix = a_indices[:, (step - 1) * features_in_step : step * features_in_step]
            perturbed = perturb_func(perturbed, ix, per_sample_shape)
        recon = transform.inverse(torch.as_tensor(perturbed.reshape(coeff_shape)))
        if torch.is_complex(recon):
            recon = recon.real
        recon = recon.cpu().numpy().reshape(x_batch.shape).astype(np.float32)
        _, probs = model.predict(recon)
        preds.append(np.asarray(probs)[np.arange(batch_size), y_batch])
        if collect_frames:
            time_frames.append(recon[frame_sample])
            coeff_frames.append(perturbed.reshape(coeff_shape)[frame_sample])

    curve = np.stack(preds, axis=1)
    if return_aoc:
        auc = np.trapz(curve, axis=1)
        result = [float(1.0 - (a_ / n_steps)) for a_ in auc]
    else:
        result = curve.tolist()

    if collect_frames:
        frames = {
            "time": np.stack(time_frames),  # (n_steps+1, C, T)
            "coeffs": np.stack(coeff_frames),  # (n_steps+1, C, *F)
            "relevance": a[frame_sample],  # (C, *F)
            "prediction": curve[frame_sample],  # (n_steps+1,)
        }
        return result, frames
    return result


def _render_animation(
    metric: object,
    animation_cls: type,
    model: object,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch: np.ndarray,
    save_path: Path | str,
    sample: int,
    fps: int,
) -> Path:
    """Collect frames for one sample and save a perturbation GIF (shared by metrics)."""
    if metric.transform is None:
        raise ValueError(
            f"{type(metric).__name__} requires a transform to animate; none set."
        )
    _, frames = _perturbation_curve(
        model,
        x_batch,
        y_batch,
        a_batch,
        metric.transform,
        features_in_step=metric.features_in_step,
        perturb_func=metric._perturb_func(),
        return_aoc=False,
        collect_frames=True,
        frame_sample=sample,
    )
    return animation_cls(frames).save(save_path, fps=fps)


class PerturbationEvaluator(EvaluatorBase, ABC):
    """
    Shared base for domain perturbation-curve metrics (frequency / time-frequency).

    Subclasses declare :attr:`required_domains`, a default ``features_in_step``
    (:attr:`_default_features_in_step`), and supply a :meth:`_perturb_func` and a
    :meth:`_animation_cls`. The domain ``transform`` is read from the explanation at
    evaluation time, or taken from the constructor for :meth:`animate`.

    Parameters
    ----------
    transform : DomainTransform, optional
        Domain transform. Usually left ``None`` for scoring (read from the
        explanation) and set explicitly only for :meth:`animate`.
    features_in_step : int, optional
        Number of coefficients perturbed per step; falls back to
        :attr:`_default_features_in_step` when ``None``.
    perturb_baseline : float
        Value the perturbed coefficients are set to (``0.0`` removes them).
    return_aoc_per_sample : bool
        Return per-sample AOC instead of the raw perturbation curves.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.TIME_SERIES}
    required_domains: ClassVar[set[Domain]] = set()
    _default_features_in_step: ClassVar[int] = 1

    def __init__(
        self,
        transform: DomainTransform | None = None,
        features_in_step: int | None = None,
        perturb_baseline: float = 0.0,
        return_aoc_per_sample: bool = False,
    ) -> None:
        self.transform = transform
        self.features_in_step = (
            self._default_features_in_step
            if features_in_step is None
            else features_in_step
        )
        self.perturb_baseline = perturb_baseline
        self.return_aoc_per_sample = return_aoc_per_sample

    @abstractmethod
    def _perturb_func(self) -> Callable:
        """Return the coefficient perturbation function for this metric."""

    @abstractmethod
    def _animation_cls(self) -> type:
        """Return the animation class for this metric's domain."""

    def evaluate(
        self,
        model: nn.Module,
        explanation: Explanation,
        data: np.ndarray,
        labels: np.ndarray,
        device: str = "cpu",
        **kwargs: object,
    ) -> list:
        """
        Compute the domain perturbation curve (or AOC) for the batch.

        The domain ``transform`` is taken from the constructor if set, else from the
        explanation; complex attributions are reduced to magnitude first.
        """
        transform = (
            self.transform if self.transform is not None else explanation.transform
        )
        if transform is None:
            raise ValueError(
                f"{type(self).__name__} requires a domain transform; none was "
                "available on the explanation."
            )
        a_batch = self.real_attribution(explanation.exp_values)
        return _perturbation_curve(
            model,
            data,
            labels,
            a_batch,
            transform,
            features_in_step=self.features_in_step,
            perturb_func=self._perturb_func(),
            return_aoc=self.return_aoc_per_sample,
        )

    def animate(
        self,
        model: object,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        save_path: Path | str,
        sample: int = 0,
        fps: int = 20,
    ) -> Path:
        """
        Render one sample's perturbation process as a GIF (does not affect scoring).

        Parameters
        ----------
        model :
            A model exposing ``predict(x) -> (classes, probs)``.
        x_batch, y_batch, a_batch : np.ndarray
            Time-domain inputs, target classes, and attributions (complex
            attributions are reduced to magnitude).
        save_path : Path or str
            Output GIF path (a ``.gif`` suffix is enforced).
        sample : int
            Index of the sample to animate.
        fps : int
            Frames per second.

        Returns
        -------
        Path
            The written GIF path.
        """
        return _render_animation(
            self,
            self._animation_cls(),
            model,
            x_batch,
            y_batch,
            a_batch,
            save_path,
            sample,
            fps,
        )


class FrequencyEvaluator(PerturbationEvaluator):
    """
    Frequency-domain perturbation curve (frequency pixel-flipping).

    Ranks the explanation's frequency coefficients by importance, progressively
    replaces the most-important ones with a baseline, inverts back to the time
    domain, and tracks the model's predicted probability for the target class.
    See :class:`PerturbationEvaluator` for the constructor parameters.
    """

    required_domains: ClassVar[set[Domain]] = {Domain.FREQUENCY}
    _default_features_in_step: ClassVar[int] = 1

    def _perturb_func(self) -> Callable:
        return _zero_replacement(self.perturb_baseline)

    def _animation_cls(self) -> type:
        from ..utils.animation import FrequencyPerturbationAnimation

        return FrequencyPerturbationAnimation
