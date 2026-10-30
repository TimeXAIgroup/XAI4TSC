"""
Perturbation helper: ``baseline_replacement``.

A perturbation function follows the canonical signature
``(arr, indices, **kwargs) -> arr``: it returns a copy of *arr* with the
positions named by *indices* replaced by a baseline value.  This mirrors the
Quantus convention already used in :mod:`xai4tsc.evaluation` and lets
perturbation-based explainers (e.g. TSHAP) treat "missingness" as a pluggable
strategy.
"""

from __future__ import annotations

import logging
from typing import Literal, get_args

import numpy as np

from .._random import DEFAULT_SEED, resolve_rng

logger = logging.getLogger(__name__)

Baseline = Literal["centroid", "black", "white", "mean", "random"]
"""Baseline strategies understood by :func:`baseline_replacement`."""

#: The values of :data:`Baseline`.
BASELINES: frozenset[str] = frozenset(get_args(Baseline))


def baseline_replacement(
    arr: np.ndarray,
    indices: np.ndarray,
    baseline: Baseline = "centroid",
    reference: np.ndarray | None = None,
    prng: np.random.Generator | None = None,
    **kwargs: object,
) -> np.ndarray:
    """
    Return a copy of *arr* with *indices* replaced by a *baseline* value.

    Parameters
    ----------
    arr : np.ndarray
        Single series of shape ``(C, T)``.
    indices : np.ndarray
        Time indices (along the last axis) to replace.  An empty array returns
        an unmodified copy.
    baseline : Baseline
        One of :data:`BASELINES`:

        - ``"centroid"`` — replace with the matching columns of *reference*
          (a precomputed ``(C, T)`` background waveform, e.g. a class/dataset
          centroid).  Falls back to ``"mean"`` if *reference* is ``None``.
        - ``"black"`` — replace with zeros.
        - ``"white"`` — replace with ones.
        - ``"mean"`` — replace with each channel's own temporal mean.
        - ``"random"`` — replace with Gaussian noise matched to each channel's
          mean and standard deviation (uses *prng* when supplied).
    reference : np.ndarray, optional
        Background waveform of shape ``(C, T)`` used by ``"centroid"``.
    prng : np.random.Generator, optional
        Random generator for ``"random"``; when ``None`` a generator seeded
        with :data:`~xai4tsc.DEFAULT_SEED` is used, so the draw is reproducible
        unless a caller shares its own stream.
    **kwargs : object
        Ignored; accepted so callers can pass through extra perturbation
        options uniformly.

    Returns
    -------
    np.ndarray
        A copy of *arr* with the requested replacement applied.

    Raises
    ------
    ValueError
        If *baseline* is not one of :data:`BASELINES`.
    """
    if baseline not in BASELINES:
        raise ValueError(
            f"Unknown baseline '{baseline}'. Supported: {sorted(BASELINES)}."
        )
    out = np.array(arr, copy=True)
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return out

    if baseline == "centroid":
        if reference is None:
            logger.warning(
                "baseline='centroid' but no reference supplied; using 'mean'."
            )
            out[:, idx] = arr.mean(axis=1, keepdims=True)
        else:
            out[:, idx] = np.asarray(reference)[:, idx]
    elif baseline == "black":
        out[:, idx] = 0.0
    elif baseline == "white":
        out[:, idx] = 1.0
    elif baseline == "mean":
        out[:, idx] = arr.mean(axis=1, keepdims=True)
    elif baseline == "random":
        rng = resolve_rng(DEFAULT_SEED, prng)
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True)
        noise = rng.standard_normal(size=(arr.shape[0], idx.size))
        out[:, idx] = mean + std * noise
    return out
