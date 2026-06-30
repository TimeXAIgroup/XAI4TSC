"""
Perturbation helpers: ``baseline_replacement`` and ``resolve_perturb_func``.

A perturbation function follows the canonical signature
``(arr, indices, **kwargs) -> arr``: it returns a copy of *arr* with the
positions named by *indices* replaced by a baseline value.  This mirrors the
Quantus convention already used in :mod:`xai4tsc.evaluation` and lets
perturbation-based explainers (e.g. TSHAP) treat "missingness" as a pluggable
strategy.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

#: Baseline strategies understood by :func:`baseline_replacement`.
BASELINES: frozenset[str] = frozenset({"centroid", "black", "white", "mean", "random"})


def baseline_replacement(
    arr: np.ndarray,
    indices: np.ndarray,
    baseline: str = "centroid",
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
    baseline : str
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
        Random generator for ``"random"``; a default generator is used when
        ``None``.
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
        rng = prng if prng is not None else np.random.default_rng()
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True)
        noise = rng.standard_normal(size=(arr.shape[0], idx.size))
        out[:, idx] = mean + std * noise
    return out


def resolve_perturb_func(value: str | Callable) -> Callable | None:
    """
    Resolve *value* to a perturbation callable.

    A callable is returned unchanged.  A string is resolved by trying, in order:
    the names in :data:`BASELINES` (bound to :func:`baseline_replacement`), the
    top-level ``quantus`` namespace, then a dotted import path
    (e.g. ``"my_module.my_perturb"``).

    Parameters
    ----------
    value : str or callable
        A callable, a baseline name, a ``quantus`` attribute name, or a dotted
        import path.

    Returns
    -------
    callable or None
        The resolved callable, or ``None`` if a string could not be resolved.
    """
    if callable(value):
        return value
    if value in BASELINES:
        return baseline_replacement
    try:
        import quantus as _q

        candidate = getattr(_q, value, None)
        if candidate and callable(candidate):
            return candidate
    except ImportError:
        pass
    if "." in value:
        module_path, _, attr = value.rpartition(".")
        try:
            mod = importlib.import_module(module_path)
            candidate = getattr(mod, attr, None)
            if candidate and callable(candidate):
                return candidate
        except (ImportError, AttributeError):
            pass
    logger.warning("Could not resolve perturb_func '%s'.", value)
    return None
