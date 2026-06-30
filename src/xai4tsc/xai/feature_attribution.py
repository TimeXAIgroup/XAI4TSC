"""
Captum-backed feature attribution explainers (IG, DeepLIFT, Occlusion, ...).

Also hosts :class:`TSHAPExplainer`, a pure-NumPy perturbation method (exact
2-player Shapley over time windows) that only queries ``model.predict``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from captum.attr import (
    Deconvolution,
    DeepLift,
    GuidedBackprop,
    IntegratedGradients,
    Occlusion,
)

from ..utils.perturbation import baseline_replacement, resolve_perturb_func
from ._types import DataType
from .base import GradientExplainer, PerturbationExplainer

if TYPE_CHECKING:
    from ._types import Explanation

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer(GradientExplainer):
    """
    Feature attribution via Integrated Gradients (Captum).

    Parameters
    ----------
    multiply_by_inputs : bool
        Multiply attributions by ``(input - baseline)``.  Matches the
        standard IG formulation.
    n_steps : int
        Number of steps along the integration path.  Higher values give more
        accurate approximations at the cost of compute.
    integration_method : str
        Quadrature rule for approximating the integral.  One of
        ``"gausslegendre"`` (default), ``"riemann_trapezoid"``,
        ``"riemann_middle"``, ``"riemann_right"``, ``"riemann_left"``.
    baselines : int or float
        Reference input value used as the integration baseline.
    internal_batch_size : int or None
        Split the integration steps into sub-batches of this size to reduce
        memory usage.  ``None`` processes all steps at once.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.AGNOSTIC}
    """Gradient attribution is domain-agnostic — applies to any input."""

    def __init__(
        self,
        multiply_by_inputs: bool = True,
        n_steps: int = 50,
        integration_method: str = "gausslegendre",
        baselines: int = 0,
        internal_batch_size: int | None = None,
    ) -> None:
        self._multiply_by_inputs = multiply_by_inputs
        self._attribute_kwargs = {
            "n_steps": n_steps,
            "method": integration_method,
            "baselines": baselines,
        }
        if internal_batch_size is not None:
            self._attribute_kwargs["internal_batch_size"] = internal_batch_size

    def _get_captum_attribution(self, model: torch.nn.Module) -> IntegratedGradients:
        return IntegratedGradients(model, multiply_by_inputs=self._multiply_by_inputs)


class DeepLiftExplainer(GradientExplainer):
    """
    Feature attribution via DeepLIFT (Captum).

    Parameters
    ----------
    multiply_by_inputs : bool
        Multiply contribution scores by ``(input - reference)``.  Matches
        the original DeepLIFT paper formulation.
    eps : float
        Small constant added for numerical stability in gradient computation.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.AGNOSTIC}

    def __init__(self, multiply_by_inputs: bool = True, eps: float = 1e-10) -> None:
        self._multiply_by_inputs = multiply_by_inputs
        self._eps = eps

    def _get_captum_attribution(self, model: torch.nn.Module) -> DeepLift:
        return DeepLift(
            model, multiply_by_inputs=self._multiply_by_inputs, eps=self._eps
        )


class DeconvolutionExplainer(GradientExplainer):
    """Feature attribution via Deconvolution (Captum)."""

    data_applicability: ClassVar[set[DataType]] = {DataType.AGNOSTIC}

    def _get_captum_attribution(self, model: torch.nn.Module) -> Deconvolution:
        return Deconvolution(model)


class GuidedBackpropagationExplainer(GradientExplainer):
    """Feature attribution via Guided Backpropagation (Captum)."""

    data_applicability: ClassVar[set[DataType]] = {DataType.AGNOSTIC}

    def _get_captum_attribution(self, model: torch.nn.Module) -> GuidedBackprop:
        return GuidedBackprop(model)


class OcclusionExplainer(PerturbationExplainer):
    """
    Feature attribution via Occlusion (Captum).

    Parameters
    ----------
    window_shape : list
        Shape of the sliding occlusion window, e.g. ``[1, 5]``.
    baseline : int or float
        Value used to fill the occluded region.
    strides : int
        Step size of the sliding window.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.AGNOSTIC}

    def __init__(
        self,
        window_shape: list | None = None,
        baseline: int = 0,
        strides: int = 4,
    ) -> None:
        self.window_shape = window_shape if window_shape is not None else [1, 5]
        self.baseline = baseline
        self.strides = strides

    def explain(
        self,
        model: torch.nn.Module,
        exp: Explanation,
        device: str | torch.device,
        targets: list | None,
        **kwargs: object,
    ) -> np.ndarray:
        """Compute occlusion attributions for the samples in *exp*."""
        explainer = Occlusion(model)
        data_tensor = torch.tensor(exp.data, dtype=torch.float32, device=device)
        return (
            explainer.attribute(
                data_tensor,
                target=targets,
                sliding_window_shapes=tuple(self.window_shape),
                baselines=self.baseline,
                strides=self.strides,
            )
            .detach()
            .cpu()
            .numpy()
        )


class TSHAPExplainer(PerturbationExplainer):
    r"""
    Feature attribution via TSHAP — exact 2-player Shapley over time windows.

    For a sample ``x`` and a background series ``x̄``, the exact Shapley value of
    a window ``w`` (the timesteps inside it, with ``w̄`` the rest) is

    .. math::

        \varphi(w) = \tfrac12\,[\,f(x) - f(\{\bar x_w, x_{\bar w}\})
                     + f(\{x_w, \bar x_{\bar w}\}) - f(\bar x)\,]

    where ``f`` is the predicted probability of the per-sample target class.
    ``{x̄_w, x_w̄}`` masks the window (inside replaced by background, rest
    original) and ``{x_w, x̄_w̄}`` keeps only the window (rest replaced by
    background).  Per-timestep relevance is the mean of ``φ(w) / window_length``
    over all windows containing that timestep.  With ``stride > 1`` the Shapley
    value is computed only at strided window starts and linearly interpolated in
    between.

    Note
    ----
    The per-window Shapley values are identical to the original ``mlgig/tshap``
    reference implementation (verified to machine precision).  The per-timestep
    aggregation here follows the paper's Eq. 9 — the **mean** over the windows
    covering each timestep (i.e. divided by the coverage count).  The reference
    *code* instead **sums** ``φ(w) / window_length`` without dividing by the
    coverage count, so its per-timestep magnitudes equal these multiplied by the
    per-timestep coverage count.  This implementation is the paper-faithful one.

    Parameters
    ----------
    window_length : float or int
        Window size.  A float in ``(0, 1]`` is a fraction of the series length
        ``T``; an int is an absolute number of timesteps.
    stride : int
        Compute the Shapley value every ``stride`` window starts and interpolate
        the rest.  ``1`` evaluates every window.
    perturb_baseline : str
        Background strategy (see
        :func:`xai4tsc.utils.perturbation.baseline_replacement`): ``"centroid"``
        (default), ``"black"``, ``"white"``, ``"mean"`` or ``"random"``.
        ``"centroid"`` requires *background_data*; without it a warning is
        emitted and ``"mean"`` is used.
    perturb_func : str or callable, optional
        Override the perturbation function.  A callable, a baseline name, a
        ``quantus`` attribute name, or a dotted import path.  ``None`` uses the
        built-in :func:`baseline_replacement`.
    n_perturb_samples : int
        Number of background draws to average over (Eq. 8).  Only meaningful for
        the stochastic ``"random"`` baseline.
    channel_mode : str
        ``"shared"`` (default) — a window spans all channels and one Shapley
        value is broadcast to every channel (cost independent of ``C``).
        ``"per_channel"`` — each ``(channel, window)`` is its own player, giving
        a channel-resolved map at ``C`` times the cost (an extension beyond the
        paper's univariate scope).
    roi : bool
        If ``True`` apply TSHAP-ROI (Algorithm 2): keep only windows whose
        ``|φ(w)|`` exceeds ``0.1 · max|φ(w)|``, merge consecutive kept windows
        into regions, recompute one Shapley value per region, and assign it
        uniformly inside the region (zero elsewhere).
    background_data : np.ndarray, optional
        Background samples of shape ``(n, C, T)``.  The centroid
        ``background_data.mean(axis=0)`` is used as the ``"centroid"`` reference.
        The runner resolves the YAML ``background_data`` selector to this array;
        passing a non-array raises (selectors are a runner-only concept).
    seed : int, optional
        Seed for the random baseline, for reproducible draws.
    """

    explanation_type = "feature_attribution"
    data_applicability: ClassVar[set[DataType]] = {DataType.TIME_SERIES}
    """TSHAP is a time-series-specific perturbation method (windows over time)."""

    def __init__(
        self,
        window_length: float | int = 0.1,
        stride: int = 5,
        perturb_baseline: str = "centroid",
        perturb_func: str | Callable | None = None,
        n_perturb_samples: int = 1,
        channel_mode: str = "shared",
        roi: bool = False,
        background_data: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        if channel_mode not in ("shared", "per_channel"):
            raise ValueError(
                f"channel_mode must be 'shared' or 'per_channel', got '{channel_mode}'."
            )
        if background_data is not None and not isinstance(background_data, np.ndarray):
            raise TypeError(
                "background_data must be a numpy array. The runner resolves the "
                "'background_data' YAML selector to an array; selectors are a "
                "runner-only concept and cannot reach the package."
            )

        self.window_length = window_length
        self.stride = max(1, int(stride))
        self.channel_mode = channel_mode
        self.roi = bool(roi)
        self.n_perturb_samples = max(1, int(n_perturb_samples))
        self.prng = np.random.default_rng(seed)

        self.perturb_baseline = perturb_baseline
        self.centroid: np.ndarray | None = None
        if perturb_baseline == "centroid":
            if isinstance(background_data, np.ndarray):
                self.centroid = background_data.mean(axis=0).astype(np.float32)
            else:
                logger.warning(
                    "perturb_baseline='centroid' but no background_data "
                    "supplied; falling back to 'mean'."
                )
                self.perturb_baseline = "mean"

        if perturb_func is None:
            self._perturb_func: Callable = baseline_replacement
        else:
            resolved = resolve_perturb_func(perturb_func)
            if resolved is None:
                logger.warning(
                    "Could not resolve perturb_func %r; using baseline_replacement.",
                    perturb_func,
                )
                resolved = baseline_replacement
            self._perturb_func = resolved
        self._is_baseline_func = self._perturb_func is baseline_replacement

    # ── public API ───────────────────────────────────────────────────────────

    def explain(
        self,
        model: torch.nn.Module,
        exp: Explanation,
        device: str | torch.device,
        targets: list | None,
        **kwargs: object,
    ) -> np.ndarray:
        """Compute TSHAP attributions for the samples in *exp*."""
        data = np.asarray(exp.data, dtype=np.float32)
        n_samples, _, n_timesteps = data.shape
        if targets is None:
            targets = model.predict(data)[0].tolist()
        w_l = self._window_len(n_timesteps)

        out = np.zeros_like(data, dtype=np.float32)
        for i in range(n_samples):
            out[i] = self._explain_sample(model, data[i], int(targets[i]), w_l)
        return out

    # ── per-sample driver ────────────────────────────────────────────────────

    def _explain_sample(
        self,
        model: torch.nn.Module,
        x: np.ndarray,
        target: int,
        w_l: int,
    ) -> np.ndarray:
        n_channels, n_timesteps = x.shape
        all_starts = list(range(max(1, n_timesteps - w_l + 1)))
        eval_starts = sorted(
            set(all_starts[:: self.stride]) | {all_starts[0], all_starts[-1]}
        )

        draws = []
        for _ in range(self.n_perturb_samples):
            x_bar = self._full_background(x)
            f_x_bar = self._target_probs(model, x_bar[None], target)[0]
            draws.append((x_bar, f_x_bar))
        f_x = self._target_probs(model, x[None], target)[0]

        channels: list[int | None] = (
            [None] if self.channel_mode == "shared" else list(range(n_channels))
        )
        attr = np.zeros((n_channels, n_timesteps), dtype=np.float32)
        for ch in channels:
            phi_eval = np.zeros(len(eval_starts))
            for x_bar, f_x_bar in draws:
                masks = [
                    self._window_mask(n_channels, n_timesteps, s, w_l, ch)
                    for s in eval_starts
                ]
                phi_eval += self._phi_for_masks(
                    model, x, x_bar, target, f_x, f_x_bar, masks
                )
            phi_eval /= len(draws)

            if len(eval_starts) > 1:
                phi_all = np.interp(all_starts, eval_starts, phi_eval)
            else:
                phi_all = np.full(len(all_starts), phi_eval[0])

            if self.roi:
                row = self._roi_1d(
                    model, x, draws, target, f_x, all_starts, phi_all, w_l, ch
                )
            else:
                row = self._aggregate_1d(phi_all, all_starts, w_l, n_timesteps)

            if ch is None:
                attr[:] = row
            else:
                attr[ch] = row
        return attr

    # ── Shapley primitives ───────────────────────────────────────────────────

    def _phi_for_masks(
        self,
        model: torch.nn.Module,
        x: np.ndarray,
        x_bar: np.ndarray,
        target: int,
        f_x: float,
        f_x_bar: float,
        masks: list[np.ndarray],
    ) -> np.ndarray:
        """Return ``φ(w)`` (Eq. 7) for each boolean player *mask*, batched."""
        variants = []
        for m in masks:
            variants.append(np.where(m, x_bar, x))  # masked: window → background
            variants.append(np.where(m, x, x_bar))  # kept: rest → background
        batch = np.stack(variants).astype(np.float32)
        probs = self._target_probs(model, batch, target)
        f_masked = probs[0::2]
        f_kept = probs[1::2]
        return 0.5 * (f_x - f_masked + f_kept - f_x_bar)

    def _target_probs(
        self, model: torch.nn.Module, batch: np.ndarray, target: int
    ) -> np.ndarray:
        """Return the *target*-class probability for each series in *batch*."""
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        _, probs = model.predict(batch)
        return np.asarray(probs)[:, target]

    def _full_background(self, x: np.ndarray) -> np.ndarray:
        """Replace every timestep of *x* with the chosen baseline → ``x̄``."""
        all_idx = np.arange(x.shape[1])
        if self._is_baseline_func:
            return self._perturb_func(
                x,
                all_idx,
                baseline=self.perturb_baseline,
                reference=self.centroid,
                prng=self.prng,
            )
        return self._perturb_func(x, all_idx)

    # ── aggregation & ROI ────────────────────────────────────────────────────

    def _aggregate_1d(
        self,
        phi_all: np.ndarray,
        all_starts: list[int],
        w_l: int,
        n_timesteps: int,
    ) -> np.ndarray:
        """Eq. 9: mean of ``φ(w) / w_l`` over windows covering each timestep."""
        acc = np.zeros(n_timesteps)
        cnt = np.zeros(n_timesteps)
        for s, phi in zip(all_starts, phi_all, strict=True):
            end = min(s + w_l, n_timesteps)
            acc[s:end] += phi / w_l
            cnt[s:end] += 1
        return np.divide(acc, cnt, out=np.zeros(n_timesteps), where=cnt > 0).astype(
            np.float32
        )

    def _roi_1d(
        self,
        model: torch.nn.Module,
        x: np.ndarray,
        draws: list[tuple[np.ndarray, float]],
        target: int,
        f_x: float,
        all_starts: list[int],
        phi_all: np.ndarray,
        w_l: int,
        ch: int | None,
    ) -> np.ndarray:
        """TSHAP-ROI (Algorithm 2): one fresh Shapley value per merged region."""
        n_channels, n_timesteps = x.shape
        out = np.zeros(n_timesteps, dtype=np.float32)
        if phi_all.size == 0:
            return out
        eps = 0.1 * np.max(np.abs(phi_all))
        relevant = np.abs(phi_all) > eps
        for group in self._consecutive_groups(relevant):
            start = all_starts[group[0]]
            end = min(all_starts[group[-1]] + w_l, n_timesteps)
            region_len = end - start
            mask = self._region_mask(n_channels, n_timesteps, start, end, ch)
            phi_region = 0.0
            for x_bar, f_x_bar in draws:
                phi_region += self._phi_for_masks(
                    model, x, x_bar, target, f_x, f_x_bar, [mask]
                )[0]
            phi_region /= len(draws)
            out[start:end] = phi_region / region_len
        return out

    @staticmethod
    def _consecutive_groups(flags: np.ndarray) -> list[list[int]]:
        """Group indices of consecutive ``True`` runs in *flags*."""
        groups: list[list[int]] = []
        current: list[int] = []
        for i, flag in enumerate(flags):
            if flag:
                current.append(i)
            elif current:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    # ── masks & window length ────────────────────────────────────────────────

    @staticmethod
    def _window_mask(
        n_channels: int, n_timesteps: int, start: int, w_l: int, ch: int | None
    ) -> np.ndarray:
        return TSHAPExplainer._region_mask(
            n_channels, n_timesteps, start, min(start + w_l, n_timesteps), ch
        )

    @staticmethod
    def _region_mask(
        n_channels: int, n_timesteps: int, start: int, end: int, ch: int | None
    ) -> np.ndarray:
        mask = np.zeros((n_channels, n_timesteps), dtype=bool)
        if ch is None:
            mask[:, start:end] = True
        else:
            mask[ch, start:end] = True
        return mask

    def _window_len(self, n_timesteps: int) -> int:
        w_l = self.window_length
        if isinstance(w_l, float):
            w_l = max(1, round(w_l * n_timesteps)) if 0 < w_l <= 1 else round(w_l)
        return max(1, min(int(w_l), n_timesteps))
