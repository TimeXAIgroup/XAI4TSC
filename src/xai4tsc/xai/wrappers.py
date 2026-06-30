"""Wrapper explainers that extend a base method (e.g. SIGN)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import GradientExplainer, WrapperExplainer

if TYPE_CHECKING:
    import torch

    from ._types import Explanation

logger = logging.getLogger(__name__)


class SignExplainer(WrapperExplainer):
    """
    SIGN extension of a gradient-based attribution method.

    Implements the SIGN rule of Gumpfer et al. (Information Fusion 2023,
    `doi:10.1016/j.inffus.2023.101883`): replace a gradient method's
    conventional "x input" weighting with the binarized sign of the input,
    ``raw_gradient * s_mu(input)`` where ``s_mu(x) = -1 if x < mu else +1``.

    For the gradient-attribution family this is exactly the paper's SIGN variant.
    The sign step is the *final* operation of these methods, so applying it to
    the raw-gradient output is mathematically identical to an in-pass rule.

    Notes
    -----
    **Correctness condition.** The base must emit a *raw gradient-space* map,
    i.e. it must not already multiply by the input. SIGN therefore *enforces*
    ``multiply_by_inputs=False`` on the base (Integrated Gradients, DeepLIFT),
    overriding a user-supplied ``True`` with a warning; gradient-only methods
    (Guided Backpropagation, Deconvolution) already satisfy this. A non-gradient
    base cannot produce the SIGN variant and is flagged with a warning.

    LRP-SIGN is *not* expressible this way — for LRP, SIGN is an input-layer
    relevance rule applied during the backward pass. That variant would be a
    separate :class:`~xai4tsc.xai.base.WrapperExplainer` subclass and is out of
    scope here.

    Parameters
    ----------
    base : dict or str
        The gradient base explainer to extend. Either a bare method name or a
        config dict with a ``"method"`` key plus that method's hyperparameters
        (e.g. ``{"method": "integrated_gradients", "multiply_by_inputs": False}``).
    mu : float
        Sign threshold. Inputs ``< mu`` map to ``-1``, the rest to ``+1``.
    """

    def __init__(self, base: dict | str, mu: float = 0.0) -> None:
        # SIGN replaces the "x input" weighting with "x sign(input)", so the base
        # must emit the raw gradient. Enforce multiply_by_inputs=False on the base
        # config; build_explainer drops the key for methods that do not accept it
        # (Guided Backpropagation, Deconvolution, Occlusion).
        base_cfg = {"method": base} if isinstance(base, str) else dict(base)
        if base_cfg.get("multiply_by_inputs"):
            logger.warning(
                "SIGN: overriding multiply_by_inputs=True to False on base '%s' "
                "so sign(input) replaces the input weighting.",
                base_cfg["method"],
            )
        base_cfg["multiply_by_inputs"] = False
        super().__init__(base_cfg)
        self.mu = mu
        # SIGN's identity only holds for gradient-space bases. Warn (rather than
        # silently produce a non-SIGN map) for an incompatible base.
        if not isinstance(self._base, GradientExplainer):
            logger.warning(
                "SIGN is defined for gradient-based methods; base '%s' is not a "
                "GradientExplainer, so the result is not the paper's SIGN variant.",
                type(self._base).__name__,
            )

    def explain(
        self,
        model: torch.nn.Module,
        exp: Explanation,
        device: str | torch.device,
        targets: list | None,
        **kwargs: object,
    ) -> np.ndarray:
        """Run the base method, then weight it by ``sign(input - mu)``."""
        relevance = self._run_base(model, exp, device, targets)
        sign = np.where(exp.data < self.mu, -1.0, 1.0).astype(relevance.dtype)
        return relevance * sign
