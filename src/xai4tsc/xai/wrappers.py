"""Wrapper explainers that extend a base method (e.g. SIGN)."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import GradientExplainer, WrapperExplainer

if TYPE_CHECKING:
    import torch

    from ._types import Explanation
    from .base import ExplainerBase

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
    overriding ``True`` on a copy of the base with a warning; gradient-only methods
    (Guided Backpropagation, Deconvolution) already satisfy this. A non-gradient
    base cannot produce the SIGN variant and is flagged with a warning.

    LRP-SIGN is *not* expressible this way — for LRP, SIGN is an input-layer
    relevance rule applied during the backward pass. That variant would be a
    separate :class:`~xai4tsc.xai.base.WrapperExplainer` subclass and is out of
    scope here.

    Parameters
    ----------
    base : ExplainerBase
        The gradient base explainer to extend, e.g.
        ``IntegratedGradientsExplainer(multiply_by_inputs=False)``.
    mu : float
        Sign threshold. Inputs ``< mu`` map to ``-1``, the rest to ``+1``.
    """

    def __init__(self, base: ExplainerBase, mu: float = 0.0) -> None:
        # SIGN replaces the "x input" weighting with "x sign(input)", so the base
        # must emit the raw gradient. The override goes on a copy so the caller's
        # explainer keeps its setting.
        if getattr(base, "_multiply_by_inputs", False):
            logger.warning(
                "SIGN: overriding multiply_by_inputs=True to False on base '%s' "
                "so sign(input) replaces the input weighting.",
                type(base).__name__,
            )
            base = copy.copy(base)
            base._multiply_by_inputs = False
        super().__init__(base)
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
        relevance = self._run_base(model, exp, device, targets, **kwargs)
        sign = np.where(exp.data < self.mu, -1.0, 1.0).astype(relevance.dtype)
        return relevance * sign
