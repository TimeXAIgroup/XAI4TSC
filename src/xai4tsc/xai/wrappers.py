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
    base cannot produce the SIGN variant and is refused.

    LRP-SIGN is *not* expressible this way — for LRP, SIGN is an input-layer
    relevance rule applied during the backward pass. That variant would be a
    separate :class:`~xai4tsc.xai.base.WrapperExplainer` subclass and is out of
    scope here.

    Parameters
    ----------
    base : GradientExplainer
        The gradient base explainer to extend, e.g.
        ``IntegratedGradientsExplainer(multiply_by_inputs=False)``.
    mu : float
        Sign threshold. Inputs ``< mu`` map to ``-1``, the rest to ``+1``.
    """

    def __init__(self, base: GradientExplainer, mu: float = 0.0) -> None:
        """
        Wrap *base*, switching off its input weighting on a copy if it has one.

        Raises
        ------
        TypeError
            If *base* is not a :class:`~xai4tsc.xai.base.GradientExplainer`.
        """
        if not isinstance(base, GradientExplainer):
            raise TypeError(
                "SIGN is defined for gradient-based methods; base "
                f"'{type(base).__name__}' is not a GradientExplainer."
            )
        # SIGN replaces the "x input" weighting with "x sign(input)", so the base
        # must emit the raw gradient. The override goes on a copy so the caller's
        # explainer keeps its setting.
        if base.multiply_by_inputs:
            logger.warning(
                "SIGN: overriding multiply_by_inputs=True to False on base '%s' "
                "so sign(input) replaces the input weighting.",
                type(base).__name__,
            )
            base = copy.copy(base)
            base.multiply_by_inputs = False
        super().__init__(base)
        self.mu = mu

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
