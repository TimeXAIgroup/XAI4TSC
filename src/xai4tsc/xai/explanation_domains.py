"""
Explanation-space wrappers: lift a time-domain attributor into freq / TF space.

Wraps any base feature-attribution method (e.g. Integrated Gradients, Guided
Backpropagation) and maps its time-domain relevance into the frequency or
time-frequency domain via a :class:`~xai4tsc.utils.fourier_transforms.DomainTransform`.

Reference
---------
S. Rezaei and X. Liu, "Explanation Space: A New Perspective into Time Series
Interpretability." arXiv, Sep. 2024. doi: 10.48550/ARXIV.2409.01354.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from ..utils.fourier_transforms import (
    DomainTransform,
    FourierTransform,
    resolve_transform,
)
from ._types import DataType, Domain
from .base import WrapperExplainer

if TYPE_CHECKING:
    from ._types import Explanation


class FrequencyExplainer(WrapperExplainer):
    """
    Map a base method's relevance into the frequency domain.

    Runs the wrapped time-domain attributor, then transforms its relevance with
    a Fourier transform. The base is configured exactly as if invoked by name.

    Parameters
    ----------
    base : dict or str
        The base explainer to wrap (method name or config dict). Defaults to
        ``"integrated_gradients"``.
    transform : DomainTransform or dict, optional
        Transform to the frequency domain. Defaults to a full
        :class:`~xai4tsc.utils.fourier_transforms.FourierTransform`.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.TIME_SERIES}
    explanation_domains: ClassVar[set[Domain]] = {Domain.FREQUENCY}

    def __init__(
        self,
        base: dict | str = "integrated_gradients",
        transform: DomainTransform | dict | None = None,
    ) -> None:
        super().__init__(base)
        # WrapperExplainer.__init__ copies the base's {TIME} onto the instance;
        # this wrapper *changes* the domain, so override after super().__init__().
        self.data_applicability = {DataType.TIME_SERIES}
        self.explanation_domains = {Domain.FREQUENCY}
        self.transform = resolve_transform(transform) or FourierTransform()

    def explain(
        self,
        model: torch.nn.Module,
        exp: Explanation,
        device: str | torch.device,
        targets: list | None,
        **kwargs: object,
    ) -> np.ndarray:
        """Run the base method, then map its relevance to the frequency domain."""
        relevance_time = self._run_base(model, exp, device, targets)
        relevance_freq = self.transform.forward(
            torch.as_tensor(relevance_time, dtype=torch.float32)
        )
        return relevance_freq.cpu().numpy()


class TimeFrequencyExplainer(WrapperExplainer):
    """
    Map a base method's relevance into the time-frequency domain.

    Runs the wrapped time-domain attributor, then transforms its relevance with
    a short-time Fourier transform.

    Parameters
    ----------
    base : dict or str
        The base explainer to wrap (method name or config dict). Defaults to
        ``"guided_backpropagation"``.
    transform : DomainTransform or dict
        Transform to the time-frequency domain (e.g. an
        :class:`~xai4tsc.utils.fourier_transforms.STFTransform`). Required — there
        is no sensible default window configuration.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.TIME_SERIES}
    explanation_domains: ClassVar[set[Domain]] = {Domain.TIME_FREQUENCY}

    def __init__(
        self,
        base: dict | str = "guided_backpropagation",
        transform: DomainTransform | dict | None = None,
    ) -> None:
        super().__init__(base)
        self.data_applicability = {DataType.TIME_SERIES}
        self.explanation_domains = {Domain.TIME_FREQUENCY}
        self.transform = resolve_transform(transform)
        if self.transform is None:
            raise ValueError(
                "TimeFrequencyExplainer requires a time-frequency transform "
                "(e.g. an STFTransform); none was supplied."
            )

    def explain(
        self,
        model: torch.nn.Module,
        exp: Explanation,
        device: str | torch.device,
        targets: list | None,
        **kwargs: object,
    ) -> np.ndarray:
        """Run the base method, then map its relevance to time-frequency space."""
        relevance_time = self._run_base(model, exp, device, targets)
        relevance_tf = self.transform.forward(
            torch.as_tensor(relevance_time, dtype=torch.float32)
        )
        return relevance_tf.cpu().numpy()
