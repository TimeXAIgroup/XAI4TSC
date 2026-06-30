"""
Random-baseline explainers in the frequency and time-frequency domains.

Sanity baselines for evaluation: they ignore the model and return uniform-random
relevance shaped like the transformed input. A faithful explainer should beat
these on perturbation metrics.
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
from .base import PerturbationExplainer

if TYPE_CHECKING:
    from ._types import Explanation


class RandomFrequencyExplainer(PerturbationExplainer):
    """
    Uniform-random baseline relevance in the frequency domain.

    Parameters
    ----------
    transform : DomainTransform or dict, optional
        Transform from time to frequency domain. Defaults to a full
        :class:`~xai4tsc.utils.fourier_transforms.FourierTransform`.
    seed : int, optional
        Seed for the random relevance, for reproducible draws.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.TIME_SERIES}
    explanation_domains: ClassVar[set[Domain]] = {Domain.FREQUENCY}

    def __init__(
        self,
        transform: DomainTransform | dict | None = None,
        seed: int | None = None,
    ) -> None:
        self.transform = resolve_transform(transform) or FourierTransform()
        self.prng = np.random.default_rng(seed)

    def explain(
        self,
        model: torch.nn.Module,
        exp: Explanation,
        device: str | torch.device,
        targets: list | None,
        **kwargs: object,
    ) -> np.ndarray:
        """Return uniform-random relevance shaped like the frequency transform."""
        frequencies = self.transform.forward(
            torch.as_tensor(exp.data, dtype=torch.float32)
        )
        return self.prng.uniform(size=tuple(frequencies.shape)).astype(np.float32)


class RandomTimeFreqExplainer(PerturbationExplainer):
    """
    Uniform-random baseline relevance in the time-frequency domain.

    Parameters
    ----------
    transform : DomainTransform or dict
        Transform from time to time-frequency domain (e.g. an
        :class:`~xai4tsc.utils.fourier_transforms.STFTransform`). Required — there
        is no sensible default window configuration.
    seed : int, optional
        Seed for the random relevance, for reproducible draws.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.TIME_SERIES}
    explanation_domains: ClassVar[set[Domain]] = {Domain.TIME_FREQUENCY}

    def __init__(
        self,
        transform: DomainTransform | dict | None = None,
        seed: int | None = None,
    ) -> None:
        self.transform = resolve_transform(transform)
        if self.transform is None:
            raise ValueError(
                "RandomTimeFreqExplainer requires a time-frequency transform "
                "(e.g. an STFTransform); none was supplied."
            )
        self.prng = np.random.default_rng(seed)

    def explain(
        self,
        model: torch.nn.Module,
        exp: Explanation,
        device: str | torch.device,
        targets: list | None,
        **kwargs: object,
    ) -> np.ndarray:
        """Return uniform-random relevance shaped like the time-frequency transform."""
        spectrograms = self.transform.forward(
            torch.as_tensor(exp.data, dtype=torch.float32)
        )
        return self.prng.uniform(size=tuple(spectrograms.shape)).astype(np.float32)
