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

from .._random import DEFAULT_SEED, resolve_rng
from ..utils.fourier_transforms import (
    DomainTransform,
    FourierTransform,
    check_transform,
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
    transform : DomainTransform, optional
        Transform from time to frequency domain. Defaults to a full
        :class:`~xai4tsc.utils.fourier_transforms.FourierTransform`.
    seed : int or None
        Seed for the relevance draw, :data:`~xai4tsc.DEFAULT_SEED` by default.
        A local generator is built from it on every :meth:`explain` call, so
        repeated explanations of one input agree. ``None`` draws from OS
        entropy and is not reproducible.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.TIME_SERIES}
    explanation_domains: ClassVar[set[Domain]] = {Domain.FREQUENCY}
    explains_in_data_domain: ClassVar[bool] = False
    accepts_data_domains: ClassVar[set[Domain]] = {Domain.TIME}

    def __init__(
        self,
        transform: DomainTransform | None = None,
        seed: int | None = DEFAULT_SEED,
    ) -> None:
        self.transform = check_transform(transform) or FourierTransform()
        self.seed = seed

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
        prng = resolve_rng(self.seed)
        return prng.uniform(size=tuple(frequencies.shape)).astype(np.float32)


class RandomTimeFreqExplainer(PerturbationExplainer):
    """
    Uniform-random baseline relevance in the time-frequency domain.

    Parameters
    ----------
    transform : DomainTransform
        Transform from time to time-frequency domain (e.g. an
        :class:`~xai4tsc.utils.fourier_transforms.STFTransform`). Required — there
        is no sensible default window configuration.
    seed : int or None
        Seed for the relevance draw, :data:`~xai4tsc.DEFAULT_SEED` by default.
        A local generator is built from it on every :meth:`explain` call, so
        repeated explanations of one input agree. ``None`` draws from OS
        entropy and is not reproducible.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.TIME_SERIES}
    explanation_domains: ClassVar[set[Domain]] = {Domain.TIME_FREQUENCY}
    explains_in_data_domain: ClassVar[bool] = False
    accepts_data_domains: ClassVar[set[Domain]] = {Domain.TIME}

    def __init__(
        self,
        transform: DomainTransform | None = None,
        seed: int | None = DEFAULT_SEED,
    ) -> None:
        self.transform = check_transform(transform)
        if self.transform is None:
            raise ValueError(
                "RandomTimeFreqExplainer requires a time-frequency transform "
                "(e.g. an STFTransform); none was supplied."
            )
        self.seed = seed

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
        prng = resolve_rng(self.seed)
        return prng.uniform(size=tuple(spectrograms.shape)).astype(np.float32)
