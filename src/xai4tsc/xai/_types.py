"""``Explanation`` dataclass plus the ``DataType`` and ``Domain`` enums."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

if TYPE_CHECKING:
    from xai4tsc.utils.fourier_transforms import DomainTransform


class DataType(Enum):
    """
    Data domains an explainer is applicable to.

    Mirrors the role of Quantus' ``DataType`` enum: each explainer declares a
    ``data_applicability`` set built from these members (see
    :attr:`xai4tsc.xai.ExplainerBase.data_applicability`).
    """

    AGNOSTIC = "Agnostic"
    """Domain-agnostic — applies to any input (e.g. images, tabular, time series)."""

    TIME_SERIES = "Time Series"
    """Specific to time series classification."""


class Domain(Enum):
    """
    Signal domain a piece of data or an explanation lives in.

    A distinct axis from :class:`DataType`: ``DataType`` says *what kind of
    input* an explainer applies to, while ``Domain`` says *which signal domain*
    a concrete array is expressed in. Used by the :class:`Explanation` fields
    ``data_domain`` / ``explanation_domain`` and by the explainer capability set
    :attr:`xai4tsc.xai.ExplainerBase.explanation_domains`.
    """

    TIME = "Time"
    """Time domain (the raw series)."""

    FREQUENCY = "Frequency"
    """Frequency domain (e.g. via a Fourier transform)."""

    TIME_FREQUENCY = "Time-Frequency"
    """Time-frequency domain (e.g. via a short-time Fourier transform)."""


@dataclass
class Explanation:
    """
    A standard container for explanation results.

    Attributes
    ----------
    explainer : str
        Name of the explainer method that produced this explanation.
    explanation_type : str
        Category of explanation, e.g. ``"feature_attribution"``.
    exp_values : np.ndarray
        Raw explanation values; same shape as ``data``.
    data : np.ndarray
        Original input samples that were explained, shape ``(N, C, T)``.
    labels : np.ndarray
        Class labels for each sample in ``data``.
    indices : np.ndarray
        Indices into the test set from which ``data`` was drawn.
    encoder : LabelEncoder or OneHotEncoder or OrdinalEncoder
        Label encoder used during dataset preparation.
    meta : dict
        Any extra metadata returned by the explainer.
    data_domain : Domain
        Signal domain of ``data``. Defaults to :attr:`Domain.TIME`.
    explanation_domain : Domain
        Signal domain of ``exp_values``. Defaults to :attr:`Domain.TIME`. May
        differ from ``data_domain`` (e.g. a frequency explainer consumes
        time-domain data and emits frequency-domain relevance).
    transform : DomainTransform, optional
        The stateful transform mapping between the time domain and
        ``data_domain`` / ``explanation_domain``. ``None`` for pure time-domain
        explanations. Freq/TF views of ``data`` are *derived* on demand via
        ``transform.forward(data)`` rather than stored.
    metadata : list, optional
        Per-sample dataset metadata for the explained samples, aligned to
        ``exp_values`` / ``indices`` (one entry per row). Carries ground-truth
        localization (e.g. ``{"ground_truth": {class: [regions]}, ...}``) from
        datasets that provide it; consumed by ground-truth localization metrics
        such as ``TimeFrequencyAUC``. ``None`` when the dataset has no metadata.
    """

    explainer: str
    explanation_type: str
    exp_values: np.ndarray
    data: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    encoder: LabelEncoder | OneHotEncoder | OrdinalEncoder
    meta: dict
    data_domain: Domain = Domain.TIME
    explanation_domain: Domain = Domain.TIME
    transform: DomainTransform | None = None
    metadata: list | None = None
