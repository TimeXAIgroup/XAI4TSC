"""
The ``DataType`` and ``Domain`` enums, plus the data-domain string spellings.

Datasets declare which signal domain their arrays are expressed in, explainers declare
which domains they consume and produce, and metrics declare which they can score.
It lives in a dependency-free module so every layer can import it -
:mod:`xai4tsc.xai._types` re-exports it, which is the import path the rest of the
package and the docs use.
"""

from enum import Enum


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
    ``data_domain`` / ``explanation_domain``, by the explainer capability set
    :attr:`xai4tsc.xai.ExplainerBase.explanation_domains`, and by
    :attr:`xai4tsc.data.DatasetBase.data_domain`.
    """

    TIME = "Time"
    """Time domain (the raw series)."""

    FREQUENCY = "Frequency"
    """Frequency domain (e.g. via a Fourier transform)."""

    TIME_FREQUENCY = "Time-Frequency"
    """Time-frequency domain (e.g. via a short-time Fourier transform)."""


DATA_DOMAINS: dict[str, Domain] = {
    "time": Domain.TIME,
    "frequency": Domain.FREQUENCY,
    "time_frequency": Domain.TIME_FREQUENCY,
}
"""String spellings of the signal domain a dataset's arrays are expressed in."""


def resolve_data_domain(value: "str | Domain") -> Domain:
    """
    Resolve a configured data-domain name to its :class:`Domain`.

    Parameters
    ----------
    value : str or Domain
        A key of :data:`DATA_DOMAINS` (``"time"``, ``"frequency"``,
        ``"time_frequency"``), or an already-resolved member.

    Returns
    -------
    Domain
        The signal domain the arrays are expressed in.

    Raises
    ------
    ValueError
        If *value* is not a known data-domain name.
    """
    if isinstance(value, Domain):
        return value
    try:
        return DATA_DOMAINS[value]
    except KeyError:
        raise ValueError(
            f"Unknown data_domain '{value}'. "
            f"Choose from: {', '.join(repr(k) for k in DATA_DOMAINS)}."
        ) from None
