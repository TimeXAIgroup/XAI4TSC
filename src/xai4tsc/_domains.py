"""
The ``DataType`` and ``Domain`` enums.

Datasets declare which signal domain their arrays are expressed in, explainers declare
which domains they consume and produce, and metrics declare which they can score.
It lives in a dependency-free module so every layer can import it -
:mod:`xai4tsc.xai._types` re-exports it, which is the import path the rest of the
package and the docs use.
"""

from enum import Enum, StrEnum


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


class Domain(StrEnum):
    """
    Signal domain a piece of data or an explanation lives in.

    A distinct axis from :class:`DataType`: ``DataType`` says *what kind of
    input* an explainer applies to, while ``Domain`` says *which signal domain*
    a concrete array is expressed in. Used by the :class:`Explanation` fields
    ``data_domain`` / ``explanation_domain``, by the explainer capability set
    :attr:`xai4tsc.xai.ExplainerBase.explanation_domains`, and by
    :attr:`xai4tsc.data.DatasetBase.data_domain`.

    A string enum, so ``Domain("time")`` and ``Domain.TIME`` name the same member
    and a parameter typed ``Domain`` also takes the plain string.
    """

    TIME = "time"
    """Time domain (the raw series)."""

    FREQUENCY = "frequency"
    """Frequency domain (e.g. via a Fourier transform)."""

    TIME_FREQUENCY = "time_frequency"
    """Time-frequency domain (e.g. via a short-time Fourier transform)."""

    @classmethod
    def _missing_(cls, value: object) -> None:
        """
        Reject an unknown domain, naming the allowed values.

        Parameters
        ----------
        value : object
            The value that matched no member.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(
            f"Unknown domain {value!r}. Choose from: "
            f"{', '.join(repr(m.value) for m in cls)}."
        )
