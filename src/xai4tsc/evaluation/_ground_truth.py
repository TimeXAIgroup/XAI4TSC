"""
Ground-truth discriminative regions, and the masks derived from them.

A dataset that knows which part of a sample carries the class signal ships that
knowledge as per-sample metadata; the ground-truth localization metrics score an
attribution against it. :class:`GroundTruthRegion` is the in-memory form of one
such region and :meth:`GroundTruthRegion.from_mapping` is the only place that
knows what a dataset may put on disk, so the stored data never has to be rewritten
to match a change here.

A region is **one interval**: ``pos`` and ``length`` define it, and every other
field is a scope that broadcasts over it. Nothing is matched by index across
fields.

Every field is optional and an omitted one means "all": no ``channels`` is every
channel, and no interval is the whole series. A region must still localize
*something* — an interval, a frequency, or both — so an empty one is rejected
rather than read as "everywhere".

The time domain comes first: ``pos`` and ``length`` are all a time-domain dataset
has to provide, and they are what every localization metric reads. ``freqs`` and
``phases`` are the optional augmentation that only the frequency and
time-frequency metrics consume.

A frequency is read as cycles per series, which is the unit a coefficient grid is
indexed by. A dataset annotating in Hz instead records the ``sampling_rate`` its
frequencies are relative to - on the region, or once on the sample row - and
:meth:`GroundTruthRegion.cycles` converts. Getting that wrong is quiet and costly:
the mask lands on bins the class signal never occupies, and the metric then reports
chance however good the attribution is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..utils.fourier_transforms import DomainTransform

__all__ = [
    "GroundTruthRegion",
    "coefficient_mask",
    "regions_for_class",
    "regions_for_sample",
    "time_mask",
]

_SCALAR_ALIASES = {
    "channels": "channel",
    "classes": "class",
    "freqs": "freq",
    "phases": "phase",
}
"""Plural field -> the singular key a dataset may have written instead."""

_LENGTH_KEYS = ("len", "length")
"""Accepted spellings of the interval length: the stored key and the field name."""

_KNOWN_KEYS = frozenset(
    {
        "pos",
        "sampling_rate",
        *_LENGTH_KEYS,
        *_SCALAR_ALIASES,
        *_SCALAR_ALIASES.values(),
    }
)
"""Every key a stored region may carry.

Anything else is rejected rather than ignored. Silently dropping an unknown key
would make a misspelled ``len`` read as *no* interval, which this schema treats as
"the whole series" — a wrong mask rather than an error.
"""


def _as_int_tuple(value: object, field: str) -> tuple[int, ...]:
    """
    Coerce a scalar or sequence to a tuple of ints.

    Parameters
    ----------
    value : object
        A scalar, or a sequence of scalars.
    field : str
        Field name, used in the error message.

    Returns
    -------
    tuple of int
        The coerced values.

    Raises
    ------
    ValueError
        If *value* or any element is not convertible to ``int``.
    """
    items = (
        value if isinstance(value, Sequence) and not isinstance(value, str) else [value]
    )
    try:
        return tuple(int(v) for v in items)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ground-truth region field {field!r} is not integral: {value!r}"
        ) from exc


def _as_float_tuple(value: object, field: str) -> tuple[float, ...]:
    """
    Coerce a scalar or sequence to a tuple of floats.

    Parameters
    ----------
    value : object
        A scalar, or a sequence of scalars.
    field : str
        Field name, used in the error message.

    Returns
    -------
    tuple of float
        The coerced values.

    Raises
    ------
    ValueError
        If *value* or any element is not convertible to ``float``.
    """
    items = (
        value if isinstance(value, Sequence) and not isinstance(value, str) else [value]
    )
    try:
        return tuple(float(v) for v in items)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ground-truth region field {field!r} is not numeric: {value!r}"
        ) from exc


@dataclass(frozen=True)
class GroundTruthRegion:
    """
    One discriminative interval of a sample.

    ``pos`` and ``length`` define the interval; the remaining fields are scopes
    that apply to the whole of it. A scope left as ``None`` means "all" — every
    channel, or every class — rather than none.

    Attributes
    ----------
    pos : int
        Start of the interval, in samples.
    length : int or None
        Length of the interval, in samples; ``None`` runs to the end of the
        series, which is what a region localized only in frequency wants.
    channels : tuple of int or None
        Channels the region applies to; ``None`` means every channel.
    classes : tuple of int or None
        Classes the region discriminates; ``None`` means every class.
    freqs : tuple of float
        Frequency components. Read only by the frequency and time-frequency
        metrics; empty for a time-domain region. The unit depends on
        :attr:`sampling_rate`: without one they are cycles per series, with one
        they are Hz. See :meth:`cycles`.
    phases : tuple of float
        Phases, paired with :attr:`freqs` by index. Recorded for completeness;
        no metric reads them.
    sampling_rate : float or None
        Sampling rate in Hz that :attr:`freqs` is relative to. ``None`` means
        :attr:`freqs` is already in cycles per series, which is the unit a
        coefficient grid is indexed by, so nothing needs converting. A dataset
        that annotates in Hz - the natural unit for a physical signal - records
        the rate alongside its regions and :meth:`cycles` converts.
    """

    pos: int = 0
    length: int | None = None
    channels: tuple[int, ...] | None = None
    classes: tuple[int, ...] | None = None
    freqs: tuple[float, ...] = ()
    phases: tuple[float, ...] = ()
    sampling_rate: float | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping) -> GroundTruthRegion:
        """
        Build a region from a dataset's stored mapping.

        Accepts the plural field names and the singular ones a dataset may have
        written instead (``channel``, ``freq``, ``phase``), so stored metadata
        does not have to be rewritten. An absent scope becomes ``None`` ("all")
        and an absent sequence becomes empty.

        Parameters
        ----------
        raw : Mapping
            One stored region. Every field is optional, but the region must
            localize something: an interval (``pos`` / ``len``), a frequency
            (``freq`` / ``freqs``), or both. An unrecognised field is an error,
            not ignored.

        Returns
        -------
        GroundTruthRegion
            The parsed region.

        Raises
        ------
        ValueError
            If the region carries an unrecognised field, localizes nothing, or
            has a field that cannot be coerced.
            Malformed metadata raises rather than reading as no region at all,
            so a typo cannot be mistaken for a sample without ground truth.
        """
        unknown = sorted(set(raw) - _KNOWN_KEYS)
        if unknown:
            raise ValueError(
                f"ground-truth region has unrecognised field(s) {unknown}; "
                f"known fields are {sorted(_KNOWN_KEYS)}"
            )

        def pick(plural: str) -> object | None:
            """Return the plural field's value, falling back to its singular alias."""
            if plural in raw:
                return raw[plural]
            return raw.get(_SCALAR_ALIASES[plural])

        def length_of(mapping: Mapping) -> object | None:
            """Return the interval length under either accepted spelling."""
            for key in _LENGTH_KEYS:
                if mapping.get(key) is not None:
                    return mapping[key]
            return None

        channels = pick("channels")
        classes = pick("classes")
        freqs = pick("freqs")
        phases = pick("phases")
        rate = raw.get("sampling_rate")

        localizing = {"pos", *_LENGTH_KEYS, "freq", "freqs"} & set(raw)
        if not localizing:
            raise ValueError(
                "ground-truth region localizes nothing: it needs an interval "
                f"('pos'/'len') or a frequency ('freq'/'freqs'), got {dict(raw)!r}"
            )

        return cls(
            pos=_as_int_tuple(raw.get("pos", 0), "pos")[0],
            length=(
                None
                if length_of(raw) is None
                else _as_int_tuple(length_of(raw), "len")[0]
            ),
            channels=None if channels is None else _as_int_tuple(channels, "channels"),
            classes=None if classes is None else _as_int_tuple(classes, "classes"),
            freqs=() if freqs is None else _as_float_tuple(freqs, "freqs"),
            phases=() if phases is None else _as_float_tuple(phases, "phases"),
            sampling_rate=(
                None if rate is None else _as_float_tuple(rate, "sampling_rate")[0]
            ),
        )

    def cycles(self, signal_length: int) -> tuple[float, ...]:
        """
        Return :attr:`freqs` as cycles per series, converting from Hz if needed.

        A coefficient grid is indexed by cycles per series: rFFT bin ``k`` of a
        length-``T`` signal is the component completing ``k`` oscillations across
        those ``T`` samples, with no sampling rate involved. A frequency in Hz is
        a count per second instead, so it scales by the series duration
        ``signal_length / sampling_rate``: at 100 Hz over 500 samples the series
        lasts 5 s, and 7 Hz is 35 cycles per series.

        Parameters
        ----------
        signal_length : int
            Length ``T`` of the time-domain series.

        Returns
        -------
        tuple of float
            One value per entry of :attr:`freqs`, in cycles per series.
        """
        if self.sampling_rate is None:
            return self.freqs
        scale = signal_length / self.sampling_rate
        return tuple(freq * scale for freq in self.freqs)

    def interval(self, signal_length: int) -> tuple[int, int]:
        """
        Return the region's ``[start, stop)`` span, clipped to the series.

        Parameters
        ----------
        signal_length : int
            Length ``T`` of the series; also the extent of a region whose
            :attr:`length` is ``None``.

        Returns
        -------
        tuple of int
            ``(start, stop)``; ``stop <= start`` means the region is empty.
        """
        length = signal_length if self.length is None else self.length
        return max(0, self.pos), min(signal_length, self.pos + length)

    def applies_to_channel(self, channel: int, n_channels: int) -> bool:
        """
        Report whether this region covers *channel*.

        Parameters
        ----------
        channel : int
            Channel index to test.
        n_channels : int
            Number of channels in the sample, used to bound the "all" case.

        Returns
        -------
        bool
            ``True`` if the region applies to *channel*.
        """
        if self.channels is None:
            return 0 <= channel < n_channels
        return channel in self.channels


def regions_for_sample(sample_meta: object) -> list[GroundTruthRegion]:
    """
    Return the discriminative regions recorded for one sample.

    Distinguishes three outcomes, which is the point of going through here: a
    sample with regions, a sample that legitimately has none (an empty list), and
    metadata that is malformed (an exception). Reading malformed metadata as "no
    regions" would turn a typo into a score.

    Both stored shapes are accepted: a flat ``[region, ...]`` list, and the
    per-class ``{class_key: [region, ...]}`` mapping (keys may be ``int`` or
    ``str`` after a JSON round-trip). The per-class form is flattened, with the
    key recorded as the region's :attr:`~GroundTruthRegion.classes` scope when
    the region does not state one itself.

    A ``sampling_rate`` recorded on the sample row applies to every region of that
    sample, so a dataset annotating in Hz states the rate once rather than on each
    region. A region carrying its own rate keeps it.

    Parameters
    ----------
    sample_meta : object
        One sample's metadata mapping. Anything without a ``"ground_truth"`` key
        yields no regions.

    Returns
    -------
    list of GroundTruthRegion
        The sample's regions; empty if it has none.

    Raises
    ------
    ValueError
        If ``ground_truth`` is present but not a list or mapping of regions, or
        if any region is malformed.
    """
    if not isinstance(sample_meta, Mapping):
        return []
    if "ground_truth" not in sample_meta:
        return []

    gt = sample_meta["ground_truth"]
    if gt is None:
        return []

    if isinstance(gt, Mapping):
        regions: list[GroundTruthRegion] = []
        for class_key, region_list in gt.items():
            if not isinstance(region_list, Sequence) or isinstance(region_list, str):
                raise ValueError(
                    f"ground_truth[{class_key!r}] should be a list of regions, "
                    f"got {type(region_list).__name__}"
                )
            scope = _as_int_tuple(class_key, "classes")
            for raw in region_list:
                region = GroundTruthRegion.from_mapping(_require_mapping(raw))
                # The outer key names the class unless the region states its own.
                if region.classes is None:
                    region = replace(region, classes=scope)
                regions.append(region)
        return _with_sampling_rate(regions, sample_meta)

    if isinstance(gt, Sequence) and not isinstance(gt, str):
        return _with_sampling_rate(
            [GroundTruthRegion.from_mapping(_require_mapping(raw)) for raw in gt],
            sample_meta,
        )

    raise ValueError(
        "ground_truth should be a list of regions or a {class: [regions]} mapping, "
        f"got {type(gt).__name__}"
    )


def _with_sampling_rate(
    regions: list[GroundTruthRegion], sample_meta: Mapping
) -> list[GroundTruthRegion]:
    """
    Apply the sample row's ``sampling_rate`` to the regions that lack one.

    Parameters
    ----------
    regions : list of GroundTruthRegion
        The sample's parsed regions.
    sample_meta : Mapping
        The sample's metadata row, which may carry ``sampling_rate``.

    Returns
    -------
    list of GroundTruthRegion
        The regions, each with a sampling rate if one was available.
    """
    rate = sample_meta.get("sampling_rate")
    if rate is None:
        return regions
    rate = _as_float_tuple(rate, "sampling_rate")[0]
    return [
        region
        if region.sampling_rate is not None
        else replace(region, sampling_rate=rate)
        for region in regions
    ]


def _require_mapping(raw: object) -> Mapping:
    """
    Return *raw* if it is a mapping, else raise.

    Parameters
    ----------
    raw : object
        A candidate stored region.

    Returns
    -------
    Mapping
        The mapping, unchanged.

    Raises
    ------
    ValueError
        If *raw* is not a mapping.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(
            "ground-truth region should be a mapping, got "
            f"{type(raw).__name__}: {raw!r}"
        )
    return raw


def regions_for_class(
    regions: list[GroundTruthRegion], class_index: int | None
) -> list[GroundTruthRegion]:
    """
    Keep the regions that are discriminative for *class_index*.

    An attribution explains one class, so it should be scored against the evidence
    for *that* class. Where a sample carries regions for several classes — a
    multi-label dataset, where each label marks a different pattern — scoring every
    class against all of them measures the wrong thing, and makes a per-class
    evaluation produce the same mask for each class.

    Parameters
    ----------
    regions : list of GroundTruthRegion
        A sample's regions, from :func:`regions_for_sample`.
    class_index : int or None
        The class the attribution explains. ``None`` keeps everything, which is
        right when the producer did not record the target: the union is the best
        available answer, and for a single-label dataset it is also the correct one.

    Returns
    -------
    list of GroundTruthRegion
        The regions scoped to *class_index*, plus any region that declares no scope
        (``classes is None`` means every class).
    """
    if class_index is None:
        return regions
    return [
        region
        for region in regions
        if region.classes is None or int(class_index) in region.classes
    ]


def time_mask(
    regions: list[GroundTruthRegion], n_channels: int, signal_length: int
) -> np.ndarray:
    """
    Build a boolean ground-truth mask on the time grid.

    Parameters
    ----------
    regions : list of GroundTruthRegion
        Regions for one sample, from :func:`regions_for_sample`.
    n_channels : int
        Number of channels ``C`` in the sample.
    signal_length : int
        Length ``T`` of the series.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(C, T)``; ``True`` inside a region's interval.
        All-``False`` when *regions* is empty — a caller that needs to tell an
        empty mask from a populated one must check, since metric backends do not.
    """
    mask = np.zeros((n_channels, signal_length), dtype=bool)
    for region in regions:
        start, stop = region.interval(signal_length)
        if stop <= start:
            continue
        for channel in range(n_channels):
            if region.applies_to_channel(channel, n_channels):
                mask[channel, start:stop] = True
    return mask


def coefficient_mask(
    regions: list[GroundTruthRegion],
    grid_shape: tuple[int, ...],
    signal_length: int,
    transform: DomainTransform | None,
) -> np.ndarray:
    """
    Build a boolean ground-truth mask in an explanation's coefficient grid.

    Parameters
    ----------
    regions : list of GroundTruthRegion
        Regions for one sample, from :func:`regions_for_sample`.
    grid_shape : tuple of int
        Shape of one sample's attribution: ``(C, n_freq)`` for a frequency
        explanation or ``(C, n_freq, n_time)`` for a time-frequency one.
    signal_length : int
        Length ``T`` of the time-domain series (sets the frequency scale, and the
        duration a region's frequency in Hz is converted against).
    transform : DomainTransform or None
        The explanation's transform; ``n_fft`` / ``hop_length`` are read from it
        for the STFT grid geometry when present.

    Returns
    -------
    np.ndarray
        Boolean array of shape *grid_shape*; ``True`` at discriminative cells.
    """
    mask = np.zeros(grid_shape, dtype=bool)
    if not regions:
        return mask

    n_channels = grid_shape[0]
    n_freq = grid_shape[1]
    is_tf = len(grid_shape) == 3
    n_time = grid_shape[2] if is_tf else None
    n_fft = getattr(transform, "n_fft", None)
    hop = getattr(transform, "hop_length", None)

    for region in regions:
        channels = [
            c for c in range(n_channels) if region.applies_to_channel(c, n_channels)
        ]
        if not channels:
            continue
        # A region with no frequency recorded is not localized in frequency, so it
        # marks the whole column; the time extent still applies.
        cycles = region.cycles(signal_length) or (0.0,)
        for freq in cycles:
            # An rFFT bin over the full series *is* the cycles-per-series count;
            # an STFT bin rescales cycles/sample (freq/T) onto the n_fft grid.
            scale = n_fft / signal_length if (is_tf and n_fft) else 1.0
            f_bin = min(max(round(freq * scale), 0), n_freq - 1)
            for channel in channels:
                if is_tf:
                    if hop:
                        # torch.stft uses center=True: frame f is centred at f*hop.
                        start, stop = region.interval(signal_length)
                        f0 = max(0, start // hop)
                        f1 = min(n_time, stop // hop + 1)
                        mask[channel, f_bin, f0:f1] = True
                    else:
                        mask[channel, f_bin, :] = True
                else:
                    mask[channel, f_bin] = True
    return mask
