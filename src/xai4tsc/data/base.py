"""``DatasetBase`` ABC: splitting, label encoding, saving, loading pre-split data."""

import inspect
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .._domains import Domain
from .._random import DEFAULT_SEED, resolve_rng
from ..utils.utils import _check_option
from .data_loaders import load_json, load_metadata

logger = logging.getLogger(__name__)

SampleStrategy = Literal["random", "first", "last", "stratified"]
"""How :class:`DatasetBase` selects the samples it keeps under ``max_samples``."""

SeriesPosition = Literal["first", "last"]
"""Which end of a long series :class:`DatasetBase` keeps under ``max_series_length``."""

Layout = Literal["auto", "bct", "btc"]
"""Axis order of a stored array, read by :func:`_ensure_bct`."""


# ── Module-level helpers ───────────────────────────────────────────────────────


def _ensure_bct(data: np.ndarray, layout: Layout = "auto") -> np.ndarray:
    """
    Return *data* in the framework's canonical ``(B, C, T)`` order.

    The framework always expects ``(batch, channels, timesteps)``. Local user data
    is often saved as ``(B, T, C)``, so ``layout="auto"`` guesses from the shape -
    but a shape alone cannot distinguish ``(B, T, C)`` from a ``(B, C, T)`` dataset
    that genuinely has more channels than timesteps (a spectrogram, say), and the
    guess corrupts the latter. ``"bct"`` and ``"btc"`` state the order instead of
    guessing. **Never call this on UCR/UEA data from sktime**, which is already
    ``(B, C, T)``.

    Parameters
    ----------
    data : np.ndarray
        Loaded data array.
    layout : str
        ``"auto"`` (default) transposes when axis 1 exceeds axis 2, with a warning;
        ``"bct"`` trusts the data as-is; ``"btc"`` always transposes a 3-D array.

    Returns
    -------
    np.ndarray
        Data in ``(B, C, T)`` order.

    Raises
    ------
    ValueError
        If *layout* is not one of ``"auto"``, ``"bct"`` or ``"btc"``.
    """
    _check_option("layout", layout, Layout)
    if data.ndim != 3 or layout == "bct":
        return data
    if layout == "btc":
        return data.transpose(0, 2, 1)
    if data.shape[1] > data.shape[2]:
        logger.warning(
            "Data shape %s has more values on axis 1 than axis 2, so it is being "
            "read as (B, T, C) and transposed to (B, C, T). If it is already "
            "(B, C, T) with more channels than timesteps, this corrupts it: set "
            "layout='bct' to keep it as-is, or layout='btc' to confirm the "
            "transpose and silence this warning.",
            data.shape,
        )
        return data.transpose(0, 2, 1)
    return data


def _impute_data(data: np.ndarray) -> np.ndarray:
    """
    Replace NaN values with per-channel mean across all samples and timesteps.

    Parameters
    ----------
    data : np.ndarray
        Array of shape ``(B, C, T)`` that may contain NaN values.

    Returns
    -------
    np.ndarray
        Array of the same shape with NaN values replaced, dtype ``float32``.
    """
    b, c, t = data.shape
    imp = SimpleImputer(strategy="mean")
    imputed = imp.fit_transform(data.reshape(b * c, t))
    return imputed.reshape(b, c, t).astype(np.float32)


def _multi_hot_rows(y: np.ndarray, encoder_name: str) -> np.ndarray:
    """
    Return *y* as a 2-D multi-hot array, raising if it is not one.

    Shared by the two encoders that consume multi-hot rows, so both report the
    same shape complaint.

    Parameters
    ----------
    y : np.ndarray
        Labels as given.
    encoder_name : str
        Name used in the error message.

    Returns
    -------
    np.ndarray
        *y* as an array, guaranteed 2-D.

    Raises
    ------
    ValueError
        If *y* is not 2-D.
    """
    arr = np.asarray(y)
    if arr.ndim != 2:
        raise ValueError(
            f"{encoder_name} expects 2-D multi-hot labels (n, n_attrs); "
            f"got shape {arr.shape}."
        )
    return arr


class PowersetLabelEncoder:
    """
    Collapse multi-hot label rows to single class indices, preserving the combos.

    Multi-label datasets emit a multi-hot label row per sample (e.g.
    ``[0, 1, 1]`` = attributes 1 and 2 present). Training with
    :class:`~torch.nn.CrossEntropyLoss` needs 1-D integer class indices, so each
    distinct combination is mapped to one class index - the label-powerset
    reduction. The combination itself is kept as the **class name**
    (``classes_``) so predictions can be decoded back to multi-hot later;
    :func:`_split_dataset` / :meth:`DatasetBase.load_saved_splits` log the
    index→name mapping.

    The cost is that the number of outputs grows as ``2**L`` while the
    supervision per output shrinks, and that an attribution explains a
    *combination* - whose meaning includes the absence of the labels it omits -
    rather than one label. :class:`MultiHotLabelEncoder` is the alternative that
    keeps one output per label.

    Implements the small subset of the sklearn encoder interface the framework
    uses (``fit`` / ``transform`` / ``fit_transform`` / ``inverse_transform`` and
    the ``classes_`` attribute), so it slots into the same code paths as
    :class:`~sklearn.preprocessing.LabelEncoder`.
    """

    def fit(self, y: np.ndarray) -> "PowersetLabelEncoder":
        """Record the sorted unique multi-hot rows as ``classes_``."""
        arr = _multi_hot_rows(y, type(self).__name__)
        self.classes_ = np.unique(arr, axis=0)
        self._index = {
            tuple(int(v) for v in row): i for i, row in enumerate(self.classes_)
        }
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Map each multi-hot row to its class index (1-D int64)."""
        arr = np.asarray(y)
        return np.array(
            [self._index[tuple(int(v) for v in row)] for row in arr], dtype=np.int64
        )

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit then transform in one call."""
        return self.fit(y).transform(y)

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        """Map class indices back to their multi-hot rows."""
        return self.classes_[np.asarray(indices, dtype=np.int64)]


class MultiHotLabelEncoder:
    """
    Keep multi-hot label rows as one target column per model output.

    A multi-label dataset emits a multi-hot row per sample (e.g. ``[0, 1, 1]`` =
    labels 1 and 2 present), which already *is* the indicator matrix a per-output
    head is trained against with :class:`~torch.nn.BCEWithLogitsLoss`. So there
    is nothing to compute here: the targets are the stored labels. The class
    exists to supply the encoder contract the rest of the framework reads off
    it - ``fit`` / ``transform``, and the ``classes_`` that names each output.

    scikit-learn has no target encoder for this input.
    :class:`~sklearn.preprocessing.LabelBinarizer` takes 1-D labels, and
    :class:`~sklearn.preprocessing.MultiLabelBinarizer` takes label *sets*:
    fitting it on multi-hot rows does not raise, it reads ``[0, 1, 1]`` as the
    set ``{0, 1}`` and yields ``classes_ = [0 1]``, a two-column target that
    would size a model at two outputs. Pass ``MultiLabelBinarizer`` itself when
    the labels really are sets.

    Parameters
    ----------
    classes : array-like, optional
        The label identities in column order, with scikit-learn's meaning:
        ``classes_`` becomes a copy of this. The columns of a multi-hot row are
        positional, so unlike
        :class:`~sklearn.preprocessing.MultiLabelBinarizer` this cannot
        *determine* how many there are - it has to agree with the row width, and
        a mismatch raises rather than attaching the wrong name to an output.
        Omitted, ``classes_`` is ``arange(n_outputs)``.

    See Also
    --------
    PowersetLabelEncoder : collapses the same rows to one index per combination.
    """

    def __init__(self, classes: object | None = None) -> None:
        self.classes = None if classes is None else list(classes)

    def fit(self, y: np.ndarray) -> "MultiHotLabelEncoder":
        """Record the label identity of each column as ``classes_``."""
        width = _multi_hot_rows(y, type(self).__name__).shape[1]
        if self.classes is None:
            self.classes_ = np.arange(width)
        elif len(self.classes) != width:
            raise ValueError(
                f"{type(self).__name__} was given {len(self.classes)} class name(s) "
                f"{self.classes} for a {width}-column multi-hot target. The columns "
                "are positional, so a mismatched list cannot be reordered to fit - "
                "it would label an output with another output's name."
            )
        else:
            self.classes_ = np.asarray(self.classes)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Return the multi-hot rows as an int64 target, one column per output."""
        arr = _multi_hot_rows(y, type(self).__name__)
        if arr.shape[1] != len(self.classes_):
            raise ValueError(
                f"{type(self).__name__} was fitted on {len(self.classes_)} outputs "
                f"but given a {arr.shape[1]}-column target."
            )
        return arr.astype(np.int64, copy=False)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit then transform in one call."""
        return self.fit(y).transform(y)

    def inverse_transform(self, rows: np.ndarray) -> np.ndarray:
        """Return the multi-hot rows unchanged; they are their own decoding."""
        return np.asarray(rows, dtype=np.int64)


def _resolve_encoder(encoder: object | None) -> object:
    """
    Return the label encoder to use, defaulting to scikit-learn's.

    Parameters
    ----------
    encoder : object or None
        A target encoder supplied by the caller, or ``None`` for the default.

    Returns
    -------
    object
        *encoder* itself, or a fresh :class:`~sklearn.preprocessing.LabelEncoder`.
    """
    if encoder is None:
        return LabelEncoder()
    _reject_feature_encoder(encoder)
    return encoder


def _reject_feature_encoder(encoder: object) -> None:
    """
    Raise when a scikit-learn *feature* encoder was passed as a label encoder.

    Labels are a target, so the encoder belongs to scikit-learn's target family
    (``LabelEncoder``, ``LabelBinarizer``, ``MultiLabelBinarizer``), which takes
    labels as given and exposes ``classes_``. A feature encoder encodes columns of
    ``X``: in this slot it misreads rather than fails, so it is refused here.

    Parameters
    ----------
    encoder : object
        The encoder a caller supplied.

    Raises
    ------
    TypeError
        If *encoder* is a scikit-learn feature encoder.
    """
    fit = getattr(encoder, "fit", None)
    first = ""
    if fit is not None:
        try:
            params = list(inspect.signature(fit).parameters)
        except (TypeError, ValueError):  # a callable without an inspectable signature
            params = []
        first = params[0] if params else ""
    if first == "X" or hasattr(encoder, "categories_"):
        raise TypeError(
            f"{type(encoder).__name__} is a scikit-learn feature encoder (it encodes "
            "columns of X and exposes 'categories_'), but labels are a target. Pass a "
            "target encoder instead: LabelEncoder for class indices, LabelBinarizer "
            "or MultiLabelBinarizer for one column per output, MultiHotLabelEncoder "
            "to keep multi-hot rows as they are, or PowersetLabelEncoder for the "
            "label-powerset reduction."
        )


def class_names(encoder: object) -> list[str]:
    """
    Return the display name of each class, ordered by class index.

    Parameters
    ----------
    encoder : object
        A fitted label encoder.

    Returns
    -------
    list of str
        One name per class index; empty when the encoder exposes no classes.
    """
    return list(_class_name_map(encoder).values())


def decode_labels(encoder: object, indices: np.ndarray) -> np.ndarray:
    """
    Map class indices back to their original label values.

    The pipeline carries ``int64`` class indices, so this is the inverse every
    consumer needs. It indexes ``classes_`` directly rather than calling
    ``inverse_transform``, which for a binarizer expects its own indicator matrix
    rather than indices.

    Parameters
    ----------
    encoder : object
        The fitted encoder the indices came from.
    indices : np.ndarray
        1-D array of class indices.

    Returns
    -------
    np.ndarray
        Original label values, one per index. A powerset encoder returns 2-D rows.
    """
    idx = np.asarray(indices, dtype=np.int64)
    classes = getattr(encoder, "classes_", None)
    if classes is not None:
        return np.asarray(classes)[idx]
    if hasattr(encoder, "inverse_transform"):
        # A custom encoder that exposes no classes but can still decode.
        return np.asarray(encoder.inverse_transform(idx))
    return idx


def _class_name_map(encoder: object) -> dict[int, str]:
    """Return an ``{index: name}`` map from a fitted encoder."""
    classes = getattr(encoder, "classes_", None)
    if classes is None:
        return {}
    names = []
    for cls in np.asarray(classes):
        arr = np.asarray(cls)
        names.append("-".join(map(str, arr.tolist())) if arr.ndim else str(cls))
    return dict(enumerate(names))


def _encoded_targets(encoder: object, labels: np.ndarray) -> np.ndarray:
    """
    Transform *labels* with a fitted *encoder* and check the target contract.

    1-D means one class index per sample (what :class:`~torch.nn.CrossEntropyLoss`
    consumes), 2-D means one column per model output. The dtype is checked rather
    than cast, because casting a float output would silently truncate it.

    Parameters
    ----------
    encoder : object
        A fitted target encoder.
    labels : np.ndarray
        Labels in the form the encoder was fitted on.

    Returns
    -------
    np.ndarray
        The encoded targets as ``int64``, 1-D or 2-D.

    Raises
    ------
    ValueError
        If the encoder returns a non-integer dtype, or more than two dimensions.
    """
    targets = np.asarray(encoder.transform(labels))
    if targets.ndim > 2:
        raise ValueError(
            f"{type(encoder).__name__}.transform returned a {targets.ndim}-D array; "
            "labels must encode to class indices (1-D) or one column per output (2-D)."
        )
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError(
            f"{type(encoder).__name__}.transform returned dtype {targets.dtype}; "
            "labels must encode to integer class codes. A float output is not cast "
            "here because truncating it would change the labels silently."
        )
    return targets.astype(np.int64, copy=False)


def _fit_label_encoder(labels: np.ndarray, encoder: object | None = None) -> tuple:
    """
    Fit *encoder* on *labels* and return it with the encoded targets.

    Shared by :func:`_split_dataset` and by datasets that compute their own
    partition (see :meth:`~xai4tsc.data.base.DatasetBase.split` overrides), so the
    default encoder, the target contract and the class-name log line are written
    once.

    Parameters
    ----------
    labels : np.ndarray
        Raw labels in the form the encoder consumes: scalars for
        :class:`~sklearn.preprocessing.LabelEncoder`, multi-hot rows for
        :class:`MultiHotLabelEncoder` and :class:`PowersetLabelEncoder`.
    encoder : object or None
        An sklearn-style target encoder. ``None`` uses
        :class:`~sklearn.preprocessing.LabelEncoder`.

    Returns
    -------
    tuple[object, np.ndarray]
        The fitted encoder and the encoded targets (1-D class indices, or 2-D with
        one column per model output).
    """
    encoder = _resolve_encoder(encoder)
    encoder.fit(labels)
    targets = _encoded_targets(encoder, labels)
    logger.info("Label classes (index -> name): %s", _class_name_map(encoder))
    return encoder, targets


def _assemble_splits(
    data: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame | None,
    index_sets: list[np.ndarray],
) -> list[tuple]:
    """
    Build ``[(x, y, metadata), ...]`` from precomputed per-split index arrays.

    The counterpart to :func:`_split_dataset` for a dataset that owns its own
    partition: that function chooses the rows itself, this one is handed them. Both
    produce the same shapes and the same metadata form (one list of per-sample
    dicts per split), so the rest of the pipeline cannot tell which produced a
    split.

    Parameters
    ----------
    data : np.ndarray
        Full ``(B, C, T)`` array.
    labels : np.ndarray
        Full label array, already encoded by :func:`_fit_label_encoder`.
    metadata : pd.DataFrame or None
        Per-sample metadata, one row per sample, index-aligned with *data*.
    index_sets : list of np.ndarray
        Row indices per split, in the project's ``[train, test, val]`` order.

    Returns
    -------
    list of tuple
        One ``(x, y, metadata)`` tuple per entry in *index_sets*.
    """
    records = metadata.to_dict("records") if metadata is not None else None
    splits = []
    for indices in index_sets:
        idx = np.asarray(indices, dtype=int)
        meta = [records[i] for i in idx] if records is not None else None
        splits.append(
            (
                np.asarray(data[idx], dtype=np.float32),
                np.asarray(labels)[idx],
                meta,
            )
        )
    return splits


def _split_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame | None = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    random_state: int | None = DEFAULT_SEED,
    encoder: object | None = None,
    stratify: bool = True,
) -> tuple:
    """Split data into train / test / (optional) val with label encoding."""
    encoder, labels = _fit_label_encoder(labels, encoder)

    val_set = not math.isclose(val_split, 0.0)
    # Fraction held out of training (validation + test combined).
    holdout = round(1 - train_split, 4)

    if metadata is not None:
        all_data = list(zip(data, metadata.to_dict("records"), strict=True))
    else:
        all_data = data

    strat_full = labels if stratify else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            all_data,
            labels,
            test_size=holdout,
            random_state=random_state,
            stratify=strat_full,
        )
    except ValueError:
        logger.warning(
            "Stratified split not possible (a class has too few samples); "
            "falling back to random split."
        )
        x_train, x_test, y_train, y_test = train_test_split(
            all_data, labels, test_size=holdout, random_state=random_state
        )

    x_val = y_val = None
    if val_set:
        # Within the holdout, the test portion is (1 - train - val) / (1 - train);
        # the remainder becomes the validation set.
        rel_test = (1 - train_split - val_split) / (1 - train_split)
        strat_holdout = y_test if stratify else None
        try:
            x_val, x_test, y_val, y_test = train_test_split(
                x_test,
                y_test,
                test_size=rel_test,
                random_state=random_state,
                stratify=strat_holdout,
            )
        except ValueError:
            logger.warning(
                "Stratified val split not possible; falling back to random split."
            )
            x_val, x_test, y_val, y_test = train_test_split(
                x_test,
                y_test,
                test_size=rel_test,
                random_state=random_state,
            )

    if metadata is not None:
        data_sets, metadata_sets = [], []
        for _set in [x for x in [x_train, x_test, x_val] if x is not None]:
            _data, _meta = zip(*_set, strict=True)
            data_sets.append(np.array(_data, dtype=np.float32))
            metadata_sets.append(list(_meta))
        x_train, x_test = data_sets[0], data_sets[1]
        metadata_train, metadata_test = metadata_sets[0], metadata_sets[1]
        if val_set:
            x_val, metadata_val = data_sets[2], metadata_sets[2]
        else:
            metadata_val = None
    else:
        metadata_train = metadata_test = metadata_val = None

    logger.info("Training data shape: %s", x_train.shape)
    logger.info("Test data shape: %s", x_test.shape)
    if val_set:
        logger.info("Validation data shape: %s", x_val.shape)

    if val_set:
        splits = [
            (x_train, y_train, metadata_train),
            (x_test, y_test, metadata_test),
            (x_val, y_val, metadata_val),
        ]
    else:
        splits = [
            (x_train, y_train, metadata_train),
            (x_test, y_test, metadata_test),
        ]
    return splits, encoder


class DatasetBase(ABC):
    """
    Base class for all datasets in xai4tsc.

    Provides concrete implementations of splitting, saving, and loading
    pre-split data.  Subclasses only need to implement :meth:`load`.

    Data domain
    -----------
    :attr:`data_domain` is the signal domain the loaded arrays are *expressed in*,
    and it travels downstream on :attr:`xai4tsc.Explanation.data_domain`, where it
    defines the model-native domain that explainers and metrics are checked against.
    It is the time domain for raw series, which is every dataset shipped here;
    a dataset of spectra declares otherwise. Subclasses that know their own
    domain set it as a class attribute; :class:`~xai4tsc.data.LocalDataset` takes it
    as a constructor argument, since only the user knows what is in the file.
    """

    name: str = None
    """Human-readable dataset identifier."""
    data_domain: Domain = Domain.TIME
    """Signal domain the loaded arrays are expressed in (see above)."""
    _splits: list = None
    """Cached list of (X, y, metadata) tuples set by :meth:`split`."""
    _encoder = None
    """Fitted sklearn encoder set alongside ``_splits``."""

    def __init__(
        self,
        max_samples: int | None = None,
        sample_strategy: SampleStrategy = "random",
        max_series_length: int | None = None,
        series_position: SeriesPosition = "first",
    ) -> None:
        """
        Store dataset restriction parameters.

        Parameters
        ----------
        max_samples : int | None
            If set, the dataset is subsampled to at most this many samples
            before splitting.  Strategy is controlled by *sample_strategy*.
        sample_strategy : str
            How to select the subset when *max_samples* is active:
            ``"random"`` (default), ``"first"``, ``"last"``, or
            ``"stratified"`` (proportional per-class draw).
        max_series_length : int | None
            If set, each time series is truncated to at most this many
            timesteps.  Which end is kept is controlled by *series_position*.
        series_position : str
            Which part of a long series to retain: ``"first"`` (default,
            keep the beginning) or ``"last"`` (keep the end).

        Raises
        ------
        ValueError
            If *sample_strategy* or *series_position* is not an allowed value.
        """
        self._max_samples = max_samples
        self._sample_strategy = _check_option(
            "sample_strategy", sample_strategy, SampleStrategy
        )
        self._max_series_length = max_series_length
        self._series_position = _check_option(
            "series_position", series_position, SeriesPosition
        )
        self._split_rng: np.random.Generator | None = None

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> tuple:
        """
        Load raw (unsplit) data from the source.

        Returns
        -------
        tuple[np.ndarray, array-like, pd.DataFrame | None]
            ``(data, labels, metadata)`` where data has shape
            ``(n_samples, n_channels, n_timesteps)``.
        """

    # ── Restriction helpers ───────────────────────────────────────────────────

    def _select_sample_idx(
        self, n: int, labels_arr: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Return a sorted index array for sample restriction."""
        k = getattr(self, "_max_samples", None)
        if k is None or n <= k:
            return np.arange(n)
        strategy = getattr(self, "_sample_strategy", "random")
        if strategy == "first":
            return np.arange(k)
        if strategy == "last":
            return np.arange(n - k, n)
        if strategy == "random":
            idx = rng.choice(n, size=k, replace=False)
            idx.sort()
            return idx
        if strategy == "stratified":
            classes, counts = np.unique(labels_arr, return_counts=True)
            per_class = np.round(counts / n * k).astype(int)
            per_class[-1] += k - per_class.sum()
            per_class = np.maximum(per_class, 0)
            parts = [
                rng.choice(np.where(labels_arr == c)[0], size=int(m), replace=False)
                for c, m in zip(classes, per_class, strict=True)
                if m > 0
            ]
            idx = np.concatenate(parts)
            idx.sort()
            return idx
        raise ValueError(f"Unknown sample_strategy: '{strategy}'")

    def _restrict_samples(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        metadata: pd.DataFrame | None,
        rng: np.random.Generator,
    ) -> tuple:
        """
        Subsample to at most ``max_samples`` rows, metadata included.

        The selected indices are applied to all three, because *metadata* is one
        row per sample and index-aligned with *data*: restricting the arrays alone
        would leave it describing the samples that were dropped, and
        :func:`_split_dataset` pairs the two by position.

        Parameters
        ----------
        data : np.ndarray
            Full ``(B, C, T)`` array.
        labels : np.ndarray
            Full label array.
        metadata : pd.DataFrame or None
            Per-sample metadata, or ``None`` when the dataset carries none.
        rng : np.random.Generator
            Stream for the sample selection.

        Returns
        -------
        tuple
            ``(data, labels, metadata)``, restricted and still index-aligned.
        """
        n = data.shape[0]
        idx = self._select_sample_idx(n, labels, rng)
        if len(idx) == n:
            return data, labels, metadata
        logger.info(
            "Subsampling dataset from %d to %d samples (strategy: %s).",
            n,
            len(idx),
            getattr(self, "_sample_strategy", "random"),
        )
        restricted = (
            metadata.iloc[idx].reset_index(drop=True) if metadata is not None else None
        )
        return data[idx], labels[idx], restricted

    def _restrict_length_numpy(self, data: np.ndarray) -> np.ndarray:
        """Truncate the time axis to at most ``max_series_length`` timesteps."""
        t = getattr(self, "_max_series_length", None)
        if t is None or data.shape[2] <= t:
            return data
        logger.info(
            "Truncating series from %d to %d timesteps (position: %s).",
            data.shape[2],
            t,
            getattr(self, "_series_position", "first"),
        )
        if getattr(self, "_series_position", "first") == "last":
            return data[:, :, -t:]
        return data[:, :, :t]

    def _restrict_samples_nested(
        self, x_nested: pd.DataFrame, y: np.ndarray, rng: np.random.Generator
    ) -> tuple:
        """Subsample a sktime nested DataFrame to at most ``max_samples`` rows."""
        n = len(x_nested)
        idx = self._select_sample_idx(n, y, rng)
        if len(idx) == n:
            return x_nested, y
        logger.info(
            "Subsampling nested dataset from %d to %d samples (strategy: %s).",
            n,
            len(idx),
            getattr(self, "_sample_strategy", "random"),
        )
        return x_nested.iloc[idx].reset_index(drop=True), y[idx]

    def _restrict_length_nested(self, x_nested: pd.DataFrame) -> pd.DataFrame:
        """Truncate each ``pd.Series`` cell to ``max_series_length`` timesteps."""
        t = getattr(self, "_max_series_length", None)
        if t is None:
            return x_nested
        logger.info(
            "Truncating nested series to %d timesteps (position: %s).",
            t,
            getattr(self, "_series_position", "first"),
        )
        if getattr(self, "_series_position", "first") == "last":

            def clip(s: pd.Series) -> pd.Series:
                return s.iloc[-t:]
        else:

            def clip(s: pd.Series) -> pd.Series:
                return s.iloc[:t]

        return x_nested.apply(lambda col: col.map(clip))

    # ── Splitting ─────────────────────────────────────────────────────────────

    def split(
        self,
        train_split: float = 0.8,
        val_split: float = 0.1,
        random_state: int | None = DEFAULT_SEED,
        encoder: object | None = None,
        impute_missing: bool = False,
        rng: np.random.Generator | None = None,
        stratify: bool = True,
        use_predefined_splits: bool = False,
    ) -> tuple:
        """
        Load and split the dataset into train / test / (optional) val.

        Parameters
        ----------
        train_split : float
            Fraction of samples for training.
        val_split : float
            Fraction of samples for validation (0 disables validation set).
        random_state : int or None
            Seed for reproducible splits, :data:`~xai4tsc.DEFAULT_SEED` by
            default.  ``None`` draws from OS entropy and is not reproducible.
            Ignored when *rng* is provided.
        encoder : object, optional
            An sklearn-style target encoder (``fit`` / ``transform`` /
            ``classes_``), used to turn the stored labels into model targets and
            retained so class indices can be mapped back to names. ``None``
            (default) uses :class:`~sklearn.preprocessing.LabelEncoder`. The
            encoded rank is the formulation: 1-D gives one class index per sample,
            2-D gives one column per model output. scikit-learn *feature*
            encoders (``OneHotEncoder``, ``OrdinalEncoder``) are rejected - see
            :func:`_reject_feature_encoder`.
        impute_missing : bool
            If ``True``, NaN values are replaced with the per-channel mean
            before splitting.  If ``False`` (default), a :class:`ValueError`
            is raised when NaN values are detected.
        rng : np.random.Generator | None
            A shared ``numpy`` Generator instance.  When supplied, *random_state*
            is ignored and this generator is used for all random operations
            (sample restriction, sklearn split seed) so that the entire pipeline
            draws from a single reproducible random stream.
        stratify : bool
            If ``True`` (default), splits are stratified by class label so that
            every class is proportionally represented in each split.  Falls back
            to random splitting with a warning when a class has too few samples
            to stratify.
        use_predefined_splits : bool
            Use the partition the dataset itself defines, where it has one, and
            ignore *train_split* / *val_split* / *stratify*. Only subclasses that
            own a partition honour it - the archive's ``_TRAIN``/``_TEST`` files
            for :class:`~xai4tsc.data.UcrUeaDataset`, the source project's
            partition for
            :class:`~xai4tsc.data.datasets.SynthMultilabelFreqDataset`. A dataset
            with no such partition reports the request and splits normally, since
            there is no other partition it could have served.

        Returns
        -------
        tuple[list, encoder]
            ``(splits, fitted_encoder)`` where *splits* is a list of
            ``(X, y, metadata)`` tuples in order ``[train, test, val]``.
        """
        if use_predefined_splits:
            logger.warning(
                "'%s' defines no partition of its own, so "
                "use_predefined_splits=True is ignored and the data is split by "
                "the requested fractions. Pre-split data on disk is loaded with "
                "load_saved_splits() instead.",
                getattr(self, "name", type(self).__name__),
            )
        rng = resolve_rng(random_state, rng)
        sklearn_seed = int(rng.integers(2**31))

        # Store rng so load() implementations can use it for pre-pad restriction.
        self._split_rng = rng
        try:
            data, labels, metadata = self.load()
        finally:
            self._split_rng = None

        labels = np.asarray(labels)
        data, labels, metadata = self._restrict_samples(data, labels, metadata, rng)
        data = self._restrict_length_numpy(data)

        if np.isnan(data).any():
            n_nan = int(np.isnan(data).sum())
            if not impute_missing:
                raise ValueError(
                    f"Dataset contains {n_nan} NaN value(s). "
                    "Remove missing values before training, or set "
                    "impute_missing=True to replace them with the per-channel "
                    "mean."
                )
            logger.info("Imputing %d NaN value(s) with per-channel mean.", n_nan)
            data = _impute_data(data)
        self._splits, self._encoder = _split_dataset(
            data,
            labels,
            metadata,
            train_split,
            val_split,
            sklearn_seed,
            encoder,
            stratify=stratify,
        )
        return self._splits, self._encoder

    def get_splits(self) -> tuple:
        """
        Return the cached splits produced by :meth:`split`.

        Raises
        ------
        RuntimeError
            If :meth:`split` has not been called yet.
        """
        if self._splits is None:
            raise RuntimeError(
                "Call split() or load_saved_splits() before get_splits()."
            )
        return self._splits, self._encoder

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_splits(self, save_path: Path | str) -> None:
        """
        Save the current splits to *save_path/splits/* as ``.npy`` + ``.json`` files.

        Parameters
        ----------
        save_path : Path or str
            Parent directory.  A ``splits/`` sub-directory is created inside it.

        Raises
        ------
        RuntimeError
            If no splits are available yet.
        """
        if self._splits is None:
            raise RuntimeError("No splits to save. Call split() first.")
        splits = self._splits
        splits_dir = Path(save_path) / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        data_splits = {"train_data": splits[0][0], "test_data": splits[1][0]}
        split_names = ["train", "test"]
        if len(splits) > 2:
            data_splits["val_data"] = splits[2][0]
            split_names.append("val")

        for name, arr in data_splits.items():
            np.save(splits_dir / (name + ".npy"), arr)

        for split, name in zip(splits, split_names, strict=True):
            labels_arr = split[1]
            metadata = split[2] if len(split) > 2 else None
            if np.ndim(labels_arr) > 1:
                # A 2-D target is one column per model output, which for a
                # multi-hot encoding already *is* the stored label row.
                # decode_labels cannot help: it indexes classes_, and these are
                # indicator columns rather than indices. Persisting the rows is
                # also what a reload re-fits on, so the class set round-trips.
                decoded = np.asarray(labels_arr)
            else:
                decoded = decode_labels(self._encoder, labels_arr)
            if np.ndim(decoded) > 1:  # multi-hot rows
                label_col = [row.tolist() for row in decoded]
            else:
                label_col = decoded
            pd.DataFrame({"label": label_col}).to_json(
                splits_dir / (name + "_labels.json"), orient="records"
            )
            # Metadata goes in its own file (one record per sample, same order).
            if metadata is not None:
                pd.DataFrame.from_records(metadata).to_json(
                    splits_dir / (name + "_metadata.json"), orient="records"
                )

        logger.info("Splits saved to %s", splits_dir)

    def load_saved_splits(
        self, directory: Path | str, encoder: object | None = None
    ) -> tuple:
        """
        Load pre-split files from *directory*.

        Looks for ``train*.npy``, ``test*.npy``, ``val*.npy`` (optional) and
        matching ``*.json`` label files.

        Parameters
        ----------
        directory : Path or str
            Directory containing the split files.
        encoder : object, optional
            An sklearn-style target encoder, as in :meth:`split`. ``None`` uses
            :class:`~sklearn.preprocessing.LabelEncoder`. It is fitted on the union
            of the splits so every class is represented even when one split lacks
            it.

        Returns
        -------
        tuple[list, encoder]
            ``(splits, fitted_encoder)``.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        npy_files = sorted(directory.glob("*.npy"))
        json_files = sorted(directory.glob("*.json"))
        # Labels and (optional) separate metadata files are both *.json.
        label_files = [f for f in json_files if "metadata" not in f.stem]
        meta_files = [f for f in json_files if "metadata" in f.stem]

        x_train = x_test = x_val = None
        y_train = y_test = y_val = None
        metadata_train = metadata_test = metadata_val = None

        for f in npy_files:
            if "train" in f.stem:
                x_train = np.load(f)
            elif "test" in f.stem:
                x_test = np.load(f)
            elif "val" in f.stem:
                x_val = np.load(f)

        for f in label_files:  # labels (+ embedded metadata columns, legacy layout)
            if "train" in f.stem:
                y_train, metadata_train = load_json(f)
            elif "test" in f.stem:
                y_test, metadata_test = load_json(f)
            elif "val" in f.stem:
                y_val, metadata_val = load_json(f)

        for f in meta_files:  # separate metadata.json overrides any embedded columns
            if "train" in f.stem:
                metadata_train = load_metadata(f)
            elif "test" in f.stem:
                metadata_test = load_metadata(f)
            elif "val" in f.stem:
                metadata_val = load_metadata(f)

        if any(s is None for s in [x_train, x_test, y_train, y_test]):
            raise ValueError(
                f"Could not find train and test split files in: {directory}"
            )

        # Labels may be scalars (single-label) or multi-hot lists; normalise to
        # arrays (1-D for scalars, 2-D for multi-hot).
        y_train = np.array(y_train.tolist())
        y_test = np.array(y_test.tolist())
        if y_val is not None:
            y_val = np.array(y_val.tolist())

        encoder = _resolve_encoder(encoder)
        present = [a for a in [y_train, y_test, y_val] if a is not None]
        # Stacking follows the stored label layout, not the encoder: rows stay rows
        # (multi-hot) and scalars stay scalars.
        all_labels = (
            np.vstack(present) if present[0].ndim == 2 else np.concatenate(present)
        )
        encoder.fit(all_labels)
        logger.info("Label classes (index -> name): %s", _class_name_map(encoder))

        # Same target contract as _split_dataset.
        y_train = _encoded_targets(encoder, y_train)
        y_test = _encoded_targets(encoder, y_test)
        if y_val is not None:
            y_val = _encoded_targets(encoder, y_val)

        if x_val is not None and y_val is not None:
            splits = [
                (x_train, y_train, metadata_train),
                (x_test, y_test, metadata_test),
                (x_val, y_val, metadata_val),
            ]
        else:
            splits = [
                (x_train, y_train, metadata_train),
                (x_test, y_test, metadata_test),
            ]

        self._splits = splits
        self._encoder = encoder
        return splits, encoder
