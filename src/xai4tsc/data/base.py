"""``DatasetBase`` ABC: splitting, label encoding, saving, loading pre-split data."""

import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from .data_loaders import load_json, load_metadata

logger = logging.getLogger(__name__)


# ── Module-level helpers ───────────────────────────────────────────────────────


def _ensure_bct(data: np.ndarray) -> np.ndarray:
    """
    Transpose ``(B, T, C)`` → ``(B, C, T)`` when ``T > C``, with a warning.

    The framework always expects ``(batch, channels, timesteps)`` ordering.
    This heuristic is only safe for local user data where channels are typically
    few and timesteps are long.  **Do not call this on UCR/UEA data loaded via
    sktime** — sktime already returns ``(B, C, T)`` and datasets with more
    channels than timesteps (e.g. spectrograms) would be silently corrupted.
    """
    if data.ndim == 3 and data.shape[1] > data.shape[2]:
        logger.warning(
            "Data shape %s looks like (B, T, C); transposing to (B, C, T). "
            "Pass data already in (B, C, T) order to suppress this.",
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


class MultiHotLabelEncoder:
    """
    Collapse multi-hot label rows to single class indices, preserving the combos.

    Multi-label datasets emit a multi-hot label row per sample (e.g.
    ``[0, 1, 1]`` = attributes 1 and 2 present). Training uses
    :class:`~torch.nn.CrossEntropyLoss` on 1-D integer class indices, so each
    distinct combination is mapped to one class index. The combination itself is
    kept as the **class name** (``classes_``) so predictions can be decoded back
    to multi-hot later; :func:`_split_dataset` / :meth:`DatasetBase.load_saved_splits`
    log the index→name mapping.

    Implements the small subset of the sklearn encoder interface the framework
    uses (``fit`` / ``transform`` / ``fit_transform`` / ``inverse_transform`` and
    the ``classes_`` attribute), so it slots into the same code paths as
    :class:`~sklearn.preprocessing.LabelEncoder`.
    """

    def fit(self, y: np.ndarray) -> "MultiHotLabelEncoder":
        """Record the sorted unique multi-hot rows as ``classes_``."""
        arr = np.asarray(y)
        if arr.ndim != 2:
            raise ValueError(
                f"MultiHotLabelEncoder expects 2-D multi-hot labels (n, n_attrs); "
                f"got shape {arr.shape}."
            )
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


def _make_encoder(
    encode: str,
) -> LabelEncoder | OneHotEncoder | OrdinalEncoder | MultiHotLabelEncoder:
    """Return a fresh encoder for the given *encode* key."""
    if encode == "onehot":
        return OneHotEncoder()
    if encode == "ordinal":
        return OrdinalEncoder()
    if encode == "multihot":
        return MultiHotLabelEncoder()
    return LabelEncoder()


def _class_name_map(encoder: object) -> dict[int, str]:
    """Return an ``{index: name}`` map from a fitted encoder's ``classes_``."""
    classes = getattr(encoder, "classes_", None)
    if classes is None:
        return {}
    names = []
    for cls in classes:
        arr = np.asarray(cls)
        names.append("-".join(map(str, arr.tolist())) if arr.ndim else str(cls))
    return dict(enumerate(names))


def _class_index_view(encoded: np.ndarray) -> np.ndarray:
    """
    Collapse encoded labels to a 1-D array of integer class indices.

    ``train_test_split(stratify=...)`` needs 1-D class labels, but one-hot and
    ordinal encoders produce 2-D arrays (one-hot may even be a scipy sparse
    matrix).  This maps any encoding back to class indices so stratification
    works regardless of *encode*.
    """
    if hasattr(encoded, "toarray"):  # scipy sparse one-hot
        return np.asarray(encoded.argmax(axis=1)).ravel()
    arr = np.asarray(encoded)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] > 1:  # dense one-hot
        return arr.argmax(axis=1)
    return arr.ravel()  # ordinal (n, 1)


def _split_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame | None = None,
    train_split: float = 0.8,
    val_split: float = 0.0,
    random_state: int = 42,
    encode: str = "label",
    stratify: bool = True,
) -> tuple:
    """Split data into train / test / (optional) val with label encoding."""
    encoder = _make_encoder(encode)
    # OneHot/Ordinal need a 2-D column; LabelEncoder (1-D) and MultiHotLabelEncoder
    # (already 2-D multi-hot rows) consume their input shape directly.
    if isinstance(encoder, (OneHotEncoder, OrdinalEncoder)):
        labels = np.array(labels).reshape(-1, 1)
    encoder.fit(labels)
    labels = encoder.transform(labels)
    logger.info("Label classes (index -> name): %s", _class_name_map(encoder))

    val_set = not math.isclose(val_split, 0.0)
    # Fraction held out of training (validation + test combined).
    holdout = round(1 - train_split, 4)

    if metadata is not None:
        all_data = list(zip(data, metadata.to_dict("records"), strict=True))
    else:
        all_data = data

    strat_full = _class_index_view(labels) if stratify else None
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
        strat_holdout = _class_index_view(y_test) if stratify else None
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
    """

    name: str = None
    """Human-readable dataset identifier."""
    _splits: list = None
    """Cached list of (X, y, metadata) tuples set by :meth:`split`."""
    _encoder = None
    """Fitted sklearn encoder set alongside ``_splits``."""

    def __init__(
        self,
        max_samples: int | None = None,
        sample_strategy: str = "random",
        max_series_length: int | None = None,
        series_position: str = "first",
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
        """
        self._max_samples = max_samples
        self._sample_strategy = sample_strategy
        self._max_series_length = max_series_length
        self._series_position = series_position
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

    def _restrict_samples_numpy(
        self, data: np.ndarray, labels: np.ndarray, rng: np.random.Generator
    ) -> tuple:
        """Subsample *data* to at most ``max_samples`` rows."""
        n = data.shape[0]
        idx = self._select_sample_idx(n, labels, rng)
        if len(idx) == n:
            return data, labels
        logger.info(
            "Subsampling dataset from %d to %d samples (strategy: %s).",
            n,
            len(idx),
            getattr(self, "_sample_strategy", "random"),
        )
        return data[idx], labels[idx]

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
        val_split: float = 0.0,
        random_state: int = 42,
        encode: str = "label",
        impute_missing: bool = False,
        rng: np.random.Generator | None = None,
        stratify: bool = True,
    ) -> tuple:
        """
        Load and split the dataset into train / test / (optional) val.

        Parameters
        ----------
        train_split : float
            Fraction of samples for training.
        val_split : float
            Fraction of samples for validation (0 disables validation set).
        random_state : int
            Seed for reproducible splits.  Ignored when *rng* is provided.
        encode : str
            Label encoding scheme: ``"label"`` (default), ``"onehot"``, or
            ``"ordinal"``.
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

        Returns
        -------
        tuple[list, encoder]
            ``(splits, fitted_encoder)`` where *splits* is a list of
            ``(X, y, metadata)`` tuples in order ``[train, test, val]``.
        """
        if rng is None:
            rng = np.random.default_rng(random_state)
        sklearn_seed = int(rng.integers(2**31))

        # Store rng so load() implementations can use it for pre-pad restriction.
        self._split_rng = rng
        try:
            data, labels, metadata = self.load()
        finally:
            self._split_rng = None

        labels = np.asarray(labels)
        data, labels = self._restrict_samples_numpy(data, labels, rng)
        data = self._restrict_length_numpy(data)

        if np.isnan(data).any():
            n_nan = int(np.isnan(data).sum())
            if not impute_missing:
                raise ValueError(
                    f"Dataset contains {n_nan} NaN value(s). "
                    "Remove missing values before training, or set "
                    "impute_missing=True (allow_imputation: true in YAML)."
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
            encode,
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
            # Multi-hot: persist the decoded multi-hot rows (not the collapsed
            # indices) so a reload re-derives the same index<->combo mapping; a
            # fresh encoder re-fit on bare indices could not.
            if isinstance(self._encoder, MultiHotLabelEncoder):
                label_col = [
                    row.tolist() for row in self._encoder.inverse_transform(labels_arr)
                ]
            else:
                label_col = labels_arr
            pd.DataFrame({"label": label_col}).to_json(
                splits_dir / (name + "_labels.json"), orient="records"
            )
            # Metadata goes in its own file (one record per sample, same order).
            if metadata is not None:
                pd.DataFrame.from_records(metadata).to_json(
                    splits_dir / (name + "_metadata.json"), orient="records"
                )

        logger.info("Splits saved to %s", splits_dir)

    def load_saved_splits(self, directory: Path | str, encode: str = "label") -> tuple:
        """
        Load pre-split files from *directory*.

        Looks for ``train*.npy``, ``test*.npy``, ``val*.npy`` (optional) and
        matching ``*.json`` label files.

        Parameters
        ----------
        directory : Path or str
            Directory containing the split files.
        encode : str
            Label encoding scheme.

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

        encoder = _make_encoder(encode)
        present = [a for a in [y_train, y_test, y_val] if a is not None]
        if isinstance(encoder, MultiHotLabelEncoder):
            all_labels = np.vstack(present)
        else:
            all_labels = np.concatenate(present)
            if isinstance(encoder, (OneHotEncoder, OrdinalEncoder)):
                all_labels = all_labels.reshape(-1, 1)
                y_train = y_train.reshape(-1, 1)
                y_test = y_test.reshape(-1, 1)
                if y_val is not None:
                    y_val = y_val.reshape(-1, 1)
        encoder.fit(all_labels)
        logger.info("Label classes (index -> name): %s", _class_name_map(encoder))

        y_train = encoder.transform(y_train)
        y_test = encoder.transform(y_test)
        if y_val is not None:
            y_val = encoder.transform(y_val)

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
