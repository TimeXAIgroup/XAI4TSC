"""Dataset classes: ``UcrUeaDataset``, ``LocalDataset``, ``SyntheticDataset``."""

import logging
import math
from abc import ABC, abstractmethod
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sktime.datasets import load_UCR_UEA_dataset

from .._domains import Domain, resolve_data_domain
from .._random import DEFAULT_SEED, resolve_rng
from .base import (
    DatasetBase,
    _assemble_splits,
    _class_index_view,
    _ensure_bct,
    _fit_label_encoder,
    _impute_data,
    _make_encoder,
)
from .data_loaders import FORMAT_LOADERS

logger = logging.getLogger(__name__)


def _powerset(items: tuple | list) -> list[tuple]:
    """
    Return all ``2 ** n`` subsets of *items*, ordered by size then by value.

    Parameters
    ----------
    items : tuple or list
        The elements to combine.

    Returns
    -------
    list of tuple
        The subsets, starting with the empty one.
    """
    return [
        combination
        for size in range(len(items) + 1)
        for combination in combinations(items, size)
    ]


def _pad_variable_length(x_nested: pd.DataFrame) -> np.ndarray:
    """
    Convert a sktime ``nested_univ`` DataFrame to a zero-padded ``(B, C, T)`` array.

    All series are right-padded with zeros to the length of the longest series
    in the dataset.

    Parameters
    ----------
    x_nested : pd.DataFrame
        Panel data in sktime ``nested_univ`` format — each cell is a
        ``pd.Series`` whose length may vary across rows.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, n_channels, max_length)``, dtype ``float32``.
    """
    n_samples = len(x_nested)
    n_channels = len(x_nested.columns)
    max_len = max(x_nested.iloc[:, 0].apply(len))
    result = np.zeros((n_samples, n_channels, max_len), dtype=np.float32)
    for i in range(n_samples):
        for c in range(n_channels):
            series = np.asarray(x_nested.iloc[i, c], dtype=np.float32)
            result[i, c, : len(series)] = series
    return result


class UcrUeaDataset(DatasetBase):
    """Download from the UCR/UEA archive and load raw (unsplit) data."""

    def __init__(
        self,
        name: str,
        cache_dir: Path | None = None,
        download: bool = True,
        pad_series: bool = False,
        max_samples: int | None = None,
        sample_strategy: str = "random",
        max_series_length: int | None = None,
        series_position: str = "first",
        use_predefined_splits: bool = False,
    ) -> None:
        """
        Instantiate a UCR/UEA dataset.

        Parameters
        ----------
        name : str
            Official UCR/UEA dataset name, e.g. ``"ECG5000"``.
        cache_dir : Path | None
            Passed to sktime as ``extract_path``.  sktime stores the downloaded
            ``.ts`` files under ``cache_dir/{name}/`` and reuses them on
            subsequent calls.  If ``None``, sktime falls back to its default
            cache inside the package directory (lost on env recreation).
        download : bool
            If ``False``, raise :class:`FileNotFoundError` when the dataset is not
            found in *cache_dir* instead of downloading.
        pad_series : bool
            If ``True``, variable-length datasets are zero-padded to the length
            of the longest series before returning.  If ``False`` (default), a
            :class:`ValueError` is raised for variable-length data.
        max_samples : int | None
            Subsample to at most this many samples before padding.  Prevents OOM
            on very large datasets.  See :class:`DatasetBase` for details.
        sample_strategy : str
            How to select the subset: ``"random"``, ``"first"``, ``"last"``, or
            ``"stratified"``.  See :class:`DatasetBase` for details.
        max_series_length : int | None
            Truncate series to at most this many timesteps.  Applied before
            padding for variable-length datasets.
        series_position : str
            Which end to retain: ``"first"`` or ``"last"``.
        use_predefined_splits : bool
            If ``True``, load the official ``_TRAIN.ts`` and ``_TEST.ts`` files
            directly instead of combining and re-splitting the data.  No
            validation set is produced.  ``val_split`` and ``stratify`` are
            ignored.
        """
        super().__init__(
            max_samples=max_samples,
            sample_strategy=sample_strategy,
            max_series_length=max_series_length,
            series_position=series_position,
        )
        self.name = name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.download = download
        self.pad_series = pad_series
        self.use_predefined_splits = use_predefined_splits

    def _load_one_split(self, split: str, rng: np.random.Generator) -> tuple:
        """
        Load one official split (``"TRAIN"`` or ``"TEST"``) via sktime.

        Applies sample/length restrictions and variable-length padding using the
        same logic as :meth:`load`.

        Parameters
        ----------
        split : str
            ``"TRAIN"`` or ``"TEST"``.
        rng : np.random.Generator
            Generator used for sample restriction.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(data, labels)`` with shape ``(n, C, T)`` and ``(n,)``.
        """
        ucr_name = self.name
        extract_path = Path(self.cache_dir) if self.cache_dir else None
        if extract_path is not None:
            extract_path.mkdir(exist_ok=True, parents=True)
            extract_path = extract_path.absolute()
        logger.info("Loading official %s split for '%s' via sktime…", split, ucr_name)
        try:
            data, labels = load_UCR_UEA_dataset(
                name=ucr_name,
                split=split,
                return_X_y=True,
                return_type="numpy3d",
                extract_path=extract_path,
            )
            labels_arr = np.asarray(labels)
            data, labels_arr = self._restrict_samples_numpy(data, labels_arr, rng)
            data = self._restrict_length_numpy(data)
        except ValueError as exc:
            if "same shape" not in str(exc):
                raise
            if not self.pad_series:
                raise ValueError(
                    f"Dataset '{self.name}' contains variable-length time series "
                    "and cannot be converted to a fixed-shape array.  Set "
                    "pad_series=True (allow_padding: true in YAML) to zero-pad "
                    "all series to the maximum length."
                ) from exc
            x_nested, labels = load_UCR_UEA_dataset(
                name=ucr_name,
                split=split,
                return_X_y=True,
                return_type="nested_univ",
                extract_path=extract_path,
            )
            labels_arr = np.asarray(labels)
            x_nested, labels_arr = self._restrict_samples_nested(
                x_nested, labels_arr, rng
            )
            x_nested = self._restrict_length_nested(x_nested)
            logger.info(
                "Dataset '%s' %s split has variable-length series; "
                "zero-padding to max length.",
                self.name,
                split,
            )
            data = _pad_variable_length(x_nested)
        logger.info("Loaded '%s' %s split — %d samples", self.name, split, len(data))
        return data, labels_arr

    def load(self) -> tuple:
        """
        Load raw data from the UCR/UEA archive via sktime.

        sktime downloads and caches the ``.ts`` files under ``cache_dir``
        (passed as ``extract_path``).  Restrictions (``max_samples``,
        ``max_series_length``) are applied after loading.  The caller
        (``split()``) saves the processed train / val / test arrays to the
        split cache.
        """
        rng = resolve_rng(DEFAULT_SEED, self._split_rng)
        ucr_name = self.name
        extract_path = Path(self.cache_dir) if self.cache_dir else None
        if extract_path is not None:
            extract_path.mkdir(exist_ok=True, parents=True)
            extract_path = extract_path.absolute()

        if not self.download:
            if extract_path is None:
                raise FileNotFoundError(
                    f"Dataset '{self.name}': download=False but no cache_dir provided."
                )
            ts_dir = Path(self.cache_dir) / ucr_name
            if not ts_dir.exists() or not any(ts_dir.glob("*.ts")):
                raise FileNotFoundError(
                    f"Dataset '{self.name}' not found in cache ({ts_dir}) "
                    "and download=False."
                )

        logger.info("Loading UCR/UEA dataset '%s' via sktime…", ucr_name)
        try:
            data, labels = load_UCR_UEA_dataset(
                name=ucr_name,
                return_X_y=True,
                return_type="numpy3d",
                extract_path=extract_path,
            )
            labels_arr = np.asarray(labels)
            data, labels_arr = self._restrict_samples_numpy(data, labels_arr, rng)
            data = self._restrict_length_numpy(data)
        except ValueError as exc:
            if "same shape" not in str(exc):
                raise
            if not self.pad_series:
                raise ValueError(
                    f"Dataset '{self.name}' contains variable-length time series "
                    "and cannot be converted to a fixed-shape array.  Set "
                    "pad_series=True (allow_padding: true in YAML) to zero-pad "
                    "all series to the maximum length."
                ) from exc
            logger.info(
                "Dataset '%s' has variable-length series; zero-padding to max length.",
                self.name,
            )
            x_nested, labels = load_UCR_UEA_dataset(
                name=ucr_name,
                return_X_y=True,
                return_type="nested_univ",
                extract_path=extract_path,
            )
            labels_arr = np.asarray(labels)
            # Restrict on the nested DataFrame BEFORE padding to prevent OOM.
            x_nested, labels_arr = self._restrict_samples_nested(
                x_nested, labels_arr, rng
            )
            x_nested = self._restrict_length_nested(x_nested)
            data = _pad_variable_length(x_nested)

        logger.info("Loaded UCR dataset '%s' — %d samples", self.name, len(data))
        return data, pd.Series(labels_arr), None

    def split(
        self,
        train_split: float = 0.8,
        val_split: float = 0.0,
        random_state: int | None = DEFAULT_SEED,
        encode: str = "label",
        impute_missing: bool = False,
        rng: np.random.Generator | None = None,
        stratify: bool = True,
        class_order: list | None = None,
    ) -> tuple:
        """
        Load and split the dataset.

        When :attr:`use_predefined_splits` is ``True``, loads the official
        ``_TRAIN.ts`` and ``_TEST.ts`` files directly and ignores
        ``train_split``, ``val_split``, and ``stratify``.  No validation set
        is produced, and passing a non-default value for any of the three is
        reported rather than silently dropped.

        Otherwise delegates to :meth:`DatasetBase.split` for random /
        stratified splitting.
        """
        if not self.use_predefined_splits:
            return super().split(
                train_split=train_split,
                val_split=val_split,
                random_state=random_state,
                encode=encode,
                impute_missing=impute_missing,
                rng=rng,
                stratify=stratify,
                class_order=class_order,
            )

        self._report_ignored_split_args(train_split, val_split, stratify)

        rng = resolve_rng(random_state, rng)
        data_train, labels_train = self._load_one_split("TRAIN", rng)
        data_test, labels_test = self._load_one_split("TEST", rng)

        if impute_missing:
            for arr, split_name in [(data_train, "TRAIN"), (data_test, "TEST")]:
                n_nan = int(np.isnan(arr).sum())
                if n_nan:
                    logger.info(
                        "Imputing %d NaN value(s) in %s split with per-channel mean.",
                        n_nan,
                        split_name,
                    )
            data_train = _impute_data(data_train)
            data_test = _impute_data(data_test)

        # Fit encoder on union of both splits so all classes are represented
        encoder = _make_encoder(encode, class_order=class_order)
        all_labels = np.concatenate([labels_train, labels_test])
        if not isinstance(encoder, LabelEncoder):
            all_labels = all_labels.reshape(-1, 1)
        encoder.fit(all_labels)

        def _transform(enc: object, arr: np.ndarray) -> np.ndarray:
            needs_2d = not isinstance(enc, LabelEncoder)
            encoded = enc.transform(arr.reshape(-1, 1) if needs_2d else arr)
            return _class_index_view(encoded).astype(np.int64)

        y_train = _transform(encoder, labels_train)
        y_test = _transform(encoder, labels_test)

        logger.info("Predefined TRAIN split shape: %s", data_train.shape)
        logger.info("Predefined TEST split shape: %s", data_test.shape)

        self._splits = [(data_train, y_train, None), (data_test, y_test, None)]
        self._encoder = encoder
        return self._splits, self._encoder

    def _report_ignored_split_args(
        self, train_split: float, val_split: float, stratify: bool
    ) -> None:
        """
        Warn when split arguments are passed that the archive partition ignores.

        Silently dropping them leaves a user editing ``val_split`` with no
        effect and no explanation - and the validation set they asked for is
        the one that decides whether early stopping monitors validation or
        training loss, so the consequence is worth stating rather than
        inferring.

        ``random_state`` is deliberately not reported: it still seeds
        ``max_samples`` restriction in :meth:`_load_one_split`.

        Parameters
        ----------
        train_split : float
            The requested training fraction.
        val_split : float
            The requested validation fraction.
        stratify : bool
            Whether stratified splitting was requested.
        """
        ignored = []
        if not math.isclose(train_split, 0.8, abs_tol=1e-9):
            ignored.append(f"train_split={train_split:.4g}")
        if not math.isclose(val_split, 0.0, abs_tol=1e-9):
            ignored.append(f"val_split={val_split:.4g}")
        if not stratify:
            ignored.append("stratify=False")
        if not ignored:
            return

        logger.warning(
            "'%s' uses the archive's own TRAIN/TEST partition, so %s %s ignored "
            "and no validation set is produced - early stopping will monitor "
            "training loss. Set use_predefined_splits=False to split the pooled "
            "data by these fractions instead.",
            self.name,
            ", ".join(ignored),
            "is" if len(ignored) == 1 else "are",
        )


class LocalDataset(DatasetBase):
    """
    Load a single-file dataset from a local directory.

    The directory must contain one ``data*.npy`` file and one ``label*.json``
    file (unsplit layout).  For pre-split local data, use
    :meth:`~xai4tsc.data.DatasetBase.load_saved_splits` directly.

    """

    def __init__(
        self,
        path: str | Path,
        name: str,
        data_format: str = "numpy",
        layout: str = "auto",
        data_domain: str | Domain = "time",
        pad_series: bool = False,
        max_samples: int | None = None,
        sample_strategy: str = "random",
        max_series_length: int | None = None,
        series_position: str = "first",
    ) -> None:
        """
        Instantiate a local dataset.

        Parameters
        ----------
        path : str | Path
            Path to the directory containing ``data*.npy`` and ``label*.json``.
        name : str
            Human-readable identifier used for logging and cache-key generation.
            Required because paths may differ across machines.
        data_format : str
            File format: ``"numpy"`` (default), ``"arff"``, ``"csv"``, ``"wfdb"``.
        layout : str
            Axis order of the stored array: ``"auto"`` (default) guesses from the
            shape and transposes when axis 1 exceeds axis 2, ``"bct"`` trusts it as
            ``(B, C, T)``, ``"btc"`` always transposes. Set ``"bct"`` for data that
            genuinely has more channels than timesteps, which ``"auto"`` would
            otherwise corrupt.
        data_domain : str or Domain
            Signal domain the stored arrays are expressed in: ``"time"`` (default),
            ``"frequency"`` or ``"time_frequency"``. Only the user knows this about
            a file, it is set downstream as the model-native domain (see
            :class:`~xai4tsc.data.DatasetBase`). Non-time data must still be real
            ``float32``, e.g. magnitude spectra, not complex coefficients.
        pad_series : bool
            If ``True``, variable-length data (list of arrays) is zero-padded to
            the length of the longest series before returning.  If ``False``
            (default), a :class:`ValueError` is raised for ragged data.
        max_samples : int | None
            Subsample to at most this many samples.  See :class:`DatasetBase`.
        sample_strategy : str
            Selection strategy: ``"random"``, ``"first"``, ``"last"``, or
            ``"stratified"``.  See :class:`DatasetBase`.
        max_series_length : int | None
            Truncate series to at most this many timesteps.
        series_position : str
            Which end to retain: ``"first"`` or ``"last"``.
        """
        super().__init__(
            max_samples=max_samples,
            sample_strategy=sample_strategy,
            max_series_length=max_series_length,
            series_position=series_position,
        )
        self.path = Path(path)
        self.name = name
        self.data_format = data_format
        self.layout = layout
        self.data_domain = resolve_data_domain(data_domain)
        self.pad_series = pad_series

    def load(self) -> tuple:
        """
        Load raw data from the local directory using the configured format loader.

        Returns
        -------
        tuple
            ``(data, labels, metadata)`` with data in ``(B, C, T)`` layout.
        """
        loader = FORMAT_LOADERS.get(self.data_format)
        if loader is None:
            raise ValueError(
                f"Unknown data_format '{self.data_format}'. "
                f"Choose from: {list(FORMAT_LOADERS)}"
            )
        data, labels, metadata = loader(self.path)
        # Apply (B, T, C) → (B, C, T) heuristic for local data only.
        data = _ensure_bct(data, self.layout)
        logger.info("Loaded local dataset '%s' — %d samples", self.name, len(data))
        return data, labels, metadata


class SyntheticDataset(DatasetBase, ABC):
    """
    Base class for programmatically generated datasets.

    Subclasses implement :meth:`generate_dataset` (pure, deterministic given the
    constructor's ``seed`` + parameters). The concrete :meth:`load` here owns
    **caching**: it first looks for a previously generated dataset on disk and
    only regenerates on a miss.

    Caching layout (under ``cache_dir/synthetic/<cache_key>/``):

    - ``data.npy``      — generated array ``(B, C, T)``
    - ``labels.json``   — labels as records (``[{"label": ...}, ...]``; multi-hot
      labels are stored as lists)
    - ``metadata.json`` — per-sample metadata records (optional)

    **Metadata convention:** one row per sample, index-aligned with ``data`` in
    the same order. For ground-truth localization datasets it carries the
    discriminative regions keyed **per class** (``ground_truth``) plus distractors
    (``non_discriminative``); see the package data conventions.

    Example::

        class GaussianDataset(SyntheticDataset):
            def _cache_params(self):
                return {"n": self.n_samples, "c": self.n_classes, "t": self.length}

            def generate_dataset(self):
                rng = np.random.default_rng(self.seed)
                data = rng.standard_normal((self.n_samples, 1, self.length))
                labels = rng.integers(0, self.n_classes, self.n_samples)
                return data.astype("float32"), labels, None
    """

    def __init__(
        self,
        name: str,
        cache_dir: str | Path | None = None,
        seed: int = DEFAULT_SEED,
        max_samples: int | None = None,
        sample_strategy: str = "random",
        max_series_length: int | None = None,
        series_position: str = "first",
    ) -> None:
        """
        Store identity, cache location, and generation seed.

        Parameters
        ----------
        name : str
            Dataset identifier (also the registry key).
        cache_dir : str | Path | None
            Cache root.  When ``None`` the dataset is generated fresh on every
            :meth:`load` with no persistence.
        seed : int
            Seed for deterministic generation.
        max_samples, sample_strategy, max_series_length, series_position
            Restriction parameters forwarded to :class:`DatasetBase`.
        """
        super().__init__(
            max_samples=max_samples,
            sample_strategy=sample_strategy,
            max_series_length=max_series_length,
            series_position=series_position,
        )
        self.name = name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.seed = seed

    @abstractmethod
    def generate_dataset(self) -> tuple:
        """
        Generate and return ``(data, labels, metadata)`` — deterministic.

        ``data`` is ``(B, C, T)`` float32; ``labels`` is ``(B,)`` integer or
        ``(B, n_attrs)`` multi-hot (collapsed to class indices by the
        ``multihot`` encoder); ``metadata`` is the per-sample DataFrame (or
        ``None``).
        """

    def _cache_params(self) -> dict:
        """Return generation parameters defining identity (override in subclasses)."""
        return {}

    def _cache_key(self) -> str:
        """Stable cache folder name from name + seed + generation parameters."""
        params = self._cache_params()
        parts = [self.name, f"s{self.seed}"]
        parts += [f"{k}{params[k]}" for k in sorted(params)]
        return "_".join(str(p) for p in parts)

    def cache_subdir(self, root: str | Path) -> Path:
        """Return this dataset's cache directory under ``root/synthetic/``."""
        return Path(root) / "synthetic" / self._cache_key()

    @staticmethod
    def is_split_layout(directory: str | Path) -> bool:
        """Return ``True`` if *directory* holds a pre-split train/test layout."""
        d = Path(directory)
        if not d.is_dir():
            return False
        stems = [f.stem for f in d.glob("*.npy")]
        return any("train" in s for s in stems) and any("test" in s for s in stems)

    @staticmethod
    def is_raw_layout(directory: str | Path) -> bool:
        """Return ``True`` if *directory* holds a single generated blob (not split)."""
        d = Path(directory)
        if not d.is_dir():
            return False
        has_data = any(f.stat().st_size > 0 for f in d.glob("data*.npy"))
        return has_data and not SyntheticDataset.is_split_layout(d)

    def load(self) -> tuple:
        """
        Return raw ``(data, labels, metadata)``, generating + caching on a miss.

        With no ``cache_dir`` the dataset is generated fresh. Otherwise the cache
        directory is checked first (first-look); on a hit the persisted blob is
        loaded without regeneration, on a miss it is generated and persisted.
        """
        if self.cache_dir is None:
            return self.generate_dataset()

        cache = self.cache_subdir(self.cache_dir)
        if self.is_raw_layout(cache):
            logger.info(
                "Loading cached synthetic dataset '%s' from %s", self.name, cache
            )
            data, labels, metadata = FORMAT_LOADERS["numpy"](cache)
            data = data.astype(np.float32)
            labels = np.array(labels.tolist()) if hasattr(labels, "tolist") else labels
            return data, labels, metadata

        data, labels, metadata = self.generate_dataset()
        self._persist_raw(cache, data, labels, metadata)
        return data, labels, metadata

    def _persist_raw(
        self,
        cache: Path,
        data: np.ndarray,
        labels: np.ndarray,
        metadata: object,
    ) -> None:
        """Write the generated dataset to *cache* (data/labels/metadata files)."""
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "data.npy", np.asarray(data, dtype=np.float32))
        pd.DataFrame({"label": np.asarray(labels).tolist()}).to_json(
            cache / "labels.json", orient="records"
        )
        if metadata is not None:
            df = (
                metadata
                if isinstance(metadata, pd.DataFrame)
                else pd.DataFrame.from_records(metadata)
            )
            df.to_json(cache / "metadata.json", orient="records")
        logger.info(
            "Generated and cached synthetic dataset '%s' to %s", self.name, cache
        )


class SynthMultilabelFreqDataset(SyntheticDataset):
    """
    Multi-label synthetic dataset with time-frequency ground-truth explanations.

    Each sample is a sum of sinusoidal bursts. A multi-hot label marks which
    discriminative frequencies are present: frequency ``k`` of
    ``discriminative_frequencies`` contributes a burst of
    ``frequency_window_length`` samples at a random time window, and the classes
    are the power set of those frequencies (3 frequencies give 8 classes).
    Non-discriminative distractor sinusoids are added at other frequencies, some
    spanning the whole series and some windowed, so the discriminative signal has
    to be localized in both time and frequency to be found.

    Per-sample metadata records the discriminative regions under ``ground_truth``
    and the distractors under ``non_discriminative``, which makes this dataset a
    ground truth for the localization metrics (e.g. ``Time-Frequency AUC``).
    Regions are stored as the plain mappings that survive the JSON round-trip of
    :meth:`~xai4tsc.data.base.DatasetBase.save_splits`, with the singular
    ``channel`` / ``freq`` / ``phase`` keys;
    :meth:`~xai4tsc.evaluation._ground_truth.GroundTruthRegion.from_mapping`
    parses them.

    Region frequencies are recorded in Hz, as the source dataset writes them, and
    every metadata row carries the ``sampling_rate`` they are relative to.

    The dataset owns its own train / test / val partition - the source
    repository's two-stage stratified split, seeded by ``seed`` - so the
    ``train_split`` and ``val_split`` arguments of :meth:`split` are not used.
    Configure the partition with ``train_ratio`` and ``test_ratio`` instead.

    Notes
    -----
    Adapted from the reference implementation at
    https://github.com/TimeXAIgroup/multilabel-ts-xai-benchmark
    (``data/benchmark.py``, ``generate_dataset.py``, ``utils.py``), MIT licensed,
    Copyright (c) 2026 TimeXAIgroup; see ``THIRD_PARTY_LICENSES.md``.

    References
    ----------
    J. Fischer, N. Gumpfer, T. Papenbrock, and J. Hannig, "A Synthetic Multi-Label
    Time Series Classification Dataset with Time-Frequency Ground-Truth Explanation
    for Evaluating Explainable Artificial Intelligence Methods", SSRN preprint
    7389286. Dataset: https://doi.org/10.5281/zenodo.20744119
    """

    data_domain: Domain = Domain.TIME
    """Raw series: the discriminative features are frequencies localized in time,
    but the arrays themselves are time-domain ``(B, 1, T)`` ``float32``."""

    def __init__(
        self,
        name: str = "Synth_Multilabel_Freq",
        cache_dir: str | Path | None = None,
        seed: int = 42,
        sample_length: int = 500,
        sampling_rate: int = 100,
        discriminative_frequencies: tuple[int, ...] = (7, 19, 37),
        frequency_window_length: int = 150,
        num_samples_per_pattern: int = 500,
        overlap_min: float = 0.25,
        overlap_max: float = 0.5,
        noise_std: float = 0.1,
        sin_amplitude: float = 1.0,
        num_full_distractions: int = 3,
        num_window_distractions: int = 3,
        train_ratio: float = 0.7,
        test_ratio: float = 0.15,
        max_samples: int | None = None,
        sample_strategy: str = "random",
        max_series_length: int | None = None,
        series_position: str = "first",
    ) -> None:
        """
        Configure the generator and the partition.

        The defaults are the source repository's, and reproduce its published
        dataset exactly: 4000 samples of length 500 at 100 Hz, split 2800/600/600.

        Parameters
        ----------
        name : str
            Dataset identifier (also the registry key).
        cache_dir : str or Path or None
            Cache root. ``None`` generates fresh on every :meth:`load` with no
            persistence.
        seed : int
            Seed for generation and for the partition, as the source repository
            uses one seed for both.
        sample_length : int
            Number of timesteps per sample.
        sampling_rate : int
            Sampling rate in Hz. Sets the time axis and the Nyquist frequency that
            bounds the distractor pool.
        discriminative_frequencies : tuple of int
            Frequencies in Hz whose presence or absence defines the label bits.
            All must lie below the Nyquist frequency.
        frequency_window_length : int
            Length in samples of each discriminative burst.
        num_samples_per_pattern : int
            Samples generated per class pattern; the total is this times
            ``2 ** len(discriminative_frequencies)``.
        overlap_min : float
            Minimum fractional overlap between consecutive discriminative bursts.
        overlap_max : float
            Maximum fractional overlap between consecutive discriminative bursts.
        noise_std : float
            Standard deviation of the additive Gaussian background noise.
        sin_amplitude : float
            Amplitude of every sinusoidal component, discriminative or distractor.
        num_full_distractions : int
            Distractor sinusoids spanning the whole sample.
        num_window_distractions : int
            Distractor sinusoids placed in random windows of
            ``frequency_window_length``. These are placed independently, so they
            are not subject to the overlap bounds.
        train_ratio : float
            Fraction of samples in the training split.
        test_ratio : float
            Fraction of samples in the test split; validation takes the remainder.
        max_samples, sample_strategy, max_series_length, series_position
            Restriction parameters forwarded to
            :class:`~xai4tsc.data.base.DatasetBase`. A sample restriction is
            applied before the partition, so it changes which rows each split
            holds.

        Raises
        ------
        ValueError
            If a burst is longer than a sample, a discriminative frequency is not
            below the Nyquist frequency, the distractor pool cannot supply the
            requested number of distinct distractors, or the split ratios do not
            leave a positive validation fraction.
        """
        super().__init__(
            name=name,
            cache_dir=cache_dir,
            seed=seed,
            max_samples=max_samples,
            sample_strategy=sample_strategy,
            max_series_length=max_series_length,
            series_position=series_position,
        )
        if frequency_window_length > sample_length:
            raise ValueError(
                f"frequency_window_length ({frequency_window_length}) must not "
                f"exceed sample_length ({sample_length})."
            )
        nyquist = int(sampling_rate / 2)
        out_of_band = [f for f in discriminative_frequencies if not 0 < f < nyquist]
        if out_of_band:
            raise ValueError(
                f"discriminative_frequencies {out_of_band} are not below the "
                f"Nyquist frequency ({nyquist} Hz) of sampling_rate "
                f"{sampling_rate}."
            )
        pool_size = len(
            [f for f in range(1, nyquist) if f not in discriminative_frequencies]
        )
        needed = max(num_full_distractions, num_window_distractions)
        if needed > pool_size:
            raise ValueError(
                f"{needed} distinct distractors requested but only {pool_size} "
                f"non-discriminative frequencies exist below {nyquist} Hz."
            )
        if not 0 < train_ratio < 1 or not 0 < test_ratio < 1:
            raise ValueError(
                f"train_ratio ({train_ratio}) and test_ratio ({test_ratio}) must "
                "each lie strictly between 0 and 1."
            )
        if train_ratio + test_ratio >= 1:
            raise ValueError(
                f"train_ratio ({train_ratio}) + test_ratio ({test_ratio}) must be "
                "below 1 so the validation split is non-empty."
            )
        self.sample_length = sample_length
        self.sampling_rate = sampling_rate
        self.discriminative_frequencies = tuple(discriminative_frequencies)
        self.frequency_window_length = frequency_window_length
        self.num_samples_per_pattern = num_samples_per_pattern
        self.overlap_min = overlap_min
        self.overlap_max = overlap_max
        self.noise_std = noise_std
        self.sin_amplitude = sin_amplitude
        self.num_full_distractions = num_full_distractions
        self.num_window_distractions = num_window_distractions
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    # ── Generation ─────────────────────────────────────────────────────────────

    @property
    def nyquist_frequency(self) -> int:
        """Highest representable frequency in Hz, bounding the distractor pool."""
        return int(self.sampling_rate / 2)

    @property
    def distractor_pool(self) -> np.ndarray:
        """Integer frequencies in Hz available to distractors, in ascending order."""
        return np.array(
            [
                f
                for f in range(1, self.nyquist_frequency)
                if f not in self.discriminative_frequencies
            ]
        )

    def _cache_params(self) -> dict:
        """
        Return the generation parameters that define the cached blob's identity.

        The partition ratios are deliberately absent: they slice the cached blob
        rather than change it, so two configs that split it differently share one
        generated copy.

        Returns
        -------
        dict
            Short keys mapped to their values, joined into the cache folder name.
        """
        return {
            "a": self.sin_amplitude,
            "dg": self.num_full_distractions,
            "dw": self.num_window_distractions,
            "f": "-".join(str(f) for f in self.discriminative_frequencies),
            "fs": self.sampling_rate,
            "n": self.num_samples_per_pattern,
            "ov": f"{self.overlap_min}-{self.overlap_max}",
            "sd": self.noise_std,
            "t": self.sample_length,
            "w": self.frequency_window_length,
        }

    def generate_dataset(self) -> tuple:
        """
        Generate the whole dataset, unsplit.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, pd.DataFrame]
            ``(data, labels, metadata)`` where *data* is ``(B, 1, T)`` float32,
            *labels* is ``(B, n_frequencies)`` multi-hot and *metadata* holds one
            row per sample with ``ground_truth``, ``non_discriminative`` and
            ``sampling_rate``.
        """
        rng = np.random.default_rng(self.seed)
        time_axis = np.arange(self.sample_length) / self.sampling_rate
        pool = self.distractor_pool

        # First pass: draw every sample's regions. The signals are synthesised only
        # afterwards, so the noise stream sits entirely downstream of this one.
        labels, metadata = [], []
        for pattern in _powerset(self.discriminative_frequencies):
            label = [int(f in pattern) for f in self.discriminative_frequencies]
            active = np.array(pattern)
            for _ in range(self.num_samples_per_pattern):
                labels.append(label)
                if len(active) > 1:
                    rng.shuffle(active)
                metadata.append(
                    {
                        "ground_truth": self._draw_bursts(rng, active),
                        "non_discriminative": self._draw_distractors(rng, pool),
                        "sampling_rate": self.sampling_rate,
                    }
                )

        # Second pass: synthesise each signal from the regions already drawn.
        data = np.stack([self._make_sample(rng, row, time_axis) for row in metadata])
        logger.info(
            "Generated %d samples of length %d at %d Hz (%d classes).",
            len(labels),
            self.sample_length,
            self.sampling_rate,
            2 ** len(self.discriminative_frequencies),
        )
        return (
            data[:, None, :],
            np.array(labels, dtype=np.int64),
            pd.DataFrame(metadata),
        )

    def _draw_bursts(self, rng: np.random.Generator, freqs: np.ndarray) -> list[dict]:
        """
        Draw one discriminative region per frequency in *freqs*.

        Parameters
        ----------
        rng : np.random.Generator
            The generation stream.
        freqs : np.ndarray
            Active discriminative frequencies in Hz, already shuffled so that the
            earliest window is not always the lowest frequency.

        Returns
        -------
        list of dict
            One region mapping per frequency, ordered by window position.
        """
        windows = self._place_windows(rng, len(freqs))
        return [
            {
                "channel": 0,
                "pos": start,
                "len": stop - start,
                "freq": int(freq),
                "phase": float(rng.uniform(0, 2 * np.pi)),
            }
            for freq, (start, stop) in zip(freqs, windows, strict=True)
        ]

    def _draw_distractors(
        self, rng: np.random.Generator, pool: np.ndarray
    ) -> list[dict]:
        """
        Draw the non-discriminative components: full-length first, then windowed.

        The two groups are drawn without replacement independently, so one
        frequency may appear once in each group of the same sample.

        Parameters
        ----------
        rng : np.random.Generator
            The generation stream.
        pool : np.ndarray
            Candidate frequencies in Hz, excluding the discriminative ones.

        Returns
        -------
        list of dict
            Region mappings for the full-length then the windowed distractors.
        """
        full = rng.choice(pool, size=self.num_full_distractions, replace=False)
        regions = [
            {
                "channel": 0,
                "pos": 0,
                "len": self.sample_length,
                "freq": int(freq),
                "phase": float(rng.uniform(0, 2 * np.pi)),
            }
            for freq in full
        ]
        windowed = rng.choice(pool, size=self.num_window_distractions, replace=False)
        # Placed one at a time, so these are exempt from the overlap bounds that
        # apply between consecutive discriminative bursts.
        windows = [self._place_windows(rng, 1)[0] for _ in windowed]
        regions += [
            {
                "channel": 0,
                "pos": start,
                "len": stop - start,
                "freq": int(freq),
                "phase": float(rng.uniform(0, 2 * np.pi)),
            }
            for freq, (start, stop) in zip(windowed, windows, strict=True)
        ]
        return regions

    def _place_windows(
        self, rng: np.random.Generator, num_windows: int
    ) -> list[tuple[int, int]]:
        """
        Return ``(start, stop)`` for *num_windows* windows placed in the series.

        A single window is placed uniformly. Two or more are placed as one chain:
        a hop is drawn per adjacent pair so consecutive windows overlap by between
        ``overlap_min`` and ``overlap_max`` of their length, and the whole chain is
        then shifted to a uniformly drawn offset. Only consecutive windows are
        constrained, so with three windows the first and last never overlap.

        Parameters
        ----------
        rng : np.random.Generator
            The generation stream.
        num_windows : int
            How many windows to place; ``0`` draws nothing and returns an empty
            list.

        Returns
        -------
        list of tuple of int
            One ``(start, stop)`` pair per window, ordered in time.
        """
        width = self.frequency_window_length
        if num_windows == 0:
            return []
        if num_windows == 1:
            start = int(rng.integers(0, max(1, self.sample_length - width + 1)))
            return [(start, start + width)]

        hop_min = max(1, int(width * (1 - self.overlap_max)))
        hop_max = max(hop_min, int(width * (1 - self.overlap_min)))
        hops = rng.integers(hop_min, hop_max + 1, size=num_windows - 1)

        span = int(hops.sum()) + width
        if span > self.sample_length:
            # The chain does not fit: shrink the hops proportionally, which tightens
            # the overlap past overlap_max rather than dropping a window.
            scale = (self.sample_length - width) / max(1, int(hops.sum()))
            hops = np.maximum(1, (hops * scale)).astype(int)
            span = int(hops.sum()) + width

        offset = int(rng.integers(0, max(1, self.sample_length - span + 1)))
        starts = [offset] + [
            offset + int(hops[:k].sum()) for k in range(1, num_windows)
        ]
        return [(start, start + width) for start in starts]

    def _make_sample(
        self, rng: np.random.Generator, row: dict, time_axis: np.ndarray
    ) -> np.ndarray:
        """
        Synthesise one signal from the regions already recorded for it.

        Every component is a slice of a sinusoid evaluated on absolute time, so a
        region's ``phase`` is its phase at ``t=0`` rather than at the window onset,
        and the envelope is rectangular.

        Parameters
        ----------
        rng : np.random.Generator
            The generation stream; one noise draw is taken per sample.
        row : dict
            One metadata row, holding ``ground_truth`` and ``non_discriminative``.
        time_axis : np.ndarray
            ``arange(sample_length) / sampling_rate``, in seconds.

        Returns
        -------
        np.ndarray
            The signal, ``(sample_length,)`` float32.
        """
        signal = rng.normal(0, self.noise_std, self.sample_length).astype(np.float32)
        regions = row["ground_truth"] + row["non_discriminative"]
        for region in regions:
            start = region["pos"]
            stop = min(start + region["len"], self.sample_length)
            signal[start:stop] += self.sin_amplitude * np.sin(
                2 * np.pi * region["freq"] * time_axis[start:stop] + region["phase"]
            )
        return signal

    # ── Splitting ──────────────────────────────────────────────────────────────

    def split(
        self,
        train_split: float = 0.8,
        val_split: float = 0.0,
        random_state: int | None = DEFAULT_SEED,
        encode: str = "label",
        impute_missing: bool = False,
        rng: np.random.Generator | None = None,
        stratify: bool = True,
        class_order: list | None = None,
    ) -> tuple:
        """
        Generate (or load from cache) and apply this dataset's own partition.

        The partition is the source repository's: two chained stratified
        ``train_test_split`` calls seeded by this dataset's ``seed``, giving
        ``train_ratio`` / ``test_ratio`` / remainder. It is a property of the
        dataset rather than of the run, so re-running returns the same rows.

        Parameters
        ----------
        train_split : float
            Not used; the partition comes from ``train_ratio``. A value that
            disagrees with it is reported once.
        val_split : float
            Not used; the validation fraction is ``1 - train_ratio - test_ratio``.
            A value that disagrees with it is reported once.
        random_state : int or None
            Seeds the sample restriction only. The partition uses the dataset's
            own ``seed``.
        encode : str
            Label encoding scheme; ``"multihot"`` is the fitting choice for this
            dataset's multi-hot labels and the one that lets a region be scoped to
            the classes it discriminates.
        impute_missing : bool
            Accepted for signature parity. Generated data holds no NaN, so there
            is nothing to impute.
        rng : np.random.Generator or None
            Shared generator for the sample restriction; overrides *random_state*.
        stratify : bool
            Accepted for signature parity. The partition is stratified by class
            combination by construction.
        class_order : list or None
            Explicit class order, forwarded to the encoder.

        Returns
        -------
        tuple[list, encoder]
            ``(splits, fitted_encoder)`` where *splits* is a list of
            ``(X, y, metadata)`` tuples in order ``[train, test, val]``.
        """
        self._report_unused_ratios(train_split, val_split)
        rng = resolve_rng(random_state, rng)

        self._split_rng = rng
        try:
            data, labels, metadata = self.load()
        finally:
            self._split_rng = None

        labels = np.asarray(labels)
        data, labels, metadata = self._restrict(data, labels, metadata, rng)
        full_length = data.shape[-1]
        data = self._restrict_length_numpy(data)
        if data.shape[-1] != full_length:
            # The regions are positions in the generated series. Truncating it does
            # not move them, so a region past the cut now describes signal that is
            # no longer there; the masks clip rather than crash, which is exactly
            # what makes this worth saying out loud.
            logger.warning(
                "'%s' was truncated from %d to %d timesteps, but its ground-truth "
                "regions still describe the full series. Localization metrics will "
                "score against regions that may fall outside the truncated signal.",
                self.name,
                full_length,
                data.shape[-1],
            )

        encoder, targets = _fit_label_encoder(labels, encode, class_order)
        splits = _assemble_splits(data, targets, metadata, self._partition(labels))
        splits = [(x, y, self._scope_metadata(m, encoder)) for x, y, m in splits]
        names = ("Training", "Test", "Validation")
        for name, (x, _, _) in zip(names, splits, strict=True):
            logger.info("%s data shape: %s", name, x.shape)
        self._splits, self._encoder = splits, encoder
        return self._splits, self._encoder

    def _report_unused_ratios(self, train_split: float, val_split: float) -> None:
        """
        Warn when configured split fractions disagree with this dataset's own.

        Silently ignoring them would leave a user editing ``train_split`` with no
        effect and no explanation.

        Parameters
        ----------
        train_split : float
            The requested training fraction.
        val_split : float
            The requested validation fraction.
        """
        own_val = round(1 - self.train_ratio - self.test_ratio, 10)
        mismatched = not math.isclose(
            train_split, self.train_ratio, abs_tol=1e-9
        ) or not math.isclose(val_split, own_val, abs_tol=1e-9)
        if mismatched:
            logger.warning(
                "'%s' supplies its own partition (train %.4g / test %.4g / val "
                "%.4g), so train_split=%.4g and val_split=%.4g are ignored. Set "
                "train_ratio / test_ratio in init_params to change it.",
                self.name,
                self.train_ratio,
                self.test_ratio,
                own_val,
                train_split,
                val_split,
            )

    def _restrict(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        metadata: pd.DataFrame | None,
        rng: np.random.Generator,
    ) -> tuple:
        """
        Apply ``max_samples`` to data, labels and metadata together.

        :meth:`~xai4tsc.data.base.DatasetBase._restrict_samples_numpy` drops rows
        from the arrays only, which would leave the metadata one row per original
        sample and so no longer aligned. Selecting the indices once and applying
        them to all three keeps the per-sample alignment the ground truth depends
        on.

        Parameters
        ----------
        data : np.ndarray
            Full ``(B, C, T)`` array.
        labels : np.ndarray
            Full label array.
        metadata : pd.DataFrame or None
            Per-sample metadata, or ``None``.
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

    def _partition(self, labels: np.ndarray) -> list[np.ndarray]:
        """
        Return the per-split row indices, in ``[train, test, val]`` order.

        Ported from the source repository: a first stratified split peels off the
        training rows, a second divides what is left into validation and test. The
        stratification key is the string form of the multi-hot row, as there, and
        both calls use this dataset's ``seed``. Sizes are rounded to absolute
        counts before the call, which is what keeps the ratios exact.

        Parameters
        ----------
        labels : np.ndarray
            Multi-hot label rows, one per sample.

        Returns
        -------
        list of np.ndarray
            Train, test and validation row indices.
        """
        indices = np.arange(len(labels))
        keys = [str(row) for row in np.asarray(labels).astype(int)]

        train_size = round(self.train_ratio * len(indices))
        train_idx, rest_idx = self._stratified_split(
            indices, keys, len(indices) - train_size
        )
        rest_keys = [keys[i] for i in rest_idx]
        test_size = round(self.test_ratio / (1 - self.train_ratio) * len(rest_idx))
        val_idx, test_idx = self._stratified_split(rest_idx, rest_keys, test_size)
        return [train_idx, test_idx, val_idx]

    def _stratified_split(
        self, indices: np.ndarray, keys: list[str], holdout: int
    ) -> tuple:
        """
        Split *indices*, stratified by *keys*, falling back to random if it cannot.

        A small ``num_samples_per_pattern`` leaves too few samples of some class
        combination to stratify, which sklearn reports as a bare error. The base
        splitter warns and falls back in that case, so this does too rather than
        failing a run over a size the user is free to choose.

        Parameters
        ----------
        indices : np.ndarray
            Row indices to divide.
        keys : list of str
            Stratification key per entry of *indices*.
        holdout : int
            Number of rows to hold out.

        Returns
        -------
        tuple
            ``(kept, holdout)`` index arrays.
        """
        try:
            return train_test_split(
                indices,
                test_size=holdout,
                stratify=keys,
                random_state=self.seed,
            )
        except ValueError:
            logger.warning(
                "Stratified split not possible for '%s' (a class combination has "
                "too few samples); falling back to a random split. Raise "
                "num_samples_per_pattern to stratify.",
                self.name,
            )
            return train_test_split(indices, test_size=holdout, random_state=self.seed)

    # ── Ground-truth class scoping ─────────────────────────────────────────────

    def load_saved_splits(self, directory: Path | str, encode: str = "label") -> tuple:
        """
        Load the saved splits and scope each region to the classes it marks.

        Parameters
        ----------
        directory : Path or str
            Directory containing the split files.
        encode : str
            Label encoding scheme.

        Returns
        -------
        tuple[list, encoder]
            As :meth:`~xai4tsc.data.base.DatasetBase.load_saved_splits`, with
            ``classes`` added to every ground-truth region.
        """
        splits, encoder = super().load_saved_splits(directory, encode=encode)
        scoped = [
            (x, y, self._scope_metadata(metadata, encoder)) for x, y, metadata in splits
        ]
        self._splits = scoped
        return scoped, encoder

    def _scope_metadata(self, metadata: object, encoder: object) -> object:
        """
        Return *metadata* with each ground-truth region scoped to its classes.

        A label bit of this dataset means "the sample contains a burst at
        ``discriminative_frequencies[bit]``", so a region's frequency says which
        bit it belongs to. Under the multi-hot encoding every combination is its
        own class, so a region marks every class whose combination has that bit
        set, and the evaluation side can then score a per-class explanation
        against the regions that actually belong to that class.

        The scoping happens here rather than in the stored files because it
        depends on the label encoding, which is only resolved at load time.

        Parameters
        ----------
        metadata : object
            Per-sample metadata: a ``pd.DataFrame`` or a list of row mappings.
            Anything else is returned unchanged.
        encoder : object
            The fitted label encoder.

        Returns
        -------
        object
            The same form as *metadata*, with ``classes`` added to each region.
            Regions are rebuilt rather than mutated, so the loaded JSON objects
            are left alone.
        """
        classes_for = self._classes_per_frequency(encoder)
        if not classes_for or metadata is None:
            return metadata

        def scope(regions: object) -> object:
            """Add the class scope to every region of one sample."""
            if not isinstance(regions, list):
                return regions
            out = []
            for region in regions:
                if not isinstance(region, dict):
                    out.append(region)
                    continue
                classes = classes_for.get(region.get("freq"))
                out.append({**region, "classes": list(classes)} if classes else region)
            return out

        if isinstance(metadata, pd.DataFrame):
            if "ground_truth" not in metadata:
                return metadata
            scoped = metadata.copy()
            scoped["ground_truth"] = scoped["ground_truth"].map(scope)
            return scoped
        return [
            {**row, "ground_truth": scope(row["ground_truth"])}
            if isinstance(row, dict) and "ground_truth" in row
            else row
            for row in metadata
        ]

    def _classes_per_frequency(self, encoder: object) -> dict[int, tuple[int, ...]]:
        """
        Map each discriminative frequency to the class indices it marks.

        Parameters
        ----------
        encoder : object
            The fitted label encoder; ``classes_`` must hold one multi-hot row per
            class for the mapping to be derivable.

        Returns
        -------
        dict
            ``{frequency: (class_index, ...)}``, or empty when the encoding is not
            multi-hot and a frequency therefore says nothing about a class.
        """
        classes = list(getattr(encoder, "classes_", []))
        n_bits = len(self.discriminative_frequencies)
        if not classes or any(np.ndim(c) != 1 or len(c) != n_bits for c in classes):
            return {}
        return {
            freq: tuple(
                index for index, combo in enumerate(classes) if int(combo[bit]) == 1
            )
            for bit, freq in enumerate(self.discriminative_frequencies)
        }


# ── Synthetic-dataset registry ──────────────────────────────────────────────────

SYNTHETIC_DATASETS: dict[str, type] = {
    "Synth_Multilabel_Freq": SynthMultilabelFreqDataset,
}


def register_synthetic_dataset(name: str, dataset_class: type) -> None:
    """
    Register a custom :class:`SyntheticDataset` subclass under *name*.

    The factory :func:`xai4tsc.data.load_dataset` dispatches a YAML
    ``dataset: <name>`` entry to the registered class (checked before the
    path/UCR branches), so registering a name makes it usable from config.

    Parameters
    ----------
    name : str
        Registry key (matched against the YAML ``dataset`` field).
    dataset_class : type
        A :class:`SyntheticDataset` subclass.
    """
    SYNTHETIC_DATASETS[name] = dataset_class


def load_dataset(
    name: str,
    path: str | None = None,
    cache_dir: str | None = None,
    pad_series: bool = False,
    max_samples: int | None = None,
    sample_strategy: str = "random",
    max_series_length: int | None = None,
    series_position: str = "first",
    layout: str = "auto",
    data_domain: str = "time",
    **kwargs: object,
) -> DatasetBase:
    """
    Resolve to the correct :class:`DatasetBase` subclass.

    Parameters
    ----------
    name : str
        Human-readable dataset name (always required). If *name* is in the
        :data:`SYNTHETIC_DATASETS` registry, the matching
        :class:`SyntheticDataset` is returned (checked first) and generation
        params are forwarded as ``**kwargs``.
    path : str | Path | None
        If provided, a :class:`LocalDataset` is returned pointing at *path*.
        If ``None`` (and *name* is not a synthetic dataset), a
        :class:`UcrUeaDataset` is returned.
    cache_dir : Path | None
        Passed through to :class:`UcrUeaDataset` when *path* is ``None``, stores
        UCR / UEA datasets to mitigate multiple downloads.
    pad_series : bool
        If ``True``, variable-length series are zero-padded to the maximum
        series length before returning data.  Passed to the selected subclass.
    max_samples : int | None
        Subsample to at most this many samples before splitting.  Prevents OOM
        on very large datasets (e.g. InsectWingbeat).
    sample_strategy : str
        How to select the subset when *max_samples* is active: ``"random"``
        (default), ``"first"``, ``"last"``, or ``"stratified"``.
    max_series_length : int | None
        Truncate each series to at most this many timesteps.  Applied before
        padding for variable-length datasets.
    series_position : str
        Which end of a long series to retain: ``"first"`` (default) or
        ``"last"``.
    layout : str
        Axis order of a local array: ``"auto"`` (default), ``"bct"`` or ``"btc"``.
        Only meaningful with *path*; UCR/UEA and synthetic data are already
        ``(B, C, T)``, so it is not forwarded to them.
    data_domain : str
        Signal domain a local array is expressed in: ``"time"`` (default),
        ``"frequency"`` or ``"time_frequency"``. Only meaningful with *path*:
        UCR/UEA data is always raw series and a synthetic dataset declares its own
        domain, so a non-default value is ignored with a warning for those.
    **kwargs
        Forwarded to the selected subclass constructor (e.g. ``download``,
        ``data_format``).
    """
    restriction_kwargs = dict(
        max_samples=max_samples,
        sample_strategy=sample_strategy,
        max_series_length=max_series_length,
        series_position=series_position,
    )
    if data_domain != "time" and path is None:
        logger.warning(
            "'data_domain: %s' is ignored for '%s': UCR/UEA data is always raw "
            "series and a synthetic dataset declares its own domain. Only a local "
            "('path') dataset takes this setting.",
            data_domain,
            name,
        )
    if name in SYNTHETIC_DATASETS:
        # Generation params arrive as kwargs (the runner forwards `init_params`).
        for k in ("data_format", "download", "use_predefined_splits"):
            kwargs.pop(k, None)
        return SYNTHETIC_DATASETS[name](
            name=name,
            cache_dir=cache_dir,
            **restriction_kwargs,
            **kwargs,
        )
    if path is not None:
        if name is None:
            raise ValueError("'name' is required when 'path' is provided.")
        kwargs.pop("download", None)
        if kwargs.pop("use_predefined_splits", False):
            logger.warning(
                "'use_predefined_splits' does not apply to the local dataset "
                "'%s' and is ignored; use 'is_presplit' to load an "
                "already-split directory.",
                name,
            )
        return LocalDataset(
            path,
            name=name,
            layout=layout,
            data_domain=data_domain,
            pad_series=pad_series,
            **restriction_kwargs,
            **kwargs,
        )
    # UcrUeaDataset does not use data_format — strip it before forwarding
    kwargs.pop("data_format", None)
    return UcrUeaDataset(
        name,
        cache_dir=cache_dir,
        pad_series=pad_series,
        **restriction_kwargs,
        **kwargs,
    )
