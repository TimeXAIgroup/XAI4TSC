"""Dataset classes: ``UcrUeaDataset``, ``LocalDataset``, ``SyntheticDataset``."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sktime.datasets import load_UCR_UEA_dataset

from .base import DatasetBase, _ensure_bct, _impute_data, _make_encoder
from .data_loaders import FORMAT_LOADERS

logger = logging.getLogger(__name__)


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
        extract_path = str(self.cache_dir) if self.cache_dir else None
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
        rng = self._split_rng or np.random.default_rng()
        ucr_name = self.name
        extract_path = str(self.cache_dir) if self.cache_dir else None

        if not self.download:
            if extract_path is None:
                raise FileNotFoundError(
                    f"Dataset '{self.name}': download=False but no cache_dir provided."
                )
            ts_dir = self.cache_dir / ucr_name
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
        random_state: int = 42,
        encode: str = "label",
        impute_missing: bool = False,
        rng: np.random.Generator | None = None,
        stratify: bool = True,
    ) -> tuple:
        """
        Load and split the dataset.

        When :attr:`use_predefined_splits` is ``True``, loads the official
        ``_TRAIN.ts`` and ``_TEST.ts`` files directly and ignores
        ``train_split``, ``val_split``, and ``stratify``.  No validation set
        is produced.

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
            )

        rng = rng or np.random.default_rng(random_state)
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
        encoder = _make_encoder(encode)
        all_labels = np.concatenate([labels_train, labels_test])
        if not isinstance(encoder, LabelEncoder):
            all_labels = all_labels.reshape(-1, 1)
        encoder.fit(all_labels)

        def _transform(enc: object, arr: np.ndarray) -> np.ndarray:
            needs_2d = not isinstance(enc, LabelEncoder)
            return enc.transform(arr.reshape(-1, 1) if needs_2d else arr)

        y_train = _transform(encoder, labels_train)
        y_test = _transform(encoder, labels_test)

        logger.info("Predefined TRAIN split shape: %s", data_train.shape)
        logger.info("Predefined TEST split shape: %s", data_test.shape)

        self._splits = [(data_train, y_train, None), (data_test, y_test, None)]
        self._encoder = encoder
        return self._splits, self._encoder


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
        # UCR/UEA data from sktime is already (B, C, T) so this call lives here,
        # not in DatasetBase.split(), to avoid silently corrupting spectrogram
        # datasets where channels > timesteps.
        data = _ensure_bct(data)
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
        seed: int = 0,
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


class FreqShapesDataset(SyntheticDataset):
    """
    Localized wave-packet dataset for XAI time-frequency localization evaluation.

    Each sample is a sum of windowed sinusoids ("wave packets"). A multi-hot label
    marks which discriminative attributes are present; attribute ``k`` adds a
    packet at its characteristic frequency ``class_freqs[k]`` placed at a random
    time window. Non-discriminative distractor packets — global background
    sinusoids and shorter local packets, all at *other* frequencies — are added so
    the discriminative signal must be localized in both time and frequency.

    The per-sample metadata records the ground-truth regions **per class**
    (``ground_truth = {attr_index: [component, ...]}``) plus distractors
    (``non_discriminative``); a component is
    ``{"channel", "pos", "len", "freq", "phase"}``. This makes the dataset a
    ground truth for time-frequency attribution metrics (e.g. TimeFrequencyAUC).

    .. note::

       The programmatic generator is **currently disabled** —
       :meth:`generate_dataset` is a placeholder that raises. ``freq_shapes`` ships
       a fixed, paper-faithful pre-split dataset committed under the synthetic cache
       dir (``cache/datasets/synthetic/freq_shapes/``); the runner loads it directly
       as a pre-split layout via ``load_saved_splits`` when ``cache_path`` is set.
       The constructor still accepts the original generation ``init_params`` so
       existing configs do not break, but they are inert. This is a temporary
       arrangement until the generator is validated.
    """

    def __init__(
        self,
        name: str = "freq_shapes",
        cache_dir: str | Path | None = None,
        seed: int = 0,
        n_samples: int = 600,
        length: int = 500,
        class_freqs: tuple[int, ...] = (7, 19, 37),
        packet_len: int = 150,
        n_global_distractors: int = 3,
        n_local_distractors: int = 3,
        max_freq: int = 49,
        max_samples: int | None = None,
        sample_strategy: str = "random",
        max_series_length: int | None = None,
        series_position: str = "first",
    ) -> None:
        """Configure the generator; see class docstring for the parameter meaning."""
        super().__init__(
            name=name,
            cache_dir=cache_dir,
            seed=seed,
            max_samples=max_samples,
            sample_strategy=sample_strategy,
            max_series_length=max_series_length,
            series_position=series_position,
        )
        if packet_len > length:
            raise ValueError(
                f"packet_len ({packet_len}) must not exceed length ({length})."
            )
        self.n_samples = n_samples
        self.length = length
        self.class_freqs = tuple(class_freqs)
        self.packet_len = packet_len
        self.n_global_distractors = n_global_distractors
        self.n_local_distractors = n_local_distractors
        self.max_freq = max_freq

    def _cache_key(self) -> str:
        """
        Return the fixed cache folder name (``self.name``, i.e. ``freq_shapes``).

        The dataset is a shipped fixed snapshot, not a parametric generation, so the
        cache key is stable and independent of the (now inert) ``init_params``. This
        makes :meth:`cache_subdir` resolve to the committed
        ``…/synthetic/freq_shapes/`` pre-split layout the runner loads.
        """
        return self.name

    def generate_dataset(self) -> tuple:
        """
        Disabled placeholder — the programmatic generator is turned off for now.

        ``freq_shapes`` ships a fixed pre-split dataset committed under the synthetic
        cache dir; the runner loads it directly (see the class docstring). This stub
        exists only to satisfy the :class:`SyntheticDataset` ABC.

        Raises
        ------
        NotImplementedError
            Always — set ``cache_path`` so the runner loads the shipped pre-split
            layout instead of generating.
        """
        raise NotImplementedError(
            "freq_shapes ships a fixed pre-split dataset committed under the "
            "synthetic cache dir (cache/datasets/synthetic/freq_shapes/); the "
            "programmatic generator is disabled for now. Set `cache_path` so the "
            "runner loads the pre-split layout via load_saved_splits."
        )


# ── Synthetic-dataset registry ──────────────────────────────────────────────────

SYNTHETIC_DATASETS: dict[str, type] = {
    "freq_shapes": FreqShapesDataset,
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
        Passed through to :class:`UcrUeaDataset` when *path* is ``None``.
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
        return LocalDataset(
            path,
            name=name,
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
