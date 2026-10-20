r"""
Cache utilities for the experiment runner.

All caching decisions live here, never in the package (src/xai4tsc/).

Cache layout
------------
The root cache directory (``cache_path`` from the experiment config) contains
two independent sub-trees:

``{cache_path}/datasets/``
    Raw UCR and UEA archive downloads.  When ``UcrUeaDataset`` is given a
    ``cache_dir``, that path is forwarded to sktime as ``extract_path``.
    sktime stores the ``.ts`` files under
    ``datasets/{dataset_name}/{dataset_name}_TRAIN.ts`` (and ``_TEST.ts``).
    Deleting this sub-tree forces a fresh download on the next run;
    everything else (split cache) remains intact.

``{cache_path}/splits/``
    Processed train / val / test splits ready for training.  Each split is
    stored in its own folder whose name encodes **all** parameters that affect
    the split, so no external metadata file is needed — if the folder exists
    and is non-empty the split is valid.  Deleting this sub-tree forces
    re-parsing from ``.ts`` and re-splitting on the next run.

    Folder naming convention::

        {dataset}_tr{train:.4g}_v{val:.4g}_s{seed}_{encode}[_p][_i][_s{N}{c}][_t{T}{c}]

    Component meanings:

    =========  ============================================================
    Component  Meaning
    =========  ============================================================
    dataset    Human-readable dataset name (e.g. ``ECG5000``)
    tr…        Training fraction (e.g. ``tr0.8``)
    v…         Validation fraction (e.g. ``v0.1``)
    s…         Integer random seed used for the split (e.g. ``s42``)
    encode     Label encoding scheme: ``label``, ``multihot``
    _p         Suffix: variable-length series were zero-padded
    _i         Suffix: NaN values were imputed before splitting
    _s{N}{c}   Suffix: dataset was subsampled to N samples; c = first char
               of the strategy (**r**\ andom / **f**\ irst / **l**\ ast /
               **s**\ tratified)
    _t{T}{c}   Suffix: series were truncated to T timesteps; c = first char
               of the position (**f**\ irst / **l**\ ast)
    =========  ============================================================

    Examples::

        ECG5000_tr0.8_v0.1_s42_label
        GunPoint_tr0.7_v0.1_s0_onehot
        AllGestureWiimoteX_tr0.8_v0.1_s42_label_p
        DodgerLoopDay_tr0.8_v0.1_s42_label_i
        InsectWingbeat_tr0.8_v0.1_s42_label_p_s5000s_t1000f
"""

import hashlib
from pathlib import Path


def get_dataset_cache_dir(cache_path: Path | None) -> Path | None:
    """Return ``cache_path/datasets/``, or ``None`` if caching is disabled."""
    if cache_path is None:
        return None
    return cache_path / "datasets"


def get_split_cache_path(
    cache_path: Path | None,
    dataset_name: str,
    train_split: float,
    val_split: float,
    random_state: int,
    encode: str,
    allow_padding: bool = False,
    allow_imputation: bool = False,
    max_samples: int | None = None,
    sample_strategy: str = "random",
    max_series_length: int | None = None,
    series_position: str = "first",
    stratify: bool = True,
    use_predefined_splits: bool = False,
    data_format: str = "numpy",
    path: str | Path | None = None,
) -> Path | None:
    """
    Return the path for a specific cached split, or ``None`` if disabled.

    The folder name encodes all split parameters so no separate metadata file
    is needed — if the folder exists the split is valid.

    Folder naming convention::

        {dataset_name}_tr{train_split:.4g}_v{val_split:.4g}_s{random_state}_{encode}
        [_p][_i][_s{N}{strategy[0]}][_t{T}{position[0]}](_strat|_rand)
        {dataset_name}[_s{random_state}]_{encode}[_p][_i][_s{N}{strategy[0]}]
        [_t{T}{position[0]}]_official

    The archive partition ignores *train_split*, *val_split* and *stratify*, so
    the ``_official`` form omits the first two rather than claim a split the
    folder does not hold.  It keeps ``_s{random_state}`` only when *max_samples*
    is set, which is the one case where the seed still decides which rows were
    stored.

    Suffixes:
      ``_p``        padding enabled
      ``_i``        imputation enabled
      ``_s{N}{c}``  max_samples=N, strategy first char (r/f/l/s)
      ``_t{T}{c}``  max_series_length=T, position first char (f/l)
      ``_official`` official archive train/test split (exclusive with _strat/_rand)
      ``_strat``    random split with stratification
      ``_rand``     random split without stratification

    Examples::

        ECG5000_label_official
        ECG5000_s42_label_s500r_official
        ECG5000_tr0.8_v0.1_s42_label_strat
        AllGestureWiimoteX_tr0.8_v0.1_s42_label_p_strat
        InsectWingbeat_tr0.8_v0.1_s42_label_p_s5000r_t1000f_rand

    Full path: ``cache_path/splits/{folder_name}/``

    Parameters
    ----------
    cache_path : Path or None
        Root cache directory.  ``None`` disables caching.
    dataset_name : str
        Human-readable dataset identifier used in the folder name.
    train_split : float
        Fraction of data used for training.  Omitted from the folder name when
        *use_predefined_splits* is ``True``, which ignores it.
    val_split : float
        Fraction of data used for validation.  Omitted from the folder name when
        *use_predefined_splits* is ``True``, which ignores it.
    random_state : int
        Seed used for the split.  Encoded in the folder name wherever it shapes
        the stored data: always on a random split, and on predefined splits
        only when *max_samples* is set, since it seeds the sample restriction.
    encode : str
        Label encoding scheme (e.g. ``"label"``, ``"onehot"``).
    allow_padding : bool
        Whether variable-length series were zero-padded before splitting.
    allow_imputation : bool
        Whether NaN values were imputed before splitting.
    max_samples : int | None
        Maximum number of samples used; ``None`` means all samples.
    sample_strategy : str
        Sample selection strategy (encoded as first character).
    max_series_length : int | None
        Maximum series length used; ``None`` means full length.
    series_position : str
        Which end was retained (encoded as first character).
    stratify : bool
        Whether splits were stratified by class label.  Ignored when
        *use_predefined_splits* is ``True``.
    use_predefined_splits : bool
        Whether the official archive ``_TRAIN``/``_TEST`` files were used.
    data_format : str
        Loader used to read the data.  Different loaders can yield different
        arrays from one directory, so a non-default format gets its own entry.
    path : str or Path, optional
        Local dataset directory.  Two datasets can share a *dataset_name* while
        pointing at different directories, so the path is folded into the key as a
        short digest to keep them apart.

    Returns
    -------
    Path or None
        ``cache_path/splits/{folder_name}/``, or ``None`` if caching is disabled.
    """
    if cache_path is None:
        return None
    suffix = ("_p" if allow_padding else "") + ("_i" if allow_imputation else "")
    if max_samples is not None:
        suffix += f"_s{max_samples}{sample_strategy[0]}"
    if max_series_length is not None:
        suffix += f"_t{max_series_length}{series_position[0]}"
    if data_format and data_format != "numpy":
        suffix += f"_f{data_format}"
    if path is not None:
        digest = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:8]
        suffix += f"_d{digest}"
    if use_predefined_splits:
        suffix += "_official"
    elif stratify:
        suffix += "_strat"
    else:
        suffix += "_rand"
    # Predefined splits ignore train_split, val_split and stratify, so they
    # ignore parts of the naming rules. random_state is the exception: it
    # seeds the sample restriction, so it still shapes the stored arrays
    # whenever max_samples is set.
    partition = ""
    if not use_predefined_splits:
        partition = f"_tr{train_split:.4g}_v{val_split:.4g}_s{random_state}"
    elif max_samples is not None:
        partition = f"_s{random_state}"
    folder = f"{dataset_name}{partition}_{encode}{suffix}"
    return cache_path / "splits" / folder
