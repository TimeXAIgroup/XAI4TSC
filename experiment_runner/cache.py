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
    encode     Label encoding scheme: ``label``, ``onehot``, ``ordinal``
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
) -> Path | None:
    """
    Return the path for a specific cached split, or ``None`` if disabled.

    The folder name encodes all split parameters so no separate metadata file
    is needed — if the folder exists the split is valid.

    Folder naming convention::

        {dataset_name}_tr{train_split:.4g}_v{val_split:.4g}_s{random_state}_{encode}[_p][_i][_s{N}{strategy[0]}][_t{T}{position[0]}](_official|_strat|_rand)

    Suffixes:
      ``_p``        padding enabled
      ``_i``        imputation enabled
      ``_s{N}{c}``  max_samples=N, strategy first char (r/f/l/s)
      ``_t{T}{c}``  max_series_length=T, position first char (f/l)
      ``_official`` official archive train/test split (exclusive with _strat/_rand)
      ``_strat``    random split with stratification
      ``_rand``     random split without stratification

    Examples::

        ECG5000_tr0.8_v0.1_s42_label_official
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
        Fraction of data used for training.
    val_split : float
        Fraction of data used for validation.
    random_state : int
        Seed used for the split, encoded in the folder name.
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
    if use_predefined_splits:
        suffix += "_official"
    elif stratify:
        suffix += "_strat"
    else:
        suffix += "_rand"
    folder = (
        f"{dataset_name}_tr{train_split:.4g}_v{val_split:.4g}"
        f"_s{random_state}_{encode}{suffix}"
    )
    return cache_path / "splits" / folder
