"""File-format loaders (``FORMAT_LOADERS``) and JSON / metadata I/O helpers."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── File-format loaders ───────────────────────────────────────────────────────
# Each loader takes a directory Path and returns (data, labels, metadata).
# data  : np.ndarray of shape (n_samples, n_channels, n_timesteps)
# labels: pd.Series
# metadata: pd.DataFrame | None


def _load_numpy(directory: Path) -> tuple:
    """
    Load a single-file numpy dataset.

    Expects one ``data*.npy`` file and one ``label*.json`` file in *directory*.
    """
    files = [f for f in directory.iterdir() if f.is_file()]
    data = None
    labels = None
    metadata = None

    for f in files:
        if "data" in f.stem and f.suffix == ".npy":
            data = np.load(f)
        elif "label" in f.stem and f.suffix == ".json":
            labels, metadata = load_json(f)

    # A separate metadata*.json (if present) takes precedence over any metadata
    # columns embedded in the labels file.
    meta_file = next(
        (f for f in files if "metadata" in f.stem and f.suffix == ".json"), None
    )
    if meta_file is not None:
        metadata = load_metadata(meta_file)

    if data is None or labels is None:
        raise ValueError(
            "Could not find data (data*.npy) and label (label*.json) files "
            f"in: {directory}"
        )
    return data, labels, metadata


def _load_arff(directory: Path) -> tuple:
    raise NotImplementedError("ARFF format loader is not yet implemented.")


def _load_csv(directory: Path) -> tuple:
    raise NotImplementedError("CSV format loader is not yet implemented.")


def _load_wfdb(directory: Path) -> tuple:
    raise NotImplementedError("WFDB format loader is not yet implemented.")


FORMAT_LOADERS = {
    "numpy": _load_numpy,
    "arff": _load_arff,
    "csv": _load_csv,
    "wfdb": _load_wfdb,
}

# ── Shared utility ────────────────────────────────────────────────────────────


def load_json(path: Path) -> tuple:
    """
    Load a labels JSON file.

    The file must contain a list of records.  A ``"label"`` key is required;
    all other keys are treated as metadata columns.

    Parameters
    ----------
    path : Path
        Path to the ``.json`` file.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame | None]
        ``(labels, metadata)`` — metadata is ``None`` when no extra columns exist.

    Raises
    ------
    ValueError
        If no ``"label"`` key is found.
    """
    loaded = pd.read_json(path, orient="records")
    metadata = pd.DataFrame()
    labels = None

    for col, series in loaded.items():
        if col == "label":
            labels = series
        else:
            metadata[col] = series.values

    if labels is None:
        raise ValueError(
            f"No 'label' key found in {path}. The file must be a list of "
            "dicts with at least a 'label' field."
        )

    return labels, (None if metadata.empty else metadata)


def load_metadata(path: Path) -> pd.DataFrame | None:
    """
    Load a standalone metadata JSON file into a one-row-per-sample DataFrame.

    Accepts two shapes:

    - a **records list** ``[{...}, {...}, ...]`` — the framework's own
      ``orient="records"`` output, and
    - an **index-keyed dict** ``{"0": {...}, "1": {...}, ...}`` — sorted by
      integer key into sample order.

    Nested values (lists / dicts, e.g. per-class ground-truth regions) are kept
    as-is in object columns.

    Parameters
    ----------
    path : Path
        Path to the metadata ``.json`` file.

    Returns
    -------
    pd.DataFrame | None
        One row per sample, or ``None`` if the file is empty.
    """
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        records = [raw[k] for k in sorted(raw, key=int)]
    else:
        records = list(raw)
    if not records:
        return None
    return pd.DataFrame.from_records(records)
