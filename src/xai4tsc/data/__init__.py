"""
Dataset loading and splitting for time series classification.

Submodules:

- :mod:`~xai4tsc.data.datasets` — the :class:`UcrUeaDataset` /
  :class:`LocalDataset` / :class:`SyntheticDataset` classes and the
  ``SYNTHETIC_DATASETS`` catalog.
- :mod:`~xai4tsc.data.data_loaders` — file-format loaders (``FORMAT_LOADERS``)
  and JSON / metadata I/O helpers.
- :mod:`~xai4tsc.data.base` — the :class:`DatasetBase` ABC (splitting, label
  encoding, save/load of pre-split data) and the two encoders for multi-hot
  labels: :class:`MultiHotLabelEncoder`, which keeps one column per output, and
  :class:`PowersetLabelEncoder`, which collapses each combination to one index.

Construct the class for the data at hand: :class:`UcrUeaDataset` for an archive
dataset, :class:`LocalDataset` for files on disk, or a :class:`SyntheticDataset`
subclass such as :class:`SynthMultilabelFreqDataset`.
"""

from .base import (
    DatasetBase,
    MultiHotLabelEncoder,
    PowersetLabelEncoder,
    class_names,
    decode_labels,
)
from .data_loaders import FORMAT_LOADERS, load_json, load_metadata
from .datasets import (
    SYNTHETIC_DATASETS,
    LocalDataset,
    SyntheticDataset,
    SynthMultilabelFreqDataset,
    UcrUeaDataset,
)

__all__ = [
    "FORMAT_LOADERS",
    "SYNTHETIC_DATASETS",
    "DatasetBase",
    "LocalDataset",
    "MultiHotLabelEncoder",
    "PowersetLabelEncoder",
    "SynthMultilabelFreqDataset",
    "SyntheticDataset",
    "UcrUeaDataset",
    "class_names",
    "decode_labels",
    "load_json",
    "load_metadata",
]
