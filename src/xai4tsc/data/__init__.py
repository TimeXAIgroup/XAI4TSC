"""
Dataset loading and splitting for time series classification.

Submodules:

- :mod:`~xai4tsc.data.datasets` — the :func:`load_dataset` factory, the
  :class:`UcrUeaDataset` / :class:`LocalDataset` / :class:`SyntheticDataset`
  classes, and the ``SYNTHETIC_DATASETS`` registry.
- :mod:`~xai4tsc.data.data_loaders` — file-format loaders (``FORMAT_LOADERS``)
  and JSON / metadata I/O helpers.
- :mod:`~xai4tsc.data.base` — the :class:`DatasetBase` ABC (splitting, label
  encoding, save/load of pre-split data) and :class:`MultiHotLabelEncoder`, the
  label-powerset encoder for multi-hot labels.

The factory function :func:`load_dataset` dispatches to
:class:`UcrUeaDataset`, :class:`LocalDataset`, or :class:`SyntheticDataset`
based on the arguments supplied.
"""

from .base import DatasetBase, MultiHotLabelEncoder, class_names, decode_labels
from .data_loaders import FORMAT_LOADERS, load_json, load_metadata
from .datasets import (
    SYNTHETIC_DATASETS,
    LocalDataset,
    SyntheticDataset,
    SynthMultilabelFreqDataset,
    UcrUeaDataset,
    load_dataset,
    register_synthetic_dataset,
)

__all__ = [
    "FORMAT_LOADERS",
    "SYNTHETIC_DATASETS",
    "DatasetBase",
    "LocalDataset",
    "MultiHotLabelEncoder",
    "SynthMultilabelFreqDataset",
    "SyntheticDataset",
    "UcrUeaDataset",
    "class_names",
    "decode_labels",
    "load_dataset",
    "load_json",
    "load_metadata",
    "register_synthetic_dataset",
]
