"""Unit tests for src/xai4tsc/data/ — no I/O beyond tmp_path, no network."""

import hashlib
import json
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from xai4tsc.data import load_dataset
from xai4tsc.data.base import (
    DatasetBase,
    MultiHotLabelEncoder,
    _class_name_map,
    _ensure_bct,
    _impute_data,
    _make_encoder,
    _split_dataset,
)
from xai4tsc.data.data_loaders import load_json, load_metadata
from xai4tsc.data.datasets import (
    SYNTHETIC_DATASETS,
    LocalDataset,
    SyntheticDataset,
    SynthMultilabelFreqDataset,
    UcrUeaDataset,
    _pad_variable_length,
    _powerset,
    register_synthetic_dataset,
)

# ── helpers ───────────────────────────────────────────────────────────────────


class _Dummy(DatasetBase):
    """Minimal concrete subclass for testing DatasetBase methods."""

    def __init__(self, data, labels):
        self._data = data
        self._labels = labels
        self.name = "dummy"

    def load(self):
        return self._data, self._labels, None


def _make_data(n=100, channels=1, length=50, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.random((n, channels, length)).astype(np.float32)
    labels = rng.integers(0, n_classes, n)
    return data, labels


def _write_numpy_dataset(path, n=30, channels=1, length=20, n_classes=2):
    data, labels = _make_data(n, channels, length, n_classes)
    np.save(path / "data.npy", data)
    records = [{"label": int(lbl)} for lbl in labels]
    (path / "labels.json").write_text(json.dumps(records))
    return data, labels


def _make_nested_univ(lengths, n_channels=1, seed=0):
    """
    Build a sktime ``nested_univ`` DataFrame with variable-length cells.

    One column per channel, one row per entry in *lengths*; each cell is a
    ``pd.Series`` whose length is taken from *lengths* (so rows differ in length).
    """
    rng = np.random.default_rng(seed)
    columns = {}
    for c in range(n_channels):
        columns[f"dim_{c}"] = [
            pd.Series(rng.standard_normal(length)) for length in lengths
        ]
    return pd.DataFrame(columns)


# ── _split_dataset ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_split_no_val_sizes():
    data, labels = _make_data(100)
    splits, _ = _split_dataset(data, labels, train_split=0.8, val_split=0.0)
    assert len(splits) == 2
    assert len(splits[0][0]) + len(splits[1][0]) == 100
    assert abs(len(splits[0][0]) - 80) <= 2


@pytest.mark.unit
def test_split_with_val_sizes():
    data, labels = _make_data(100)
    splits, _ = _split_dataset(data, labels, train_split=0.8, val_split=0.1)
    assert len(splits) == 3
    assert sum(len(s[0]) for s in splits) == 100
    # Realized proportions must honor the config: train=80, test=10, val=10.
    # Guards against the holdout being mis-divided between val and test.
    train, test, val = (len(s[0]) for s in splits)
    assert abs(train - 80) <= 2
    assert abs(test - 10) <= 2
    assert abs(val - 10) <= 2


@pytest.mark.unit
def test_split_reproducibility():
    data, labels = _make_data(100)
    splits1, _ = _split_dataset(data, labels, random_state=42)
    splits2, _ = _split_dataset(data, labels, random_state=42)
    np.testing.assert_array_equal(splits1[0][0], splits2[0][0])
    np.testing.assert_array_equal(splits1[1][0], splits2[1][0])


@pytest.mark.unit
def test_split_different_seeds_differ():
    data, labels = _make_data(100)
    splits1, _ = _split_dataset(data, labels, random_state=42)
    splits2, _ = _split_dataset(data, labels, random_state=99)
    assert not np.array_equal(splits1[0][0], splits2[0][0])


# The target contract: whatever `encode` reads, the split carries 1-D int64 class
# indices, and the encoder is kept only to map those indices back to class names.
# Before this, the two sklearn FEATURE encoders leaked their own shapes - a sparse
# (n, K) matrix for onehot, an (n, 1) float column for ordinal - which no
# CrossEntropy target, np.unique() or fancy-indexing consumer could use.
@pytest.mark.unit
@pytest.mark.parametrize("encode", ["label", "onehot", "ordinal", "multihot"])
def test_split_always_yields_1d_int64_class_indices(encode):
    n_classes = 3
    data, labels = _make_data(100, n_classes=n_classes)
    if encode == "multihot":
        # multihot reads multi-hot rows, so give it rows rather than scalars.
        labels = np.eye(n_classes, dtype=int)[np.asarray(labels, dtype=int)]
    splits, encoder = _split_dataset(data, labels, encode=encode)
    y = splits[0][1]
    assert y.ndim == 1, f"{encode} leaked a {y.ndim}-D target"
    assert y.dtype == np.int64
    assert set(np.unique(y)) <= set(range(n_classes))
    # The encoder must still be able to name the classes it collapsed.
    assert _class_name_map(encoder), f"{encode} exposes no index-to-name map"


@pytest.mark.unit
def test_split_encode_onehot_reads_the_hot_index():
    # A one-hot label column resolves to the index of its hot position.
    data, _ = _make_data(6, n_classes=3)
    splits, _ = _split_dataset(
        data,
        np.array([0, 1, 2, 0, 1, 2]),
        encode="onehot",
        train_split=0.5,
        val_split=0.0,
        stratify=False,
    )
    assert set(np.unique(splits[0][1])) <= {0, 1, 2}


@pytest.mark.unit
def test_split_encode_ordinal_without_order_warns_for_string_labels(caplog):
    # Ordinal exists to preserve a class order; lexicographic is the silent trap.
    data, _ = _make_data(6, n_classes=3)
    y = np.array(["low", "medium", "high", "low", "medium", "high"])
    with caplog.at_level("WARNING", logger="xai4tsc.data.base"):
        _, encoder = _split_dataset(
            data, y, encode="ordinal", train_split=0.5, val_split=0.0, stratify=False
        )
    assert "class_order" in caplog.text
    assert _class_name_map(encoder) == {0: "high", 1: "low", 2: "medium"}


@pytest.mark.unit
def test_split_encode_ordinal_honours_class_order():
    data, _ = _make_data(6, n_classes=3)
    y = np.array(["low", "medium", "high", "low", "medium", "high"])
    _, encoder = _split_dataset(
        data,
        y,
        encode="ordinal",
        train_split=0.5,
        val_split=0.0,
        stratify=False,
        class_order=["low", "medium", "high"],
    )
    assert _class_name_map(encoder) == {0: "low", 1: "medium", 2: "high"}


@pytest.mark.unit
def test_split_encode_ordinal_numeric_labels_need_no_order():
    # sklearn refuses non-sorted numeric categories, and sorted is already the
    # meaningful order for numbers, so no class_order is required.
    data, _ = _make_data(6, n_classes=3)
    _, encoder = _split_dataset(
        data,
        np.array([10, 20, 30, 10, 20, 30]),
        encode="ordinal",
        train_split=0.5,
        val_split=0.0,
        stratify=False,
    )
    assert _class_name_map(encoder) == {0: "10", 1: "20", 2: "30"}


# ── save_splits / load_saved_splits ───────────────────────────────────────────


@pytest.mark.unit
def test_save_load_round_trip(tmp_path):
    data, labels = _make_data(50)
    ds = _Dummy(data, labels)
    splits_orig, _ = ds.split(train_split=0.8, val_split=0.0, random_state=42)

    save_dir = tmp_path / "splits"
    ds.save_splits(save_dir)

    ds2 = _Dummy(data, labels)
    splits_loaded, _ = ds2.load_saved_splits(save_dir / "splits", encode="label")

    np.testing.assert_array_equal(splits_orig[0][0], splits_loaded[0][0])
    np.testing.assert_array_equal(splits_orig[1][0], splits_loaded[1][0])
    np.testing.assert_array_equal(splits_orig[0][1], splits_loaded[0][1])


@pytest.mark.unit
def test_save_splits_before_split_raises(tmp_path):
    data, labels = _make_data(50)
    ds = _Dummy(data, labels)
    with pytest.raises(RuntimeError):
        ds.save_splits(tmp_path)


@pytest.mark.unit
def test_load_saved_splits_missing_dir_raises(tmp_path):
    data, labels = _make_data(50)
    ds = _Dummy(data, labels)
    with pytest.raises(ValueError):
        ds.load_saved_splits(tmp_path / "nonexistent", encode="label")


# ── load_json ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_load_json_label_only(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{"label": i} for i in range(5)]))
    labels, metadata = load_json(path)
    assert len(labels) == 5
    assert metadata is None


@pytest.mark.unit
def test_load_json_with_extra_columns(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{"label": i, "patient": f"P{i}"} for i in range(5)]))
    labels, metadata = load_json(path)
    assert len(labels) == 5
    assert metadata is not None
    assert "patient" in metadata.columns


@pytest.mark.unit
def test_load_json_missing_label_key_raises(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{"value": i} for i in range(5)]))
    with pytest.raises(ValueError):
        load_json(path)


# ── LocalDataset ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_local_dataset_name_attribute(tmp_path):
    _write_numpy_dataset(tmp_path)
    ds = LocalDataset(tmp_path, name="my_ecg")
    assert ds.name == "my_ecg"


@pytest.mark.unit
def test_local_dataset_missing_data_file_raises(tmp_path):
    records = [{"label": 0}] * 5
    (tmp_path / "labels.json").write_text(json.dumps(records))
    ds = LocalDataset(tmp_path, name="test")
    with pytest.raises(ValueError):
        ds.load()


@pytest.mark.unit
def test_local_dataset_missing_label_file_raises(tmp_path):
    np.save(tmp_path / "data.npy", np.zeros((5, 1, 10), dtype=np.float32))
    ds = LocalDataset(tmp_path, name="test")
    with pytest.raises(ValueError):
        ds.load()


@pytest.mark.unit
def test_local_dataset_load_returns_bct_and_labels(tmp_path):
    data, labels = _write_numpy_dataset(tmp_path, n=8, channels=1, length=20)
    ds = LocalDataset(tmp_path, name="my_ecg")
    loaded_data, loaded_labels, _ = ds.load()
    # Already (B, C, T) with C < T → returned unchanged.
    assert loaded_data.shape == data.shape
    assert len(loaded_labels) == len(labels)


@pytest.mark.unit
def test_local_dataset_load_transposes_btc_to_bct(tmp_path):
    # Write data as (B, T, C) with T > C to trigger the _ensure_bct heuristic.
    btc = np.zeros((6, 30, 2), dtype=np.float32)
    np.save(tmp_path / "data.npy", btc)
    (tmp_path / "labels.json").write_text(json.dumps([{"label": 0} for _ in range(6)]))
    ds = LocalDataset(tmp_path, name="spectro")
    loaded_data, _, _ = ds.load()
    assert loaded_data.shape == (6, 2, 30)  # transposed to (B, C, T)


@pytest.mark.unit
def test_local_dataset_unknown_format_raises(tmp_path):
    ds = LocalDataset(tmp_path, name="test", data_format="csv")
    # CSV loader is registered but not implemented.
    with pytest.raises(NotImplementedError):
        ds.load()


@pytest.mark.unit
def test_local_dataset_max_samples_restricts_in_split(tmp_path):
    _write_numpy_dataset(tmp_path, n=30, channels=1, length=20)
    ds = LocalDataset(tmp_path, name="test", max_samples=10)
    splits, _ = ds.split(train_split=0.8, val_split=0.0, random_state=0)
    assert sum(len(s[0]) for s in splits) == 10


# ── load_dataset factory ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_load_dataset_with_path_returns_local(tmp_path):
    _write_numpy_dataset(tmp_path)
    ds = load_dataset(name="my_ecg", path=str(tmp_path))
    assert isinstance(ds, LocalDataset)
    assert ds.name == "my_ecg"


@pytest.mark.unit
def test_load_dataset_without_path_returns_ucr():
    ds = load_dataset(name="GunPoint", download=False)
    assert isinstance(ds, UcrUeaDataset)
    assert ds.name == "GunPoint"


@pytest.mark.unit
def test_load_dataset_with_path_tolerates_runner_only_kwargs(tmp_path):
    # The runner passes `download` and `use_predefined_splits` on every
    # non-synthetic dataset, but LocalDataset accepts neither. They must be
    # stripped here, as the synthetic and UCR branches already do for their own
    # inapplicable keys, or `path:` is unusable from a config.
    _write_numpy_dataset(tmp_path)
    ds = load_dataset(
        name="my_ecg",
        path=str(tmp_path),
        download=True,
        use_predefined_splits=False,
        data_format="numpy",
    )
    assert isinstance(ds, LocalDataset)


@pytest.mark.unit
def test_load_dataset_with_path_warns_on_use_predefined_splits(tmp_path, caplog):
    # `use_predefined_splits` is a UCR/UEA concept. Asking for it on a local
    # dataset cannot work, so say so instead of dropping it silently.
    _write_numpy_dataset(tmp_path)
    with caplog.at_level("WARNING", logger="xai4tsc.data.datasets"):
        ds = load_dataset(name="my_ecg", path=str(tmp_path), use_predefined_splits=True)
    assert isinstance(ds, LocalDataset)
    assert "use_predefined_splits" in caplog.text
    assert "is_presplit" in caplog.text


@pytest.mark.unit
def test_load_dataset_forwards_restriction_kwargs(tmp_path):
    _write_numpy_dataset(tmp_path)
    ds = load_dataset(name="my_ecg", path=str(tmp_path), max_samples=5)
    assert ds._max_samples == 5


# ── _pad_variable_length ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_pad_variable_length_zero_pads_to_max():
    nested = _make_nested_univ([3, 5, 4])
    out = _pad_variable_length(nested)
    assert out.shape == (3, 1, 5)
    assert out.dtype == np.float32
    # The length-3 row is right-padded with zeros in the last two positions.
    np.testing.assert_array_equal(out[0, 0, 3:], np.zeros(2, dtype=np.float32))
    # The longest row (length 5) is fully populated (no trailing zeros required).
    assert np.count_nonzero(out[1, 0]) == 5


@pytest.mark.unit
def test_pad_variable_length_multichannel():
    nested = _make_nested_univ([4, 6], n_channels=2)
    out = _pad_variable_length(nested)
    assert out.shape == (2, 2, 6)
    # Shorter row padded on both channels.
    np.testing.assert_array_equal(out[0, 0, 4:], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(out[0, 1, 4:], np.zeros(2, dtype=np.float32))


# ── _ensure_bct ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_ensure_bct_transposes_when_t_gt_c(caplog):
    # (B, T, C) with T=10 > C=3 → transposed to (B, C, T) with a warning.
    btc = np.zeros((2, 10, 3), dtype=np.float32)
    with caplog.at_level("WARNING", logger="xai4tsc.data.base"):
        out = _ensure_bct(btc)
    assert out.shape == (2, 3, 10)
    # The message must flag the ambiguity and name the way to resolve it, rather
    # than matching one particular wording.
    assert "transpos" in caplog.text.lower()
    assert "layout" in caplog.text.lower()


@pytest.mark.unit
def test_ensure_bct_no_transpose_when_already_bct():
    # C=3 < T=10 already in (B, C, T) order → returned unchanged.
    bct = np.arange(2 * 3 * 10, dtype=np.float32).reshape(2, 3, 10)
    out = _ensure_bct(bct)
    assert out.shape == (2, 3, 10)
    np.testing.assert_array_equal(out, bct)


@pytest.mark.unit
def test_ensure_bct_no_transpose_when_c_eq_t(caplog):
    # Square middle/last axes (5 == 5) must not trigger the heuristic.
    square = np.zeros((2, 5, 5), dtype=np.float32)
    with caplog.at_level("WARNING", logger="xai4tsc.data.base"):
        out = _ensure_bct(square)
    assert out.shape == (2, 5, 5)
    assert "transposing" not in caplog.text.lower()


# ── _impute_data ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_impute_data_replaces_nan_with_column_mean():
    # SimpleImputer("mean") works column-wise over the reshaped (B*C, T) view,
    # so a NaN at timestep t is filled with the mean of that timestep column.
    data = np.array(
        [[[1.0, np.nan, 3.0, 4.0]], [[3.0, 2.0, np.nan, 6.0]]], dtype=np.float32
    )
    out = _impute_data(data)
    assert out.shape == data.shape
    assert out.dtype == np.float32
    assert not np.any(np.isnan(out))
    # Column 1: only 2.0 present → imputed value is 2.0.
    assert out[0, 0, 1] == pytest.approx(2.0)
    # Column 2: only 3.0 present → imputed value is 3.0.
    assert out[1, 0, 2] == pytest.approx(3.0)


# ── _make_encoder ───────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("encode", "expected_type"),
    [
        ("label", LabelEncoder),
        ("onehot", OneHotEncoder),
        ("ordinal", OrdinalEncoder),
        ("unknown_key", LabelEncoder),  # anything unrecognised falls back to label
    ],
)
def test_make_encoder_returns_expected_types(encode, expected_type):
    assert isinstance(_make_encoder(encode), expected_type)


# ── LocalDataset unknown (unregistered) format ──────────────────────────────────


@pytest.mark.unit
def test_local_dataset_unregistered_format_raises_value_error(tmp_path):
    # A format key absent from FORMAT_LOADERS raises ValueError (vs. the
    # registered-but-unimplemented "csv" which raises NotImplementedError).
    ds = LocalDataset(tmp_path, name="test", data_format="bogus")
    with pytest.raises(ValueError, match="Unknown data_format"):
        ds.load()


# ── MultiHotLabelEncoder ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_make_encoder_multihot_returns_multihot_encoder():
    assert isinstance(_make_encoder("multihot"), MultiHotLabelEncoder)


@pytest.mark.unit
def test_multihot_encoder_maps_combos_to_indices_and_inverts():
    y = np.array([[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]])
    enc = MultiHotLabelEncoder().fit(y)
    # classes_ are the sorted unique combos; 3 distinct here.
    assert enc.classes_.tolist() == [[0, 0], [0, 1], [1, 1]]
    idx = enc.transform(y)
    assert idx.dtype == np.int64
    assert idx.tolist() == [0, 2, 1, 2, 0]
    # inverse_transform round-trips.
    np.testing.assert_array_equal(enc.inverse_transform(idx), y)


@pytest.mark.unit
def test_multihot_encoder_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D multi-hot"):
        MultiHotLabelEncoder().fit(np.array([0, 1, 0]))


@pytest.mark.unit
def test_split_dataset_multihot_logs_class_names(caplog):
    data = np.zeros((8, 1, 10), dtype=np.float32)
    labels = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    with caplog.at_level("INFO", logger="xai4tsc.data.base"):
        splits, encoder = _split_dataset(
            data, labels, None, train_split=0.5, val_split=0.0, encode="multihot"
        )
    assert isinstance(encoder, MultiHotLabelEncoder)
    assert len(encoder.classes_) == 8
    # 1-D int indices feed CrossEntropy and num_classes = len(classes_).
    assert splits[0][1].dtype == np.int64
    assert "Label classes" in caplog.text


# ── metadata loader: both layouts ───────────────────────────────────────────────


@pytest.mark.unit
def test_load_metadata_records_list(tmp_path):
    recs = [{"ground_truth": {"0": [{"pos": 1}]}}, {"ground_truth": {}}]
    f = tmp_path / "metadata.json"
    f.write_text(json.dumps(recs))
    df = load_metadata(f)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.iloc[0]["ground_truth"] == {"0": [{"pos": 1}]}


@pytest.mark.unit
def test_load_metadata_index_keyed_dict_sorted_by_int_key(tmp_path):
    # The user's sample layout: dict keyed by sample index (string keys).
    raw = {"1": {"v": "b"}, "0": {"v": "a"}, "10": {"v": "c"}}
    f = tmp_path / "metadata.json"
    f.write_text(json.dumps(raw))
    df = load_metadata(f)
    assert df["v"].tolist() == ["a", "b", "c"]  # sorted 0, 1, 10 (not lexicographic)


@pytest.mark.unit
def test_load_numpy_prefers_separate_metadata_file(tmp_path):
    np.save(tmp_path / "data.npy", np.zeros((2, 1, 5), dtype=np.float32))
    (tmp_path / "labels.json").write_text(json.dumps([{"label": 0}, {"label": 1}]))
    (tmp_path / "metadata.json").write_text(
        json.dumps([{"region": "x"}, {"region": "y"}])
    )
    ds = LocalDataset(tmp_path, name="meta")
    _, _, metadata = ds.load()
    assert metadata["region"].tolist() == ["x", "y"]


# ── SyntheticDataset / SynthMultilabelFreqDataset ───────────────────────────────


@pytest.mark.unit
def test_load_dataset_dispatches_to_synthetic_not_ucr():
    ds = load_dataset(
        "Synth_Multilabel_Freq",
        num_samples_per_pattern=2,
        sample_length=64,
        frequency_window_length=16,
        seed=1,
    )
    assert isinstance(ds, SynthMultilabelFreqDataset)
    assert not isinstance(ds, UcrUeaDataset)


class _SynthStub(SyntheticDataset):
    """
    Minimal concrete SyntheticDataset for exercising the ABC (caching/splits).

    The shipped dataset overrides split() to apply its own partition, so the ABC's
    own cache / split-roundtrip behaviour is covered with this tiny stand-in.
    """

    _COMBOS = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )

    def __init__(self, n=16, t=8, multihot=False, **kw):
        kw.setdefault("name", "_synth_stub")
        super().__init__(**kw)
        self._n = n
        self._t = t
        self._multihot = multihot

    def generate_dataset(self):
        rng = np.random.default_rng(self.seed)
        data = rng.standard_normal((self._n, 1, self._t)).astype(np.float32)
        if self._multihot:
            labels = self._COMBOS[np.arange(self._n) % len(self._COMBOS)]
            meta = pd.DataFrame.from_records(
                [
                    {
                        "ground_truth": {0: [{"channel": 0, "freq": 7}]},
                        "non_discriminative": [],
                    }
                    for _ in range(self._n)
                ]
            )
        else:
            labels = (np.arange(self._n) % 2).astype(np.int64)
            meta = pd.DataFrame.from_records([{"info": i} for i in range(self._n)])
        return data, labels, meta


# The published dataset this generator is a port of, pinned by digest rather than
# by a committed copy of the 11 MB it used to ship as. Covers all three splits, so
# it pins the partition as well as the generation; `n` is stated separately so a
# shape change fails with a readable message instead of a hash mismatch.
# Source: https://doi.org/10.5281/zenodo.20744119
_PUBLISHED = {
    "train": {
        "n": 2800,
        "data": "7661c5d39dcfc68de8c7080dcc79456b933bc432d8f7a742ae53656ac73a725b",
        "labels": "3cea107e41ad3fe37cdc1fbbf3f8e725cb1730bdb48eb499508939d55b4aee99",
        "metadata": "0b38370e06502fda337c675dda142c337b1915eda2ed4e6d1f123c371218030d",
    },
    "test": {
        "n": 600,
        "data": "30f31ce7590964ea9fe597d75cd9b375c5f165a8136ef626a0f30176be32049b",
        "labels": "0d7d008f97f93c4e665e85944147d266b92b26b07b8eb39b1d3d19e27a54af17",
        "metadata": "a6883b881164b9f110b6804c9406a000082590c514ccd69554320f580687479a",
    },
    "val": {
        "n": 600,
        "data": "1d3f56fcd8fe837b1be7b08cc97f55821d8d7610194e398c061dd43186f53659",
        "labels": "f7d42e41a9a15b71358d4cf658d50b60046d4921c982e36017cc9cede6efd31c",
        "metadata": "3653e1cc47d245ae8da930c9a054a29032f3420b82367b7879b5c913488dc04a",
    },
}


def _sha(payload):
    """Hash raw bytes, or a JSON-serialisable object in a canonical form."""
    if not isinstance(payload, bytes):
        payload = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _published_form(row):
    """
    Strip a metadata row to the keys the published file carries.

    The generator adds ``sampling_rate`` to the row and ``classes`` to each region
    at load time; neither exists upstream, so both are dropped before hashing.
    """
    return {
        key: [
            {k: v for k, v in region.items() if k != "classes"} for region in row[key]
        ]
        for key in ("ground_truth", "non_discriminative")
    }


@pytest.fixture(scope="module")
def published_splits():
    """Generate the dataset at its shipped defaults, once for the module."""
    ds = SynthMultilabelFreqDataset()
    splits, encoder = ds.split(train_split=0.7, val_split=0.15, encode="multihot")
    return ds, splits, encoder


@pytest.mark.unit
def test_generator_reproduces_the_published_dataset(published_splits):
    """
    The defaults must reproduce the dataset this generator was ported from.

    This is the whole justification for generating rather than shipping: if the
    port drifts, the framework quietly stops evaluating the published benchmark
    while still claiming to. Four details in the generator are load-bearing for
    this (draw order, absolute-time phase, rectangular envelope, single RNG), and
    none of them is visible from the output - this test is what makes them
    testable.
    """
    _ds, splits, encoder = published_splits
    for name, (x, y, metadata) in zip(("train", "test", "val"), splits, strict=True):
        expected = _PUBLISHED[name]
        assert x.shape == (expected["n"], 1, 500), name
        labels = [{"label": [int(v) for v in encoder.classes_[i]]} for i in y]
        assert (
            _sha(np.ascontiguousarray(x, dtype=np.float32).tobytes())
            == (expected["data"])
        ), f"{name} signals differ from the published dataset"
        assert _sha(labels) == expected["labels"], f"{name} labels differ"
        assert _sha([_published_form(r) for r in metadata]) == expected["metadata"], (
            f"{name} ground-truth metadata differs"
        )


@pytest.mark.unit
def test_recorded_frequency_names_the_bin_the_class_signal_occupies(published_splits):
    """
    A recorded frequency must identify the coefficient bin carrying its class.

    The regions are stored in Hz, so reading one directly as a bin index lands
    ``T / fs`` bins away - on frequencies the class signal never occupies, which
    scores as chance however good the attribution is. Deriving the bin from the
    signals rather than from the schema is what catches that, and asserting the
    naive reading carries *no* signal is what documents why the unit matters.
    """
    _ds, splits, encoder = published_splits
    x, y, metadata = splits[0]
    signal_length = x.shape[-1]
    rate = metadata[0]["sampling_rate"]
    spectra = np.abs(np.fft.rfft(x[:, 0, :], axis=1))
    combos = np.array([encoder.classes_[i] for i in y])

    for bit, freq in enumerate((7, 19, 37)):
        present = combos[:, bit] == 1
        correct = round(freq * signal_length / rate)
        ratio_at_correct = (
            spectra[present, correct].mean() / spectra[~present, correct].mean()
        )
        ratio_at_naive = spectra[present, freq].mean() / spectra[~present, freq].mean()
        assert ratio_at_correct > 5, (
            f"{freq} Hz should dominate bin {correct} when its label bit is set, "
            f"but the on/off amplitude ratio is only {ratio_at_correct:.2f}"
        )
        assert ratio_at_naive < 2, (
            f"bin {freq} unexpectedly carries class information "
            f"(ratio {ratio_at_naive:.2f}); the Hz-as-bin confusion this guards "
            "against would no longer be detectable"
        )


@pytest.mark.unit
def test_split_is_stratified_and_metadata_stays_aligned(published_splits):
    """Each split holds every class in proportion, with its ground truth intact."""
    _ds, splits, encoder = published_splits
    for _x, y, metadata in splits:
        counts = np.bincount(y, minlength=len(encoder.classes_))
        assert set(counts.tolist()) == {len(y) // len(encoder.classes_)}
        assert len(metadata) == len(y)
    # A region's class scope must agree with the label bit it stands for, which is
    # only checkable because metadata and labels are still row-aligned.
    _x, y, metadata = splits[1]
    for target, row in zip(y, metadata, strict=True):
        combo = encoder.classes_[target]
        for region in row["ground_truth"]:
            bit = (7, 19, 37).index(region["freq"])
            assert int(combo[bit]) == 1 or target not in region["classes"]


@pytest.mark.unit
def test_package_api_generates_and_splits(tmp_path):
    """
    The documented package call must work, which it could not before.

    `load_dataset(name, cache_dir=...).split(...)` used to raise
    NotImplementedError and tell a package user to set a runner config key, so the
    only shipped synthetic dataset was reachable from the runner alone.
    """
    ds = load_dataset(
        "Synth_Multilabel_Freq",
        cache_dir=tmp_path,
        num_samples_per_pattern=8,
        sample_length=64,
        sampling_rate=32,
        frequency_window_length=16,
        discriminative_frequencies=(3, 7, 11),
    )
    splits, encoder = ds.split(train_split=0.7, val_split=0.15, encode="multihot")

    assert len(splits) == 3
    assert sum(len(y) for _, y, _ in splits) == 64
    assert all(x.shape[1:] == (1, 64) for x, _, _ in splits)
    assert isinstance(encoder, MultiHotLabelEncoder)
    assert len(encoder.classes_) == 8
    # Cached as a raw blob, so the second call does not regenerate.
    assert SyntheticDataset.is_raw_layout(ds.cache_subdir(tmp_path))


@pytest.mark.unit
def test_generation_is_cached_and_not_repeated(tmp_path):
    calls = {"n": 0}
    real = SynthMultilabelFreqDataset.generate_dataset

    ds = SynthMultilabelFreqDataset(
        cache_dir=tmp_path,
        num_samples_per_pattern=2,
        sample_length=64,
        frequency_window_length=16,
    )

    def counting(self):
        calls["n"] += 1
        return real(self)

    ds.generate_dataset = counting.__get__(ds, SynthMultilabelFreqDataset)
    first = ds.load()[0]
    second = ds.load()[0]
    assert calls["n"] == 1
    np.testing.assert_array_equal(first, second)


@pytest.mark.unit
def test_cache_key_follows_the_generation_parameters():
    """A parameter change must land in a different cache folder."""
    base = SynthMultilabelFreqDataset()
    other = SynthMultilabelFreqDataset(sample_length=256)
    assert base._cache_key() != other._cache_key()
    assert base.cache_subdir("/tmp/c").parts[-2] == "synthetic"
    assert base._cache_key().startswith("Synth_Multilabel_Freq_s42")
    # The partition slices the blob rather than changing it, so it must not key it.
    assert base._cache_key() == SynthMultilabelFreqDataset(train_ratio=0.5)._cache_key()


@pytest.mark.unit
def test_configured_split_fractions_are_reported_not_obeyed(caplog):
    """
    A train_split the dataset cannot honour must say so.

    This dataset owns its partition, so silently ignoring the configured fractions
    would leave a user editing them with no effect and no explanation.
    """
    ds = SynthMultilabelFreqDataset(
        num_samples_per_pattern=8, sample_length=32, frequency_window_length=8
    )
    with caplog.at_level("WARNING", logger="xai4tsc.data.datasets"):
        splits, _ = ds.split(train_split=0.9, val_split=0.05, encode="multihot")
    assert "supplies its own partition" in caplog.text
    # The partition is the dataset's own regardless of what was asked for: 0.7 of
    # 64 samples, then the remaining 19 divided in [train, test, val] order.
    assert [len(y) for _, y, _ in splits] == [45, 9, 10]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"val_split": 0.1}, "val_split=0.1"),
        ({"train_split": 0.7}, "train_split=0.7"),
        ({"stratify": False}, "stratify=False"),
    ],
)
def test_predefined_splits_report_the_arguments_they_ignore(kwargs, expected, caplog):
    """
    Arguments the archive partition cannot honour must say so.

    The official _TRAIN/_TEST files are used as-is, so a configured fraction has
    no effect. Dropping it silently is what let the shipped example ask for a
    validation split and train without one.
    """
    ds = UcrUeaDataset("GunPoint", use_predefined_splits=True)

    with caplog.at_level("WARNING", logger="xai4tsc.data.datasets"):
        ds._report_ignored_split_args(
            kwargs.get("train_split", 0.8),
            kwargs.get("val_split", 0.0),
            kwargs.get("stratify", True),
        )

    assert expected in caplog.text
    assert "no validation set" in caplog.text
    # The remedy belongs in the message, not just in the docstring.
    assert "use_predefined_splits=False" in caplog.text


@pytest.mark.unit
def test_predefined_splits_stay_quiet_when_nothing_was_asked_for(caplog):
    # Warning on the defaults would make every official-split run noisy, and
    # nothing was actually ignored.
    ds = UcrUeaDataset("GunPoint", use_predefined_splits=True)

    with caplog.at_level("WARNING", logger="xai4tsc.data.datasets"):
        ds._report_ignored_split_args(0.8, 0.0, True)

    assert caplog.text == ""


@pytest.mark.unit
def test_sample_restriction_keeps_metadata_aligned():
    """
    max_samples must drop metadata rows with the samples they describe.

    The base helper restricts the arrays only, which would leave the ground truth
    describing samples that are no longer there - every mask silently wrong.
    """
    ds = SynthMultilabelFreqDataset(
        num_samples_per_pattern=8,
        sample_length=32,
        frequency_window_length=8,
        max_samples=16,
    )
    splits, _ = ds.split(train_split=0.7, val_split=0.15, encode="multihot")
    assert sum(len(y) for _, y, _ in splits) == 16
    for x, y, metadata in splits:
        assert len(metadata) == len(y) == len(x)


@pytest.mark.unit
def test_truncating_the_series_warns_that_the_ground_truth_no_longer_fits(caplog):
    """
    Regions are positions in the generated series, and truncation does not move them.

    The masks clip rather than crash, so a region past the cut would silently
    describe signal that is no longer there - which is why this says so.
    """
    ds = SynthMultilabelFreqDataset(
        num_samples_per_pattern=8,
        sample_length=64,
        frequency_window_length=16,
        max_series_length=32,
    )
    with caplog.at_level("WARNING", logger="xai4tsc.data.datasets"):
        splits, _ = ds.split(train_split=0.7, val_split=0.15, encode="multihot")
    assert "still describe the full series" in caplog.text
    assert all(x.shape[-1] == 32 for x, _, _ in splits)


@pytest.mark.unit
def test_powerset_orders_by_size_then_value():
    assert _powerset((7, 19, 37)) == [
        (),
        (7,),
        (19,),
        (37,),
        (7, 19),
        (7, 37),
        (19, 37),
        (7, 19, 37),
    ]


@pytest.mark.unit
def test_bursts_overlap_within_the_configured_bounds():
    """Consecutive discriminative bursts must respect overlap_min / overlap_max."""
    ds = SynthMultilabelFreqDataset(num_samples_per_pattern=20)
    _data, _labels, metadata = ds.generate_dataset()
    pairs = 0
    for row in metadata.to_dict("records"):
        regions = row["ground_truth"]
        for earlier, later in pairwise(regions):
            overlap = (earlier["pos"] + earlier["len"] - later["pos"]) / earlier["len"]
            assert 0.25 <= overlap <= 0.5 + 1e-9, row
            pairs += 1
    assert pairs > 0


@pytest.mark.unit
def test_rejects_parameters_it_cannot_generate():
    with pytest.raises(ValueError, match="frequency_window_length"):
        SynthMultilabelFreqDataset(sample_length=32, frequency_window_length=64)
    with pytest.raises(ValueError, match="Nyquist"):
        SynthMultilabelFreqDataset(sampling_rate=20, discriminative_frequencies=(7, 19))
    with pytest.raises(ValueError, match="distinct distractors"):
        SynthMultilabelFreqDataset(
            sampling_rate=10, discriminative_frequencies=(1,), num_full_distractions=8
        )
    with pytest.raises(ValueError, match="below 1"):
        SynthMultilabelFreqDataset(train_ratio=0.9, test_ratio=0.2)


@pytest.mark.unit
def test_loads_a_saved_presplit_layout(tmp_path):
    # A saved layout is still loadable directly, and still gets its regions scoped
    # to the classes each frequency marks.
    ds = SynthMultilabelFreqDataset(cache_dir=tmp_path)
    cache = ds.cache_subdir(tmp_path)
    cache.mkdir(parents=True)
    combos = [[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]]
    for split, n in [("train", 4), ("test", 2), ("val", 2)]:
        np.save(cache / f"{split}_data.npy", np.zeros((n, 1, 16), dtype=np.float32))
        (cache / f"{split}_labels.json").write_text(
            json.dumps([{"label": combos[i % len(combos)]} for i in range(n)])
        )
        region = {"channel": 0, "freq": 7}
        (cache / f"{split}_metadata.json").write_text(
            json.dumps(
                [{"ground_truth": [region], "non_discriminative": []} for _ in range(n)]
            )
        )
    assert SyntheticDataset.is_split_layout(cache)
    splits, encoder = ds.load_saved_splits(cache, encode="multihot")
    assert len(splits) == 3
    assert splits[0][0].shape == (4, 1, 16)
    assert isinstance(encoder, MultiHotLabelEncoder)
    assert "ground_truth" in pd.DataFrame(splits[1][2]).columns
    # The scoped splits are what get_splits() returns, not the unscoped ones.
    assert ds.get_splits()[0] is splits
    # Four combinations here, of which [1,0,1] and [1,1,1] have bit 0 (7 Hz) set.
    scoped = pd.DataFrame(splits[1][2])["ground_truth"].iloc[0][0]
    assert tuple(scoped["classes"]) == (2, 3)


@pytest.mark.unit
def test_synthetic_first_look_cache_skips_regeneration(tmp_path):
    calls = {"n": 0}
    real = _SynthStub.generate_dataset

    def counting(self):
        calls["n"] += 1
        return real(self)

    ds = _SynthStub(cache_dir=tmp_path, n=16, t=8, seed=2)
    ds.generate_dataset = counting.__get__(ds, _SynthStub)

    d1 = ds.load()[0]  # miss → generates + persists
    d2 = ds.load()[0]  # hit → loads from disk, no regen
    assert calls["n"] == 1
    np.testing.assert_array_equal(d1, d2)
    cache = ds.cache_subdir(tmp_path)
    assert {f.name for f in cache.glob("*")} >= {
        "data.npy",
        "labels.json",
        "metadata.json",
    }


@pytest.mark.unit
def test_synthetic_split_round_trip_preserves_metadata_and_classes(tmp_path):
    ds = _SynthStub(cache_dir=tmp_path, n=24, t=8, multihot=True, seed=4)
    _splits, encoder = ds.split(
        train_split=0.75, val_split=0.0, encode="multihot", stratify=False
    )
    ds.save_splits(tmp_path)
    # Separate metadata.json is written, not embedded in labels.json.
    assert (tmp_path / "splits" / "train_metadata.json").exists()

    ds2 = _SynthStub(cache_dir=tmp_path)
    splits2, encoder2 = ds2.load_saved_splits(tmp_path / "splits", encode="multihot")
    # Combo class names survive the round-trip.
    np.testing.assert_array_equal(encoder.classes_, encoder2.classes_)
    # Per-class ground-truth metadata survives.
    assert splits2[0][2] is not None
    # Metadata from the load path is a DataFrame; accept either it or records.
    assert "ground_truth" in pd.DataFrame(splits2[0][2]).columns


@pytest.mark.unit
def test_synthetic_layout_predicates(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    np.save(raw / "data.npy", np.zeros((2, 1, 4), dtype=np.float32))
    assert SyntheticDataset.is_raw_layout(raw)
    assert not SyntheticDataset.is_split_layout(raw)

    split = tmp_path / "split"
    split.mkdir()
    np.save(split / "train_data.npy", np.zeros((2, 1, 4), dtype=np.float32))
    np.save(split / "test_data.npy", np.zeros((2, 1, 4), dtype=np.float32))
    assert SyntheticDataset.is_split_layout(split)
    assert not SyntheticDataset.is_raw_layout(split)

    assert not SyntheticDataset.is_raw_layout(tmp_path / "missing")


@pytest.mark.unit
def test_register_synthetic_dataset_roundtrip():
    class _Tiny(SyntheticDataset):
        def generate_dataset(self):
            return np.zeros((4, 1, 8), dtype=np.float32), np.zeros(4, dtype=int), None

    try:
        register_synthetic_dataset("_tiny_test", _Tiny)
        assert SYNTHETIC_DATASETS["_tiny_test"] is _Tiny
        assert isinstance(load_dataset("_tiny_test"), _Tiny)
    finally:
        SYNTHETIC_DATASETS.pop("_tiny_test", None)


# ── _ensure_bct layout ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_ensure_bct_layout_bct_keeps_c_gt_t_data_intact():
    # The case "auto" corrupts: genuinely (B, C, T) with more channels than
    # timesteps, e.g. a spectrogram.
    arr = np.zeros((4, 40, 20))
    out = _ensure_bct(arr, "bct")
    assert out.shape == (4, 40, 20)
    assert out is arr


@pytest.mark.unit
def test_ensure_bct_layout_bct_emits_no_warning(caplog):
    with caplog.at_level("WARNING", logger="xai4tsc.data.base"):
        _ensure_bct(np.zeros((4, 40, 20)), "bct")
    assert caplog.text == ""


@pytest.mark.unit
def test_ensure_bct_layout_btc_always_transposes():
    # Even when axis 1 < axis 2, so the heuristic would have left it alone.
    assert _ensure_bct(np.zeros((4, 20, 40)), "btc").shape == (4, 40, 20)


@pytest.mark.unit
def test_ensure_bct_layout_auto_is_the_default_and_unchanged():
    arr = np.zeros((4, 40, 20))
    assert _ensure_bct(arr).shape == _ensure_bct(arr, "auto").shape == (4, 20, 40)


@pytest.mark.unit
def test_ensure_bct_rejects_unknown_layout():
    with pytest.raises(ValueError, match="Unknown layout"):
        _ensure_bct(np.zeros((4, 2, 8)), "bogus")


@pytest.mark.unit
@pytest.mark.parametrize("layout", ["auto", "bct", "btc"])
def test_local_dataset_honours_layout(tmp_path, layout):
    # Stored as (6, 30, 2): axis 1 > axis 2, so "auto" reads it as (B, T, C).
    _write_numpy_dataset(tmp_path, n=6, channels=30, length=2)
    out = LocalDataset(tmp_path, name="d", layout=layout).load()[0]
    expected = (6, 30, 2) if layout == "bct" else (6, 2, 30)
    assert out.shape == expected


@pytest.mark.unit
@pytest.mark.parametrize("encode", ["label", "onehot", "ordinal"])
def test_save_load_round_trip_per_encode(tmp_path, encode):
    # save_splits persists DECODED labels and load_saved_splits re-encodes, so the
    # codes and the index-to-name map must survive a cache round-trip. Before this,
    # onehot and ordinal raised in save_splits.
    data, labels = _make_data(60, n_classes=3)
    ds = _Dummy(data, labels)
    splits, encoder = ds.split(train_split=0.7, val_split=0.15, encode=encode)

    save_dir = tmp_path / "out"
    ds.save_splits(save_dir)
    reloaded, encoder2 = _Dummy(data, labels).load_saved_splits(
        save_dir / "splits", encode=encode
    )
    for original, restored in zip(splits, reloaded, strict=True):
        np.testing.assert_array_equal(original[1], restored[1])
    assert _class_name_map(encoder2) == _class_name_map(encoder)
