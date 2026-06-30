"""Unit tests for src/xai4tsc/data/ — no I/O beyond tmp_path, no network."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from xai4tsc.data import load_dataset
from xai4tsc.data.base import (
    DatasetBase,
    MultiHotLabelEncoder,
    _ensure_bct,
    _impute_data,
    _make_encoder,
    _split_dataset,
)
from xai4tsc.data.data_loaders import load_json, load_metadata
from xai4tsc.data.datasets import (
    SYNTHETIC_DATASETS,
    FreqShapesDataset,
    LocalDataset,
    SyntheticDataset,
    UcrUeaDataset,
    _pad_variable_length,
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


@pytest.mark.unit
def test_split_encode_label():
    data, labels = _make_data(100, n_classes=3)
    splits, _ = _split_dataset(data, labels, encode="label")
    assert splits[0][1].ndim == 1


@pytest.mark.unit
def test_split_encode_onehot():
    data, labels = _make_data(100, n_classes=3)
    splits, _ = _split_dataset(data, labels, encode="onehot")
    # OneHotEncoder produces a sparse/dense 2-D array (n_samples, n_classes)
    assert splits[0][1].ndim == 2
    assert splits[0][1].shape[1] == 3


@pytest.mark.unit
def test_split_encode_ordinal():
    data, labels = _make_data(100, n_classes=3)
    splits, _ = _split_dataset(data, labels, encode="ordinal")
    assert splits[0][1].ndim == 2  # OrdinalEncoder produces (n, 1)


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
    assert "transposing" in caplog.text.lower()


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


# ── SyntheticDataset / FreqShapesDataset ────────────────────────────────────────


@pytest.mark.unit
def test_load_dataset_dispatches_to_synthetic_not_ucr():
    ds = load_dataset("freq_shapes", n_samples=16, length=64, packet_len=24, seed=1)
    assert isinstance(ds, FreqShapesDataset)
    assert not isinstance(ds, UcrUeaDataset)


class _SynthStub(SyntheticDataset):
    """
    Minimal concrete SyntheticDataset for exercising the ABC (caching/splits).

    FreqShapesDataset's generator is now a disabled placeholder, so the cache /
    split-roundtrip behaviour of the ABC is covered with this tiny stand-in.
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


@pytest.mark.unit
def test_freq_shapes_generator_disabled():
    # The programmatic generator is a placeholder for now; loading uses the shipped
    # pre-split layout in the synthetic cache dir instead.
    with pytest.raises(NotImplementedError, match="pre-split"):
        FreqShapesDataset(n_samples=8, length=32, packet_len=8).generate_dataset()


@pytest.mark.unit
def test_freq_shapes_cache_key_is_fixed():
    # Fixed key (independent of the now-inert init_params) so cache_subdir lands on
    # the committed cache/datasets/synthetic/freq_shapes/ layout.
    ds = FreqShapesDataset(n_samples=600, length=500, class_freqs=(7, 19, 37))
    assert ds._cache_key() == "freq_shapes"
    assert ds.cache_subdir("/tmp/c").parts[-2:] == ("synthetic", "freq_shapes")


@pytest.mark.unit
def test_freq_shapes_loads_committed_presplit(tmp_path):
    # Fabricate the committed pre-split layout the runner loads (the generator never
    # runs). Multi-hot labels + per-sample flat-list ground-truth metadata round-trip.
    ds = FreqShapesDataset(cache_dir=tmp_path)
    cache = ds.cache_subdir(tmp_path)  # …/synthetic/freq_shapes
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
    # Per-sample flat-list ground-truth metadata survives the round-trip.
    assert "ground_truth" in pd.DataFrame(splits[1][2]).columns


@pytest.mark.unit
def test_freq_shapes_packet_len_exceeds_length_raises():
    with pytest.raises(ValueError, match="packet_len"):
        FreqShapesDataset(length=32, packet_len=64)


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
