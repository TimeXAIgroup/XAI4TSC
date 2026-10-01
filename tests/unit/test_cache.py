"""Unit tests for experiment_runner/cache.py — pure path/string construction."""

from pathlib import Path

import pytest
from experiment_runner.cache import get_dataset_cache_dir, get_split_cache_path

_ROOT = Path("/tmp/xai4tsc_cache")


def _base_kwargs(**overrides):
    kwargs = dict(
        cache_path=_ROOT,
        dataset_name="GunPoint",
        train_split=0.8,
        val_split=0.1,
        random_state=42,
        encode="label",
    )
    kwargs.update(overrides)
    return kwargs


def _folder_name(**overrides):
    return get_split_cache_path(**_base_kwargs(**overrides)).name


# ── get_dataset_cache_dir ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_get_dataset_cache_dir_none_disables():
    assert get_dataset_cache_dir(None) is None


@pytest.mark.unit
def test_get_dataset_cache_dir_appends_datasets():
    assert get_dataset_cache_dir(_ROOT) == _ROOT / "datasets"


# ── get_split_cache_path ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_get_split_cache_path_none_disables():
    assert get_split_cache_path(**_base_kwargs(cache_path=None)) is None


@pytest.mark.unit
def test_get_split_cache_path_base_case():
    path = get_split_cache_path(**_base_kwargs())
    # Default stratify=True → _strat suffix; lives under splits/.
    assert path == _ROOT / "splits" / "GunPoint_tr0.8_v0.1_s42_label_strat"


@pytest.mark.unit
def test_get_split_cache_path_rand_suffix_when_not_stratified():
    assert _folder_name(stratify=False).endswith("_label_rand")


@pytest.mark.unit
def test_get_split_cache_path_official_overrides_strat_and_rand():
    name = _folder_name(use_predefined_splits=True, stratify=True)
    assert name.endswith("_label_official")
    assert "_strat" not in name and "_rand" not in name


@pytest.mark.unit
def test_get_split_cache_path_padding_and_imputation_suffixes():
    name = _folder_name(allow_padding=True, allow_imputation=True)
    # Order is _p then _i, immediately after the encode token.
    assert name[name.index("_label") :] == "_label_p_i_strat"


@pytest.mark.unit
def test_get_split_cache_path_sample_restriction_suffix():
    name = _folder_name(max_samples=5000, sample_strategy="stratified")
    assert "_s5000s" in name  # N=5000, strategy first char = 's'


@pytest.mark.unit
def test_get_split_cache_path_length_restriction_suffix():
    name = _folder_name(max_series_length=1000, series_position="last")
    assert "_t1000l" in name  # T=1000, position first char = 'l'


@pytest.mark.unit
def test_get_split_cache_path_is_stable_across_calls():
    # The cache key must be deterministic for identical logical params, otherwise
    # cached splits would never be reused.
    first = get_split_cache_path(**_base_kwargs())
    second = get_split_cache_path(**_base_kwargs())
    assert first == second


# ── format / path in the key ──────────────────────────────────────────────────


@pytest.mark.unit
def test_default_format_and_no_path_leave_the_name_unchanged():
    # Tokens are emitted only when non-default, so caches written by earlier
    # versions stay valid.
    assert _folder_name() == _folder_name(data_format="numpy", path=None)


@pytest.mark.unit
def test_non_default_format_changes_the_key():
    csv = _folder_name(data_format="csv")
    assert csv != _folder_name()
    assert "_fcsv" in csv
    assert csv.endswith("_strat")  # the terminal token stays last


@pytest.mark.unit
def test_different_paths_get_different_keys(tmp_path):
    # Two local datasets can share a name while pointing at different folders;
    # without the path in the key one would silently serve the other's data.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    name_a = _folder_name(path=tmp_path / "a")
    name_b = _folder_name(path=tmp_path / "b")
    assert name_a != name_b
    assert name_a != _folder_name()
    assert name_a.endswith("_strat") and name_b.endswith("_strat")


@pytest.mark.unit
def test_path_key_is_stable_across_calls(tmp_path):
    assert _folder_name(path=tmp_path) == _folder_name(path=tmp_path)
