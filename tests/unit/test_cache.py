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
    # Pinned against a literal, like the _strat case above: the random-split
    # names must stay byte-identical so existing caches keep resolving.
    assert _folder_name(stratify=False) == "GunPoint_tr0.8_v0.1_s42_label_rand"


@pytest.mark.unit
def test_get_split_cache_path_official_overrides_strat_and_rand():
    name = _folder_name(use_predefined_splits=True, stratify=True)
    assert name.endswith("_label_official")
    assert "_strat" not in name and "_rand" not in name


# ── the official path names only what shaped it (R46) ─────────────────────────


@pytest.mark.unit
def test_official_path_omits_the_partition_it_does_not_use():
    # The archive's own TRAIN/TEST files are used as-is, so a name carrying
    # train_split/val_split would describe a partition the folder does not hold.
    assert _folder_name(use_predefined_splits=True) == "GunPoint_label_official"


@pytest.mark.unit
def test_official_path_omits_the_seed_when_no_samples_are_dropped():
    # With max_samples unset the official split is seed-independent:
    # _select_sample_idx returns arange(n) without drawing.
    assert "_s42" not in _folder_name(use_predefined_splits=True)


@pytest.mark.unit
def test_official_path_keeps_the_seed_when_samples_are_restricted():
    # The seed picks which rows survive the restriction, so it does shape the
    # stored arrays here and must stay in the key.
    name = _folder_name(use_predefined_splits=True, max_samples=500)
    assert name == "GunPoint_s42_label_s500r_official"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("key", "value"),
    [("train_split", 0.7), ("val_split", 0.3), ("random_state", 7)],
)
def test_editing_an_ignored_key_does_not_mint_a_new_official_entry(key, value):
    # Previously each of these forced a re-split that rewrote byte-identical
    # data under a new directory.
    assert _folder_name(use_predefined_splits=True, **{key: value}) == _folder_name(
        use_predefined_splits=True
    )


@pytest.mark.unit
def test_two_seeds_stay_apart_on_the_official_path_when_restricted():
    # The dangerous direction of R40: one entry must never serve two seeds'
    # subsamples.
    restricted = {"use_predefined_splits": True, "max_samples": 500}
    assert _folder_name(**restricted, random_state=7) != _folder_name(
        **restricted, random_state=42
    )


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
