"""
Unit tests for experiment_runner/download_datasets.py — no network access.

The pure cache/zip helpers are tested directly; the ``ensure_datasets_cached``
download-strategy heuristic is tested by monkeypatching the actual download
functions so no HTTP request is ever made.
"""

import zipfile

import experiment_runner.download_datasets as dd
import pytest


def _write_ts(cache_dir, name, *, train="content", test="content"):
    """Create cache_dir/name/{name}_TRAIN.ts and _TEST.ts with given contents."""
    d = cache_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}_TRAIN.ts").write_text(train)
    (d / f"{name}_TEST.ts").write_text(test)


def _make_zip(path, entries):
    """Build a zip at *path* containing *entries* (mapping arcname → content)."""
    with zipfile.ZipFile(path, "w") as zf:
        for arcname, content in entries.items():
            zf.writestr(arcname, content)
    return path


# ── _is_cached ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_is_cached_true_when_both_ts_nonempty(tmp_path):
    _write_ts(tmp_path, "Foo")
    assert dd._is_cached(tmp_path, "Foo") is True


@pytest.mark.unit
def test_is_cached_false_when_test_missing(tmp_path):
    d = tmp_path / "Foo"
    d.mkdir()
    (d / "Foo_TRAIN.ts").write_text("content")  # no TEST file
    assert dd._is_cached(tmp_path, "Foo") is False


@pytest.mark.unit
def test_is_cached_false_when_file_empty(tmp_path):
    _write_ts(tmp_path, "Foo", test="")  # zero-byte TEST file
    assert dd._is_cached(tmp_path, "Foo") is False


# ── _zip_top_level_dir ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_zip_top_level_dir_single(tmp_path):
    zip_path = _make_zip(
        tmp_path / "a.zip",
        {"Foo/Foo_TRAIN.ts": "x", "Foo/Foo_TEST.ts": "y"},
    )
    assert dd._zip_top_level_dir(zip_path) == "Foo"


@pytest.mark.unit
def test_zip_top_level_dir_multiple_returns_none(tmp_path):
    zip_path = _make_zip(
        tmp_path / "a.zip",
        {"A/x.ts": "x", "B/y.ts": "y"},
    )
    assert dd._zip_top_level_dir(zip_path) is None


@pytest.mark.unit
def test_zip_top_level_dir_flat_returns_none(tmp_path):
    zip_path = _make_zip(tmp_path / "a.zip", {"x.ts": "x", "y.ts": "y"})
    assert dd._zip_top_level_dir(zip_path) is None


# ── ensure_datasets_cached (download strategy, no network) ───────────────────────


@pytest.mark.unit
def test_ensure_datasets_cached_skips_when_all_present(tmp_path, monkeypatch):
    names = ["Foo", "Bar"]
    for n in names:
        _write_ts(tmp_path, n)

    def _boom(*_args, **_kwargs):
        raise AssertionError("no download should happen when all cached")

    monkeypatch.setattr(dd, "_download_bulk", _boom)
    monkeypatch.setattr(dd, "_download_individual", _boom)

    dd.ensure_datasets_cached(tmp_path, names)  # must not raise


@pytest.mark.unit
def test_ensure_datasets_cached_bulk_when_majority_missing(tmp_path, monkeypatch):
    # Use real UCR names so they land in the UCR branch of the heuristic.
    names = sorted(dd._UCR_SET)[:4]
    bulk_calls = []
    individual_calls = []
    monkeypatch.setattr(
        dd, "_download_bulk", lambda _cache, url: bulk_calls.append(url)
    )
    monkeypatch.setattr(
        dd, "_download_individual", lambda _cache, name: individual_calls.append(name)
    )

    dd.ensure_datasets_cached(tmp_path, names, uses_ucr_wildcard=True)

    # ≥50% missing + wildcard active → bulk archive requested...
    assert bulk_calls == [dd.UCR_BULK_URL]
    # ...and since the mocked bulk wrote nothing, the per-dataset fallback runs
    # for every still-missing dataset.
    assert sorted(individual_calls) == names


@pytest.mark.unit
def test_ensure_datasets_cached_individual_when_no_wildcard(tmp_path, monkeypatch):
    names = sorted(dd._UCR_SET)[:3]
    bulk_calls = []
    individual_calls = []
    monkeypatch.setattr(
        dd, "_download_bulk", lambda _cache, url: bulk_calls.append(url)
    )
    monkeypatch.setattr(
        dd, "_download_individual", lambda _cache, name: individual_calls.append(name)
    )

    dd.ensure_datasets_cached(tmp_path, names, uses_ucr_wildcard=False)

    assert bulk_calls == []  # no wildcard → never bulk
    assert sorted(individual_calls) == names
