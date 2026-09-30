"""Integration tests for experiment_runner/main.py — full end-to-end pipeline."""

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml
from experiment_runner.main import main

_TEST_CONFIG = Path(__file__).parent.parent / "fixtures" / "test_config.yaml"


def _resolve_config(cache_path: Path, results_path: str, seed: int = 42) -> dict:
    """Load test_config.yaml and substitute placeholders."""
    text = _TEST_CONFIG.read_text()
    text = text.replace("__CACHE_PATH__", str(cache_path))
    text = text.replace("__RESULTS_PATH__", results_path)
    config = yaml.safe_load(text)
    config["general"]["seed"] = seed
    return config


def _write_config(config: dict, path: Path) -> Path:
    out = path / "config.yaml"
    out.write_text(yaml.dump(config))
    return out


# ── end-to-end ────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_runner_end_to_end(ucr_dataset_path, tmp_path, monkeypatch):
    """Full pipeline runs without error and produces expected output files."""
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))

    # Use a relative results path so initial_setup can call path.relative_to(".")
    results_rel = "tests/_runner_e2e"
    monkeypatch.chdir(Path(__file__).parent.parent.parent)

    config = _resolve_config(cache_root, results_rel)
    config_path = _write_config(config, tmp_path)

    main(str(config_path))

    experiment_dir = Path(results_rel) / "test_run"
    assert experiment_dir.exists(), f"Results dir not created: {experiment_dir}"

    metrics_csv = experiment_dir / "metrics.csv"
    assert metrics_csv.exists(), "metrics.csv missing"

    df = pd.read_csv(metrics_csv)
    assert not df.empty
    for col in ("dataset", "model", "explainer", "metric", "score"):
        assert col in df.columns
    assert not df["score"].isna().any(), "NaN scores in metrics.csv"

    dataset_dir = experiment_dir / "GunPoint"
    assert dataset_dir.is_dir(), "Dataset sub-directory missing"

    model_dir = dataset_dir / "FCN"
    assert model_dir.is_dir(), "Model sub-directory missing"

    checkpoints = list(model_dir.glob("*.pt"))
    assert checkpoints, "No .pt checkpoint file found"

    exp_dirs = list(model_dir.glob("explanations/explanations_*"))
    assert exp_dirs, "No explanation sub-folder found"


# ── caching ───────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_runner_with_caching(ucr_dataset_path, tmp_path, monkeypatch):
    """Split cache is created on the first run and reused on the second."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)

    # Isolated cache so the first run definitely has no pre-existing splits.
    cache_root = tmp_path / "cache"
    ucr_target = cache_root / "ucr" / "GunPoint"
    ucr_target.parent.mkdir(parents=True)
    ucr_target.symlink_to(ucr_dataset_path.resolve())

    # Run 1 — splits are generated and saved to cache.
    config = _resolve_config(cache_root, "tests/_runner_cache")
    main(str(_write_config(config, tmp_path)))

    split_folders = list((cache_root / "splits").glob("GunPoint_tr0.8_v0.1_s*_label_*"))
    assert split_folders, "Split cache folder not created after run 1"

    # Record mtimes of cached split files to confirm run 2 does not recreate them.
    split_dir = split_folders[0] / "splits"
    mtimes_before = {f: f.stat().st_mtime for f in split_dir.iterdir()}

    # Run 2 — same config, same cache → splits must be loaded, not regenerated.
    config2 = _resolve_config(cache_root, "tests/_runner_cache_2")
    main(str(_write_config(config2, tmp_path)))

    mtimes_after = {f: f.stat().st_mtime for f in split_dir.iterdir()}
    assert mtimes_before == mtimes_after, (
        "Split cache files were modified on the second run — cache was not reused"
    )


# ── cache invalidation ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_runner_cache_invalidation(ucr_dataset_path, tmp_path, monkeypatch):
    """Different seeds produce distinct split cache folders; originals untouched."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)

    # Use an isolated cache per test run so pre-existing splits don't interfere.
    cache_root = tmp_path / "cache"
    # Symlink the already-downloaded UCR data so no re-download is needed.
    ucr_target = cache_root / "ucr" / "GunPoint"
    ucr_target.parent.mkdir(parents=True)
    ucr_target.symlink_to(ucr_dataset_path.resolve())

    # Run with seed 42
    config_42 = _resolve_config(cache_root, "tests/_runner_inv_42", seed=42)
    main(str(_write_config(config_42, tmp_path)))

    folders_after_run1 = set(
        (cache_root / "splits").glob("GunPoint_tr0.8_v0.1_s*_label_*")
    )
    assert folders_after_run1, "No split folder after seed-42 run"

    # Run with seed 99 — legacy_prng will differ → different folder name
    config_99 = _resolve_config(cache_root, "tests/_runner_inv_99", seed=99)
    main(str(_write_config(config_99, tmp_path)))

    folders_after_run2 = set(
        (cache_root / "splits").glob("GunPoint_tr0.8_v0.1_s*_label_*")
    )
    new_folders = folders_after_run2 - folders_after_run1

    assert new_folders, "No new split folder created for seed-99 run"
    # Original folder(s) still present
    for folder in folders_after_run1:
        assert folder in folders_after_run2, (
            f"Original split folder disappeared: {folder}"
        )


# ── failure reporting ─────────────────────────────────────────────────────────


def _experiment_dir(results_rel: str) -> Path:
    """
    Return the run directory under *results_rel*.

    ``initial_setup`` appends a timestamp suffix when the directory already
    exists, so the name is matched by prefix rather than assumed.
    """
    candidates = sorted(Path(results_rel).glob("test_run*"))
    assert candidates, f"No run directory under {results_rel}"
    return candidates[-1]


@pytest.mark.integration
def test_runner_returns_zero_when_all_datasets_complete(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """A fully successful run returns 0 and reports no failures."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_ok"

    config = _resolve_config(cache_root, results_rel)
    failures = main(str(_write_config(config, tmp_path)))

    assert failures == 0

    log = (_experiment_dir(results_rel) / "experiment.log").read_text()
    assert "1/1 dataset(s) completed, 0 skipped, 0 failed" in log
    assert "Failed dataset(s)" not in log


@pytest.mark.integration
def test_runner_reports_and_returns_dataset_failures(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """A failing dataset is counted, named in the summary, and returned."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_fail"

    config = _resolve_config(cache_root, results_rel)
    # PatchTST validates d_model % nhead == 0, so this raises at construction
    # and the dataset iteration's `except` catches it.
    config["train_config"]["models"] = [
        {
            "model": "PatchTST",
            "init_params": {
                "in_channels": 1,
                "num_classes": 2,
                "d_model": 30,
                "nhead": 7,
            },
        }
    ]
    failures = main(str(_write_config(config, tmp_path)))

    assert failures == 1, "A failed dataset must be reported in the return value"

    log = (_experiment_dir(results_rel) / "experiment.log").read_text()
    assert "0/1 dataset(s) completed, 0 skipped, 1 failed" in log
    assert "Failed dataset(s): GunPoint" in log


@pytest.mark.integration
def test_runner_counts_skipped_datasets_separately(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """`skip_dataset` counts as skipped, not completed, and is not a failure."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_skip"

    config = _resolve_config(cache_root, results_rel)
    config["data_config"]["datasets"][0]["skip_dataset"] = True
    failures = main(str(_write_config(config, tmp_path)))

    assert failures == 0

    log = (_experiment_dir(results_rel) / "experiment.log").read_text()
    assert "0/1 dataset(s) completed, 1 skipped, 0 failed" in log
    assert "Skipped dataset(s): GunPoint" in log
