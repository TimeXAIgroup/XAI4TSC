"""Integration tests for experiment_runner/main.py — full end-to-end pipeline."""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from experiment_runner.main import main

from xai4tsc.models.base import ModelBase

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

    experiment_dir = _experiment_dir(results_rel)
    assert experiment_dir.exists(), f"Results dir not created: {experiment_dir}"

    metrics_csv = experiment_dir / "metrics.csv"
    assert metrics_csv.exists(), "metrics.csv missing"

    df = pd.read_csv(metrics_csv)
    assert not df.empty
    for col in ("dataset", "model", "explainer", "metric", "status", "score", "detail"):
        assert col in df.columns
    # Non-"ok" rows carry an empty score by design, so only successes are checked.
    assert (df["status"] == "ok").all(), f"unexpected statuses: {set(df['status'])}"
    assert not df.loc[df["status"] == "ok", "score"].isna().any(), (
        "NaN scores on ok rows in metrics.csv"
    )

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
    # A pre-split path that does not exist fails while preparing the data, i.e.
    # at the dataset level, before any model is built. Component-level failures
    # (model, explainer, metric) are isolated and covered separately below.
    config["data_config"]["datasets"] = [
        {"dataset": "GunPoint", "is_presplit": True, "path": str(tmp_path / "missing")}
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


# ── failure isolation ─────────────────────────────────────────────────────────


@pytest.mark.integration
def test_failing_metric_does_not_discard_the_rest(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """A raising metric costs only itself; later metrics still run and are saved."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_iso_metric"

    config = _resolve_config(cache_root, results_rel)
    # Max-Sensitivity re-explains internally and needs an `explain_func` the
    # framework never supplies, so it raises. Complexity is queued behind it.
    config["evaluation_config"]["metrics"] = [
        {"metric": "Max-Sensitivity"},
        {"metric": "Complexity"},
    ]
    failures = main(str(_write_config(config, tmp_path)))

    assert failures == 1, "The failing metric should be the only reported failure"

    experiment_dir = _experiment_dir(results_rel)
    log = (experiment_dir / "experiment.log").read_text()
    # The dataset itself completed: only the metric failed.
    assert "1/1 dataset(s) completed, 0 skipped, 0 failed" in log
    assert "Failed metric evaluation(s) (1):" in log
    assert "Max-Sensitivity" in log

    metrics_csv = experiment_dir / "metrics.csv"
    assert metrics_csv.exists(), "metrics.csv must still be written"
    df = pd.read_csv(metrics_csv)
    status = dict(zip(df["metric"], df["status"], strict=True))
    # The metric queued behind the failure still ran, and the failure itself is
    # now recorded rather than dropped from the results.
    assert status["Complexity"] == "ok", "The metric after the failure must run"
    assert status["Max-Sensitivity"] == "error"


@pytest.mark.integration
def test_failing_explainer_does_not_discard_the_rest(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """A raising explainer costs only itself; later explainers still run."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_iso_explainer"

    config = _resolve_config(cache_root, results_rel)
    # TimeFrequency requires an explicit transform and raises without one.
    config["explanation_config"]["explainers"] = [
        {
            "method": "TimeFrequency",
            "base": {"method": "Guided_Backpropagation"},
            "transform": None,
        },
        {"method": "Integrated_Gradients"},
    ]
    failures = main(str(_write_config(config, tmp_path)))

    assert failures == 1

    experiment_dir = _experiment_dir(results_rel)
    log = (experiment_dir / "experiment.log").read_text()
    assert "1/1 dataset(s) completed, 0 skipped, 0 failed" in log
    assert "Failed explainer(s) (1):" in log

    df = pd.read_csv(experiment_dir / "metrics.csv")
    assert "Integrated_Gradients" in set(df["explainer"]), (
        "The explainer after the failure must still run"
    )


@pytest.mark.integration
def test_failing_model_does_not_discard_the_rest(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """A raising model costs only itself; later models still train and are scored."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_iso_model"

    config = _resolve_config(cache_root, results_rel)
    # PatchTST validates d_model % nhead == 0 and raises at construction.
    config["train_config"]["models"] = [
        {
            "model": "PatchTST",
            "init_params": {
                "in_channels": 1,
                "num_classes": 2,
                "d_model": 30,
                "nhead": 7,
            },
        },
        {"model": "FCN", "init_params": {"in_channels": 1, "num_classes": 2}},
    ]
    failures = main(str(_write_config(config, tmp_path)))

    assert failures == 1

    experiment_dir = _experiment_dir(results_rel)
    log = (experiment_dir / "experiment.log").read_text()
    assert "1/1 dataset(s) completed, 0 skipped, 0 failed" in log
    assert "Failed model(s) (1):" in log
    assert "PatchTST" in log

    df = pd.read_csv(experiment_dir / "metrics.csv")
    assert "FCN" in set(df["model"]), "The model after the failure must still run"
    assert "PatchTST" not in set(df["model"])


# ── result identity: duplicate entries ────────────────────────────────────────


@pytest.mark.integration
def test_duplicate_dataset_entries_keep_separate_artifacts(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """Two entries for one dataset get their own directory, checkpoint and rows."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_dup_dataset"

    config = _resolve_config(cache_root, results_rel)
    base = config["data_config"]["datasets"][0]
    config["data_config"]["datasets"] = [
        {**base, "max_samples": 40},
        {**base, "max_samples": 30},
    ]
    assert main(str(_write_config(config, tmp_path))) == 0

    experiment_dir = _experiment_dir(results_rel)
    # Without identities both entries shared "GunPoint/" and the second
    # overwrote the first's checkpoint and evaluation plots.
    for ident in ("GunPoint_1", "GunPoint_2"):
        model_dir = experiment_dir / ident / "FCN"
        assert model_dir.is_dir(), f"missing output directory for {ident}"
        assert list(model_dir.glob("*.pt")), f"no checkpoint under {ident}"
    assert not (experiment_dir / "GunPoint").exists()

    df = pd.read_csv(experiment_dir / "metrics.csv")
    assert set(df["dataset"]) == {"GunPoint_1", "GunPoint_2"}


@pytest.mark.integration
def test_duplicate_explainer_entries_keep_separate_plots(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """Two entries for one explainer get their own output directory and PNGs."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_dup_explainer"

    config = _resolve_config(cache_root, results_rel)
    config["explanation_config"]["explainers"] = [
        {"method": "Integrated_Gradients", "n_steps": 4},
        {"method": "Integrated_Gradients", "n_steps": 16},
    ]
    assert main(str(_write_config(config, tmp_path))) == 0

    exp_root = _experiment_dir(results_rel) / "GunPoint" / "FCN" / "explanations"
    # This is the artifact-loss regression: previously both entries wrote into
    # explanations_Integrated_Gradients/ and the second clobbered the first.
    for ident in ("Integrated_Gradients_1", "Integrated_Gradients_2"):
        d = exp_root / f"explanations_{ident}"
        assert d.is_dir(), f"missing explanation directory for {ident}"
        assert list(d.glob("*.png")), f"no plots under {ident}"
    assert not (exp_root / "explanations_Integrated_Gradients").exists()

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    assert set(df["explainer"]) == {
        "Integrated_Gradients_1",
        "Integrated_Gradients_2",
    }


@pytest.mark.integration
def test_duplicate_metric_entries_are_distinguishable_in_results(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """Two entries for one metric produce rows that can be told apart."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_dup_metric"

    config = _resolve_config(cache_root, results_rel)
    config["evaluation_config"]["metrics"] = [
        {"metric": "Complexity"},
        {"metric": "Complexity"},
    ]
    assert main(str(_write_config(config, tmp_path))) == 0

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    assert set(df["metric"]) == {"Complexity_1", "Complexity_2"}


@pytest.mark.integration
def test_unique_entries_keep_their_plain_names(ucr_dataset_path, tmp_path, monkeypatch):
    """No duplicates means no suffixes anywhere - paths stay as before."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_nodup"

    config = _resolve_config(cache_root, results_rel)
    assert main(str(_write_config(config, tmp_path))) == 0

    experiment_dir = _experiment_dir(results_rel)
    assert (experiment_dir / "GunPoint" / "FCN").is_dir()
    assert not list(experiment_dir.glob("GunPoint_*"))

    df = pd.read_csv(experiment_dir / "metrics.csv")
    assert set(df["dataset"]) == {"GunPoint"}
    assert set(df["model"]) == {"FCN"}
    assert set(df["explainer"]) == {"Integrated_Gradients"}
    assert set(df["metric"]) == {"Complexity"}


# ── result status ─────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_status_error_row_records_the_exception(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """A raising metric is recorded as an `error` row naming the exception."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_status_error"

    config = _resolve_config(cache_root, results_rel)
    config["evaluation_config"]["metrics"] = [
        {"metric": "Max-Sensitivity"},
        {"metric": "Complexity"},
    ]
    assert main(str(_write_config(config, tmp_path))) == 1

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    by_metric = dict(zip(df["metric"], df["status"], strict=True))
    assert by_metric == {"Max-Sensitivity": "error", "Complexity": "ok"}

    failed = df.loc[df["status"] == "error"].iloc[0]
    assert pd.isna(failed["score"]), "an error row must carry no score"
    assert "explain_func" in failed["detail"]


@pytest.mark.integration
def test_status_skipped_domain_row_is_recorded(ucr_dataset_path, tmp_path, monkeypatch):
    """A domain-incompatible metric is recorded rather than silently dropped."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_status_skipped"

    config = _resolve_config(cache_root, results_rel)
    # A FREQUENCY-only metric alongside a FREQUENCY explainer (so the config
    # validates) and a TIME explainer that cannot satisfy it.
    config["explanation_config"]["explainers"] = [
        {"method": "Integrated_Gradients"},
        {
            "method": "Frequency",
            "base": {"method": "Integrated_Gradients"},
            "transform": {"name": "rfft", "params": {}},
        },
    ]
    config["evaluation_config"]["metrics"] = [{"metric": "Frequency Perturbation"}]
    assert main(str(_write_config(config, tmp_path))) == 0

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    statuses = dict(zip(df["explainer"], df["status"], strict=True))
    assert statuses["Integrated_Gradients"] == "skipped-domain"
    assert statuses["FREQUENCY-Integrated_Gradients"] == "ok"

    skipped = df.loc[df["status"] == "skipped-domain"].iloc[0]
    assert "Frequency" in skipped["detail"]


@pytest.mark.integration
def test_header_only_metrics_csv_when_nothing_succeeds(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """A finished run always leaves a results file with the known schema."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_header_only"

    config = _resolve_config(cache_root, results_rel)
    config["data_config"]["datasets"] = [
        {"dataset": "GunPoint", "is_presplit": True, "path": str(tmp_path / "missing")}
    ]
    assert main(str(_write_config(config, tmp_path))) == 1

    metrics_csv = _experiment_dir(results_rel) / "metrics.csv"
    assert metrics_csv.exists(), "a header-only metrics.csv must still be written"
    df = pd.read_csv(metrics_csv)
    assert df.empty
    assert list(df.columns) == [
        "dataset",
        "model",
        "explainer",
        "metric",
        "status",
        "score",
        "detail",
    ]


# ── local datasets through the runner ─────────────────────────────────────────


@pytest.mark.integration
def test_runner_drives_a_local_dataset(tmp_path, monkeypatch):
    """
    A `path:` dataset runs end to end.

    The suite previously covered only UCR and synthetic datasets, which is how
    `load_dataset` came to forward runner-only kwargs that `LocalDataset` cannot
    accept, making `path:` unusable from any config.
    """
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    results_rel = "tests/_runner_local"

    # A small, self-contained (B, C, T) dataset.
    data_dir = tmp_path / "local_data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 40)
    arr = rng.normal(size=(40, 1, 48)) + labels[:, None, None]
    np.save(data_dir / "data.npy", arr.astype(np.float32))
    (data_dir / "label.json").write_text(
        json.dumps([{"label": str(int(y))} for y in labels])
    )

    config = _resolve_config(tmp_path / "cache", results_rel)
    config["data_config"]["datasets"] = [{"dataset": "local_ds", "path": str(data_dir)}]
    config["train_config"]["models"] = [
        {"model": "LeNet", "init_params": {"in_channels": 1, "num_classes": 2}}
    ]
    assert main(str(_write_config(config, tmp_path))) == 0

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    assert set(df["dataset"]) == {"local_ds"}
    assert (df["status"] == "ok").all()


@pytest.mark.integration
def test_local_dataset_with_more_channels_than_timesteps_keeps_its_shape(
    tmp_path, monkeypatch
):
    """
    `layout: bct` stops the shape heuristic corrupting genuine C>T data.

    With `layout: auto` the array is read as (B, T, C) and transposed, so the
    model is built on the wrong axis. Metrics are skipped on purpose: Quantus runs
    the same shape guess in `infer_channel_first` and transposes `x_batch` for a
    C>T input while leaving `a_batch` alone, so no Quantus metric can score this
    data however it was loaded.
    """
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    results_rel = "tests/_runner_layout"

    data_dir = tmp_path / "wide"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 40)
    # 40 channels, 24 timesteps: axis 1 > axis 2, so "auto" would transpose it,
    # while 24 timesteps is still long enough for LeNet's pooling stack.
    arr = rng.normal(size=(40, 40, 24)) + labels[:, None, None]
    np.save(data_dir / "data.npy", arr.astype(np.float32))
    (data_dir / "label.json").write_text(
        json.dumps([{"label": str(int(y))} for y in labels])
    )

    config = _resolve_config(tmp_path / "cache", results_rel)
    config["data_config"]["datasets"] = [
        {
            "dataset": "wide",
            "path": str(data_dir),
            "layout": "bct",
            "skip_metrics": True,
        }
    ]
    config["train_config"]["models"] = [
        {"model": "LeNet", "init_params": {"in_channels": 40, "num_classes": 2}}
    ]
    assert main(str(_write_config(config, tmp_path))) == 0

    # Training and explaining both ran with in_channels=40, which only works if
    # the 40-channel axis survived the load instead of becoming 24.
    model_dir = _experiment_dir(results_rel) / "wide" / "LeNet"
    assert list(model_dir.glob("*.pt")), "model did not train on the C>T data"
    assert list(model_dir.glob("explanations/explanations_*/*.png")), (
        "no explanations produced for the C>T data"
    )


@pytest.mark.integration
def test_changing_format_does_not_reuse_a_cache_built_by_another_loader(
    tmp_path, monkeypatch
):
    """
    `format` is part of the split-cache key.

    Without it, a numpy-built cache entry is served for a `csv` request and the
    requested loader is never called, so an unimplemented format looks like a
    success.
    """
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = tmp_path / "cache"

    data_dir = tmp_path / "local"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 40)
    np.save(data_dir / "data.npy", (rng.normal(size=(40, 1, 48))).astype(np.float32))
    (data_dir / "label.json").write_text(
        json.dumps([{"label": str(int(y))} for y in labels])
    )

    entry = {"dataset": "local_ds", "path": str(data_dir)}
    base = _resolve_config(cache_root, "tests/_runner_fmt_numpy")
    base["data_config"]["datasets"] = [{**entry, "format": "numpy"}]
    base["train_config"]["models"] = [
        {"model": "LeNet", "init_params": {"in_channels": 1, "num_classes": 2}}
    ]
    assert main(str(_write_config(base, tmp_path))) == 0
    numpy_caches = list((cache_root / "splits").iterdir())
    assert numpy_caches, "no split cache written for the numpy run"

    # Same dataset name and path, different loader: must not reuse the entry above.
    csv_cfg = _resolve_config(cache_root, "tests/_runner_fmt_csv")
    csv_cfg["data_config"]["datasets"] = [{**entry, "format": "csv"}]
    csv_cfg["train_config"]["models"] = base["train_config"]["models"]
    # The CSV loader is not implemented, so the dataset must fail rather than
    # silently succeed from the numpy cache.
    assert main(str(_write_config(csv_cfg, tmp_path))) == 1


# ── reproducible: false (R1) ──────────────────────────────────────────────────


@pytest.mark.integration
def test_reproducible_false_completes_the_run(ucr_dataset_path, tmp_path, monkeypatch):
    """The documented `reproducible: false` value must not abort the dataset."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_unseeded"

    config = _resolve_config(cache_root, results_rel)
    config["general"]["reproducible"] = False
    failures = main(str(_write_config(config, tmp_path)))

    assert failures == 0, "An unseeded run must not fail its dataset"

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    assert not df.empty
    assert (df["status"] == "ok").all(), f"unexpected statuses: {set(df['status'])}"


@pytest.mark.integration
def test_reproducible_false_bypasses_the_split_cache(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """Unseeded splits cannot be described by a cache key, so none is written."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)

    # Isolated cache so only this run can have written to it.
    cache_root = tmp_path / "cache"
    ucr_target = cache_root / "ucr" / "GunPoint"
    ucr_target.parent.mkdir(parents=True)
    ucr_target.symlink_to(ucr_dataset_path.resolve())

    config = _resolve_config(cache_root, "tests/_runner_unseeded_cache")
    config["general"]["reproducible"] = False
    assert main(str(_write_config(config, tmp_path))) == 0

    assert not list((cache_root / "splits").glob("GunPoint_*")), (
        "An unseeded run must not persist splits to the cache"
    )


# ── checkpoint reload without retraining (R22) ────────────────────────────────


@pytest.mark.integration
def test_checkpoint_reload_without_retraining(ucr_dataset_path, tmp_path, monkeypatch):
    """Omitting `train` and pointing at a written checkpoint scores that model."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))

    # Run 1 - train, so a checkpoint and its config file exist.
    train_rel = "tests/_runner_ckpt_train"
    config = _resolve_config(cache_root, train_rel)
    assert main(str(_write_config(config, tmp_path))) == 0

    model_dir = _experiment_dir(train_rel) / "GunPoint" / "FCN"
    checkpoints = sorted(model_dir.glob("FCN_epoch_*.pt"))
    assert checkpoints, "Run 1 wrote no checkpoint to reload"
    assert ModelBase.config_path_for(checkpoints[-1]).is_file(), (
        "No config file written beside the checkpoint"
    )

    # Run 2 - no training, just load the checkpoint and score it. The reload
    # config names only the model and the path; init_params come from the
    # config file beside the checkpoint. `train` is *omitted*, not falsified:
    # the inherited default_settings value must not reinstate training.
    reload_rel = "tests/_runner_ckpt_reload"
    reload_config = _resolve_config(cache_root, reload_rel)
    assert reload_config["train_config"]["default_settings"]["train"]
    reload_config["train_config"]["models"] = [
        {"model": "FCN", "model_ckpt_path": str(checkpoints[-1])}
    ]
    failures = main(str(_write_config(reload_config, tmp_path)))

    assert failures == 0, "Scoring a pre-trained checkpoint must not fail"

    log = (_experiment_dir(reload_rel) / "experiment.log").read_text()
    assert "Not training FCN" in log

    df = pd.read_csv(_experiment_dir(reload_rel) / "metrics.csv")
    assert not df.empty, "No metric rows produced for the reloaded checkpoint"
    assert (df["status"] == "ok").all(), f"unexpected statuses: {set(df['status'])}"

    # Nothing was retrained: no new checkpoint under the reload run.
    assert not list((_experiment_dir(reload_rel) / "GunPoint" / "FCN").glob("*.pt"))


@pytest.mark.integration
def test_checkpoint_entry_with_explicit_train_still_trains(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """Only the *inherited* `train` default is dropped, so fine-tuning works."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))

    train_rel = "tests/_runner_ckpt_ft_train"
    config = _resolve_config(cache_root, train_rel)
    assert main(str(_write_config(config, tmp_path))) == 0
    checkpoint = sorted(
        (_experiment_dir(train_rel) / "GunPoint" / "FCN").glob("FCN_epoch_*.pt")
    )[-1]

    trained_epoch = int(checkpoint.stem.rsplit("_", 1)[1])

    # The legacy string is used deliberately: it must still train, so the
    # deprecation path stays covered end to end.
    ft_rel = "tests/_runner_ckpt_ft"
    ft_config = _resolve_config(cache_root, ft_rel)
    ft_config["train_config"]["models"] = [
        {
            "model": "FCN",
            "model_ckpt_path": str(checkpoint),
            "train": "for_epochs",
        }
    ]
    assert main(str(_write_config(ft_config, tmp_path))) == 0

    ft_dir = _experiment_dir(ft_rel)
    written = list((ft_dir / "GunPoint" / "FCN").glob("*.pt"))
    assert written, "Explicit `train` must still train"

    # Epoch numbering continues from the checkpoint rather than restarting at 1,
    # so the new checkpoint cannot collide with the one it resumed from.
    resumed_epoch = int(written[0].stem.rsplit("_", 1)[1])
    assert resumed_epoch > trained_epoch, (
        f"resumed run wrote {written[0].name}, restarting the epoch count"
    )

    log = (ft_dir / "experiment.log").read_text()
    assert "deprecated" in log, "the legacy `train` value must warn"
    # The epoch number is %2d-padded in the log line.
    assert re.search(rf"Epoch +{trained_epoch + 1}/", log), (
        f"expected the first epoch of the resumed run to be {trained_epoch + 1}"
    )
    # load_model resolves the path, so the log carries the absolute form.
    assert f"Loaded FCN from checkpoint {checkpoint.absolute()}" in log
    assert f"trained {trained_epoch} epoch(s)" in log

    # The config file records the resume, and only then.
    config_file = ModelBase.config_path_for(written[0])
    recorded = json.loads(config_file.read_text())
    assert recorded["epoch"] == resumed_epoch
    assert recorded["resumed_from"] == str(checkpoint.absolute())


@pytest.mark.integration
def test_train_false_with_a_checkpoint_does_not_train(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """`train: false` is the explicit way to say "score these weights as-is"."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))

    train_rel = "tests/_runner_trainfalse_train"
    config = _resolve_config(cache_root, train_rel)
    assert main(str(_write_config(config, tmp_path))) == 0
    checkpoint = sorted(
        (_experiment_dir(train_rel) / "GunPoint" / "FCN").glob("FCN_epoch_*.pt")
    )[-1]

    off_rel = "tests/_runner_trainfalse"
    off_config = _resolve_config(cache_root, off_rel)
    off_config["train_config"]["models"] = [
        {"model": "FCN", "model_ckpt_path": str(checkpoint), "train": False}
    ]
    assert main(str(_write_config(off_config, tmp_path))) == 0

    off_dir = _experiment_dir(off_rel)
    assert not list((off_dir / "GunPoint" / "FCN").glob("*.pt"))
    assert "Not training FCN" in (off_dir / "experiment.log").read_text()
    assert not pd.read_csv(off_dir / "metrics.csv").empty
