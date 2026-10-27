"""Integration tests for experiment_runner/main.py — full end-to-end pipeline."""

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from experiment_runner.main import main
from sklearn.preprocessing import LabelEncoder

from xai4tsc.evaluation import METRICS, EvaluatorBase, register_metric
from xai4tsc.models.base import ModelBase
from xai4tsc.models.models import load_model

_TEST_CONFIG = Path(__file__).parent.parent / "fixtures" / "test_config.yaml"


class _RaisingEvaluator(EvaluatorBase):
    """
    A metric that always raises, for exercising per-metric error isolation.

    Earlier these tests borrowed ``Max-Sensitivity``, which raised only because
    the framework supplied no ``explain_func``. That is fixed, so the failure has
    to be deliberate rather than inherited from whichever metric is broken today.
    """

    def evaluate(self, model, explanation, data, labels, device="cpu", **kwargs):
        """Raise unconditionally."""
        raise RuntimeError("deliberate metric failure for isolation testing")


@pytest.fixture
def raising_metric():
    """Register ``Always-Raises`` for the duration of one test."""
    register_metric("Always-Raises", _RaisingEvaluator)
    yield "Always-Raises"
    METRICS.pop("Always-Raises", None)


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
    for col in (
        "dataset",
        "model",
        "explainer",
        "metric",
        "seed",
        "status",
        "score",
        "detail",
    ):
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

    # The per-model log holds this model's part of the run.
    model_log = (model_dir / "GunPoint_FCN.log").read_text()
    assert "Now also logging to file" in model_log
    assert "FCN" in model_log


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
    ucr_dataset_path, tmp_path, monkeypatch, raising_metric
):
    """A raising metric costs only itself; later metrics still run and are saved."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_iso_metric"

    config = _resolve_config(cache_root, results_rel)
    # A metric that always raises, with Complexity queued behind it.
    config["evaluation_config"]["metrics"] = [
        {"metric": raising_metric},
        {"metric": "Complexity"},
    ]
    failures = main(str(_write_config(config, tmp_path)))

    assert failures == 1, "The failing metric should be the only reported failure"

    experiment_dir = _experiment_dir(results_rel)
    log = (experiment_dir / "experiment.log").read_text()
    # The dataset itself completed: only the metric failed.
    assert "1/1 dataset(s) completed, 0 skipped, 0 failed" in log
    assert "Failed metric evaluation(s) (1):" in log
    assert raising_metric in log

    metrics_csv = experiment_dir / "metrics.csv"
    assert metrics_csv.exists(), "metrics.csv must still be written"
    df = pd.read_csv(metrics_csv)
    status = dict(zip(df["metric"], df["status"], strict=True))
    # The metric queued behind the failure still ran, and the failure itself is
    # now recorded rather than dropped from the results.
    assert status["Complexity"] == "ok", "The metric after the failure must run"
    assert status[raising_metric] == "error"


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
    ucr_dataset_path, tmp_path, monkeypatch, raising_metric
):
    """A raising metric is recorded as an `error` row naming the exception."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_status_error"

    config = _resolve_config(cache_root, results_rel)
    config["evaluation_config"]["metrics"] = [
        {"metric": raising_metric},
        {"metric": "Complexity"},
    ]
    assert main(str(_write_config(config, tmp_path))) == 1

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    by_metric = dict(zip(df["metric"], df["status"], strict=True))
    assert by_metric == {raising_metric: "error", "Complexity": "ok"}

    failed = df.loc[df["status"] == "error"].iloc[0]
    assert pd.isna(failed["score"]), "an error row must carry no score"
    assert "deliberate metric failure" in failed["detail"]


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
def test_quantus_metrics_follow_their_group_across_domains(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """
    A Quantus metric is paired with a frequency explanation only if it can score it.

    `Complexity` reads the attribution alone, so it applies in any domain;
    `Pixel-Flipping` perturbs the input and re-queries the model, so it is
    time-domain only. Both used to be run against both explainers, and both used to
    raise `AssertionError` on the frequency one.
    """
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_quantus_domains"

    config = _resolve_config(cache_root, results_rel)
    config["explanation_config"]["explainers"] = [
        {"method": "Integrated_Gradients"},
        {
            "method": "Frequency",
            "base": {"method": "Integrated_Gradients"},
            "transform": {"name": "rfft", "params": {}},
        },
    ]
    config["evaluation_config"]["metrics"] = [
        {"metric": "Complexity"},
        {"metric": "Pixel-Flipping"},
    ]
    assert main(str(_write_config(config, tmp_path))) == 0

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    status = {
        (row["explainer"], row["metric"]): row["status"] for _, row in df.iterrows()
    }
    assert status[("Integrated_Gradients", "Complexity")] == "ok"
    assert status[("Integrated_Gradients", "Pixel-Flipping")] == "ok"
    assert status[("FREQUENCY-Integrated_Gradients", "Complexity")] == "ok"
    assert status[("FREQUENCY-Integrated_Gradients", "Pixel-Flipping")] == (
        "skipped-domain"
    )


@pytest.mark.integration
def test_frequency_native_dataset_scores_input_perturbing_metrics(
    tmp_path, monkeypatch
):
    """
    `data_domain: frequency` makes the model-native domain the frequency domain.

    A model trained on spectra consumes coefficients, so a plain attributor's
    relevance is a frequency explanation and `Pixel-Flipping` - which perturbs the
    input and re-queries the model - applies there. It used to be refused as
    "time-domain only", which was never about time but about the model.
    """
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    results_rel = "tests/_runner_freq_native"

    # Real-valued magnitude spectra: 33 rFFT bins of a 64-sample series.
    data_dir = tmp_path / "spectra"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 40)
    series = rng.normal(size=(40, 1, 64)) + labels[:, None, None]
    spectra = np.abs(np.fft.rfft(series, axis=-1)).astype(np.float32)
    np.save(data_dir / "data.npy", spectra)
    (data_dir / "label.json").write_text(
        json.dumps([{"label": str(int(y))} for y in labels])
    )

    config = _resolve_config(tmp_path / "cache", results_rel)
    config["data_config"]["datasets"] = [
        {
            "dataset": "spectra",
            "path": str(data_dir),
            "layout": "bct",
            "data_domain": "frequency",
        }
    ]
    config["train_config"]["models"] = [
        {"model": "LeNet", "init_params": {"in_channels": 1, "num_classes": 2}}
    ]
    config["explanation_config"]["explainers"] = [{"method": "Integrated_Gradients"}]
    config["evaluation_config"]["metrics"] = [
        {"metric": "Pixel-Flipping"},
        {"metric": "Complexity"},
    ]
    assert main(str(_write_config(config, tmp_path))) == 0

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    assert set(df["metric"]) == {"Pixel-Flipping", "Complexity"}
    assert (df["status"] == "ok").all(), df.loc[df["status"] != "ok"].to_dict("records")
    assert df["score"].notna().all()


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
        "target",
        "samples",
        "seed",
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
    model is built on the wrong axis. The data is also scored: Quantus used to run
    the same shape guess in `infer_channel_first` and transpose `x_batch` for a C>T
    input while leaving `a_batch` alone, until the adapter started stating the
    canonical layout instead of letting it be inferred.
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
        {"dataset": "wide", "path": str(data_dir), "layout": "bct"}
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

    # ...and the metrics scored it rather than failing an axis-order mismatch.
    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    assert not df.empty, "no metric rows for the C>T data"
    assert (df["status"] == "ok").all(), df.loc[df["status"] != "ok"].to_dict("records")
    assert df["score"].notna().all()


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


# ── required settings and the reference-only master (R26) ─────────────────────

_MASTER = (
    Path(__file__).parent.parent.parent
    / "experiment_runner"
    / "configs"
    / "master.yaml"
)


@pytest.mark.integration
def test_master_config_is_refused(tmp_path, monkeypatch):
    """master.yaml is the reference, so running it must say so rather than run."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)

    with pytest.raises(ValueError) as excinfo:
        main(str(_MASTER))

    message = str(excinfo.value)
    assert "master.yaml" in message
    assert "not a runnable config" in message


@pytest.mark.integration
def test_missing_component_list_fails_before_creating_anything(tmp_path, monkeypatch):
    """The check runs before the results directory, so nothing lands on disk."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    # Unique per run, so the assertion cannot be confused by an earlier run's
    # leftovers (a pre-fix run does create this directory).
    results_rel = f"tests/_runner_missing_metrics_{tmp_path.name}"
    shutil.rmtree(results_rel, ignore_errors=True)

    config = _resolve_config(cache_root, results_rel)
    del config["evaluation_config"]["metrics"]

    with pytest.raises(ValueError, match=r"evaluation_config\.metrics"):
        main(str(_write_config(config, tmp_path)))

    assert not Path(results_rel).exists(), "nothing may be created before validation"


@pytest.mark.integration
def test_missing_results_path_names_the_key(tmp_path, monkeypatch):
    """R26: this used to mkdir the user's own config file."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))

    config = _resolve_config(cache_root, "tests/_runner_no_results_path")
    del config["results_rel_path"]
    config_path = _write_config(config, tmp_path)

    with pytest.raises(ValueError, match="results_rel_path"):
        main(str(config_path))

    assert config_path.is_file(), "the config file must not be turned into a directory"


# ── R2: a frequency explainer no longer inherits master's STFT transform ──────


@pytest.mark.integration
def test_freqrise_fft_needs_no_transform(ucr_dataset_path, tmp_path, monkeypatch):
    """`domain: fft` with no transform was unusable while master backfilled one."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_freqrise_fft"

    config = _resolve_config(cache_root, results_rel)
    config["explanation_config"]["explainers"] = [
        {
            "method": "FreqRISE",
            "domain": "fft",  # deliberately no `transform`
            "num_batches": 2,
            "batch_size": 4,
            "num_cells": 8,
            "seed": 0,
        }
    ]
    config["evaluation_config"]["metrics"] = [{"metric": "Frequency Perturbation"}]

    assert main(str(_write_config(config, tmp_path))) == 0

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    assert not df.empty
    assert (df["status"] == "ok").all(), f"unexpected statuses: {set(df['status'])}"


# ── entry validation end to end (R6, R7) ──────────────────────────────────────


@pytest.mark.integration
def test_misspelled_metric_fails_before_training(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """R6: this used to exit 0 with a silently short metrics.csv."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = f"tests/_runner_bad_metric_{tmp_path.name}"
    shutil.rmtree(results_rel, ignore_errors=True)

    config = _resolve_config(cache_root, results_rel)
    config["evaluation_config"]["metrics"] = [{"metric": "Complexety"}]

    with pytest.raises(ValueError, match="Complexity"):
        main(str(_write_config(config, tmp_path)))

    # The run directory is created during setup, but no model was ever trained.
    assert not list(Path(results_rel).glob("**/*.pt"))


@pytest.mark.integration
def test_typo_in_an_explainer_parameter_warns_but_completes(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """R7: the value is ignored, so say so rather than silently use the default."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_param_typo"

    config = _resolve_config(cache_root, results_rel)
    config["explanation_config"]["explainers"] = [
        {"method": "Integrated_Gradients", "nsteps": 3}
    ]
    assert main(str(_write_config(config, tmp_path))) == 0

    log = (_experiment_dir(results_rel) / "experiment.log").read_text()
    assert "'nsteps' is not a known explainer entry key" in log
    assert "Did you mean 'n_steps'?" in log


@pytest.mark.integration
def test_config_warnings_are_repeated_at_the_end(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """A long run scrolls the config report away, so it is restated at the end."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_repeat_warnings"

    config = _resolve_config(cache_root, results_rel)
    config["explanation_config"]["explainers"] = [
        {"method": "Integrated_Gradients", "nsteps": 3}
    ]
    assert main(str(_write_config(config, tmp_path))) == 0

    log = (_experiment_dir(results_rel) / "experiment.log").read_text()

    # Reported once up front, under the header, and again after the run finished.
    assert "Configuration check: 0 error(s), 1 warning(s)" in log
    assert "Configuration warnings, repeated (1)" in log
    assert log.index("Experiment Finished") < log.index("repeated (1)")
    # The whole report sits after file logging starts, so experiment.log has it all.
    assert log.index("Now also logging to file") < log.index("Configuration check")


@pytest.mark.integration
def test_empty_list_warning_reaches_the_log_file(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """Deferred, because the check runs before experiment.log exists."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_empty_metrics"

    config = _resolve_config(cache_root, results_rel)
    config["evaluation_config"]["metrics"] = []
    assert main(str(_write_config(config, tmp_path))) == 0

    log = (_experiment_dir(results_rel) / "experiment.log").read_text()
    assert "evaluation_config.metrics: this list is empty" in log


# ── per-result reproducibility (R31) ──────────────────────────────────────────


def _make_local_dataset(directory: Path, seed: int, n: int = 40) -> Path:
    """Write a small self-contained ``(B, C, T)`` dataset and its labels."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    arr = rng.normal(size=(n, 1, 48)) + labels[:, None, None]
    np.save(directory / "data.npy", arr.astype(np.float32))
    (directory / "label.json").write_text(
        json.dumps([{"label": str(int(y))} for y in labels])
    )
    return directory


def _score_of(results_rel: str, **keys: str) -> float:
    """Return the single metrics.csv score matching *keys*."""
    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    for column, value in keys.items():
        df = df[df[column] == value]
    assert len(df) == 1, f"expected one row for {keys}, got {len(df)}"
    assert df.iloc[0]["status"] == "ok", df.iloc[0]["detail"]
    return float(df.iloc[0]["score"])


@pytest.mark.integration
def test_a_result_is_identical_alone_and_in_a_multi_entry_run(tmp_path, monkeypatch):
    """
    The end-to-end regression test for R31.

    A single run seed used to be applied once, after which the RNG advanced through
    every split, model and explanation - so a result's numbers depended on which
    entries ran before it, and re-running one dataset to check a number gave a
    different number. The same (dataset, model, explainer, metric) row must now
    score the same whether it runs alone or surrounded by other entries.
    """
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    data_dir = _make_local_dataset(tmp_path / "local_data", seed=0)
    target = {
        "dataset": "local_ds",
        "model": "FCN",
        "explainer": "Integrated_Gradients",
        "metric": "Complexity",
    }

    def run(results_rel: str, crowded: bool) -> float:
        config = _resolve_config(tmp_path / f"cache_{results_rel[-5:]}", results_rel)
        config["data_config"]["datasets"] = [
            {"dataset": "local_ds", "path": str(data_dir)}
        ]
        models = [{"model": "FCN", "init_params": {"in_channels": 1, "num_classes": 2}}]
        explainers = [{"method": "Integrated_Gradients"}]
        metrics = [{"metric": "Complexity"}]
        if crowded:
            # Extra entries *before* the target ones, which is what used to move it.
            models = [
                {
                    "model": "LeNet",
                    "init_params": {"in_channels": 1, "num_classes": 2},
                },
                *models,
            ]
            explainers = [{"method": "DeepLift"}, *explainers]
            metrics = [{"metric": "Sparseness"}, *metrics]
        config["train_config"]["models"] = models
        config["explanation_config"]["explainers"] = explainers
        config["evaluation_config"]["metrics"] = metrics
        assert main(str(_write_config(config, tmp_path / results_rel[-5:]))) == 0
        return _score_of(results_rel, **target)

    (tmp_path / "alone").mkdir()
    (tmp_path / "crowd").mkdir()
    alone = run("tests/_runner_seed_alone", crowded=False)
    crowded = run("tests/_runner_seed_crowd", crowded=True)
    assert alone == crowded, (
        f"the same result scored {alone} alone and {crowded} beside other entries"
    )


@pytest.mark.integration
def test_split_cache_key_matches_the_split_it_contains(tmp_path, monkeypatch):
    """
    A dataset's cache folder must describe the split stored under it.

    The folder name used to carry a value drawn once per run while the split seed
    was drawn per dataset from the shared stream, so the same dataset in two
    positions wrote different splits under one key - and a later run could be
    served a split its own config would never have produced.
    """
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    first = _make_local_dataset(tmp_path / "ds_a", seed=1)
    second = _make_local_dataset(tmp_path / "ds_b", seed=2)
    entries = {
        "ds_a": {"dataset": "ds_a", "path": str(first)},
        "ds_b": {"dataset": "ds_b", "path": str(second)},
    }

    def run(results_rel: str, order: list[str]) -> set[str]:
        cache = tmp_path / f"cache_{order[0]}"
        config = _resolve_config(cache, results_rel)
        config["data_config"]["datasets"] = [entries[name] for name in order]
        config["train_config"]["models"] = [
            {"model": "LeNet", "init_params": {"in_channels": 1, "num_classes": 2}}
        ]
        assert main(str(_write_config(config, cache.parent / order[0]))) == 0
        return {p.name for p in (cache / "splits").glob("ds_b_*")}

    (tmp_path / "ds_b_first").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ds_a_first").mkdir(parents=True, exist_ok=True)
    as_first = run("tests/_runner_key_first", ["ds_b", "ds_a"])
    as_second = run("tests/_runner_key_second", ["ds_a", "ds_b"])

    assert as_first and as_second
    assert as_first == as_second, (
        "the same dataset landed in different cache folders depending on its "
        f"position: {as_first} vs {as_second}"
    )


@pytest.mark.integration
def test_metrics_csv_records_the_derived_seed(tmp_path, monkeypatch, ucr_dataset_path):
    """Every row carries the seed that produced it, so it can be re-run."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    results_rel = "tests/_runner_seed_column"
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    config = _resolve_config(cache_root, results_rel)
    assert main(str(_write_config(config, tmp_path))) == 0

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    assert not df.empty
    assert df["seed"].notna().all(), "a seeded run must record a seed on every row"
    assert (df["seed"].astype("int64") >= 0).all()


@pytest.mark.integration
def test_reproducible_false_leaves_the_seed_column_empty(
    tmp_path, monkeypatch, ucr_dataset_path
):
    """An unseeded run must not claim a seed it never applied."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    results_rel = "tests/_runner_seed_column_unseeded"
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    config = _resolve_config(cache_root, results_rel)
    config["general"]["reproducible"] = False
    assert main(str(_write_config(config, tmp_path))) == 0

    df = pd.read_csv(_experiment_dir(results_rel) / "metrics.csv")
    assert not df.empty
    assert df["seed"].isna().all()


# ── reproducing a logged result (R31's promise, end to end) ───────────────────

_SEED_LINE = re.compile(r"Seed for (\w+) '(.+?)' \(#(\d+)\): (\d+)")


def _logged_seeds(log_path: Path) -> dict:
    """
    Return ``{(kind, name): seed}`` for every seed a run recorded.

    The log is the only place a user can read these from, which is the point of
    the tests below: whatever is missing here cannot be reproduced.
    """
    seeds = {}
    for kind, name, _occurrence, seed in _SEED_LINE.findall(log_path.read_text()):
        seeds[(kind, name)] = int(seed)
    return seeds


@pytest.mark.integration
def test_logfile_records_every_seed_a_result_depends_on(
    tmp_path, monkeypatch, ucr_dataset_path
):
    """
    A result's randomness comes from five boundaries; the log must name all five.

    `split` and `samples` used to be derived with ``log=False``, which left the
    log unable to answer the one question it is kept for. `samples` is the
    load-bearing one: it decides which test samples are explained, and metrics
    are computed only on those, so a logged score could not be reproduced from
    the log no matter how many other seeds were in it.
    """
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    results_rel = "tests/_runner_seed_logged"
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    config = _resolve_config(cache_root, results_rel)
    assert main(str(_write_config(config, tmp_path))) == 0

    log = _experiment_dir(results_rel) / "experiment.log"
    seeds = _logged_seeds(log)
    recorded = {kind for kind, _ in seeds}

    assert recorded >= {
        "dataset",
        "split",
        "samples",
        "model",
        "explainer",
        "metric",
    }, (
        f"the log is missing boundaries: {{'dataset', 'split', 'samples', 'model', "
        f"'explainer', 'metric'}} - {recorded}"
    )
    assert "Run seed:" in log.read_text()
    # Each boundary is a distinct stream; two sharing a value would mean the
    # derivation collapsed.
    assert len(set(seeds.values())) == len(seeds)


@pytest.mark.integration
def test_logged_metric_seed_matches_the_metrics_csv_column(
    tmp_path, monkeypatch, ucr_dataset_path
):
    """
    The two records of a result's seed must agree, or neither can be trusted.

    `metrics.csv` carries the seed per row and the log carries it per boundary.
    They are written by different code paths, so nothing but a test stops them
    drifting apart.
    """
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    results_rel = "tests/_runner_seed_agreement"
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    config = _resolve_config(cache_root, results_rel)
    assert main(str(_write_config(config, tmp_path))) == 0

    experiment = _experiment_dir(results_rel)
    seeds = _logged_seeds(experiment / "experiment.log")
    df = pd.read_csv(experiment / "metrics.csv")

    assert not df.empty
    for _, row in df.iterrows():
        logged = seeds.get(("metric", row["metric"]))
        assert logged is not None, f"metric {row['metric']!r} logged no seed"
        assert int(row["seed"]) == logged, (
            f"{row['metric']}: metrics.csv says {row['seed']}, log says {logged}"
        )


@pytest.mark.integration
def test_a_logged_result_is_reproducible_through_the_package(
    tmp_path, monkeypatch, ucr_dataset_path
):
    """
    Read the seeds a run logged, reproduce its score with the package alone.

    This is the claim the `seed` column exists to support, and the only test that
    closes the loop: everything here comes either from the logfile or from the
    public API, never from runner internals. The explained subset is drawn at
    random, so the `samples` seed is load-bearing - which is what makes this a
    test of the logging change rather than of the explainer.

    The checkpoint is reloaded rather than retrained, so a mismatch points at the
    seeding rather than at training determinism.
    """
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    from xai4tsc import evaluate, generate_explanation, resolve_rng
    from xai4tsc.data import load_dataset
    from xai4tsc.models import load_model

    results_rel = "tests/_runner_seed_replay"
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    config = _resolve_config(cache_root, results_rel)
    # A random subset, so which samples get explained depends on the samples seed.
    config["explanation_config"]["default_settings"]["samples"] = {
        "type": "random",
        "count": 3,
    }
    assert main(str(_write_config(config, tmp_path))) == 0

    experiment = _experiment_dir(results_rel)
    seeds = _logged_seeds(experiment / "experiment.log")
    expected = pd.read_csv(experiment / "metrics.csv")
    assert len(expected) == 1, "this fixture configures exactly one metric row"
    expected_score = float(expected.loc[0, "score"])

    # Everything below uses only the package and the seeds read from the log.
    ds = load_dataset("GunPoint", cache_dir=cache_root / "datasets")
    splits, encoder = ds.split(
        train_split=0.8,
        val_split=0.1,
        encoder=LabelEncoder(),
        rng=resolve_rng(seeds[("split", "GunPoint")]),
    )
    test_data, test_labels = splits[1][0], splits[1][1]

    checkpoints = sorted((experiment / "GunPoint" / "FCN").glob("FCN_epoch_*.pt"))
    assert checkpoints, "the run wrote no checkpoint to replay"
    model = load_model(
        {"model": "FCN", "model_ckpt_path": str(checkpoints[-1])}, device="cpu"
    )

    explanation = generate_explanation(
        method="Integrated_Gradients",
        model=model,
        data=test_data,
        labels=test_labels,
        targets="predicted",
        params={"seed": seeds[("explainer", "Integrated_Gradients")]},
        encoder=encoder,
        indices=None,
        n_samples=3,
        prng=resolve_rng(seeds[("samples", "GunPoint")]),
        device="cpu",
        save_path=tmp_path / "replayed",
    )
    # Metrics score only the explained samples, so the arrays are subset to
    # `explanation.indices` exactly as the runner does.
    score = evaluate(
        model=model,
        metric="Complexity",
        explanation=explanation,
        data=test_data[explanation.indices],
        labels=test_labels[explanation.indices],
        metric_class_params={
            "normalise": True,
            "abs": True,
            "disable_warnings": True,
            "display_progressbar": False,
            "return_aggregate": True,
        },
        metric_call_params={"softmax": False},
        device="cpu",
        seed=seeds[("metric", "Complexity")],
    )
    score = float(np.asarray(score).ravel()[0])

    assert score == pytest.approx(expected_score, rel=1e-9, abs=1e-9), (
        f"replaying the logged seeds gave {score}, the run recorded {expected_score}"
    )

    # And the samples seed is load-bearing: a different one picks a different
    # subset, so the assertion above is not passing for free.
    other = generate_explanation(
        method="Integrated_Gradients",
        model=model,
        data=test_data,
        labels=test_labels,
        targets="predicted",
        params={"seed": seeds[("explainer", "Integrated_Gradients")]},
        encoder=encoder,
        indices=None,
        n_samples=3,
        prng=resolve_rng(seeds[("samples", "GunPoint")] + 1),
        device="cpu",
        save_path=tmp_path / "other",
    )
    assert list(other.indices) != list(explanation.indices), (
        "a different samples seed selected the same subset, so this test would "
        "pass even if the seed were ignored"
    )


# ── models.csv: test-set model performance ────────────────────────────────────


@pytest.mark.integration
def test_model_performance_reaches_an_aggregate_csv(
    ucr_dataset_path, tmp_path, monkeypatch
):
    """evaluate_model's summary is collected, not left in the model's own folder."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_models_csv"

    config = _resolve_config(cache_root, results_rel)
    main(str(_write_config(config, tmp_path)))

    experiment_dir = _experiment_dir(results_rel)
    models_csv = experiment_dir / "models.csv"
    assert models_csv.exists(), "models.csv missing"

    df = pd.read_csv(models_csv)
    assert not df.empty
    for col in ("dataset", "model", "epoch", "status", "detail"):
        assert col in df.columns
    assert (df["status"] == "ok").all(), f"unexpected statuses: {set(df['status'])}"
    # A binary problem, so the columns are the binary ones rather than a
    # formulation-neutral subset.
    assert {"accuracy", "auc", "pos", "neg"} <= set(df.columns)
    # The same numbers the model's own summary file carries - the aggregate is a
    # collection of them, not a recomputation.
    own = pd.read_csv(next(experiment_dir.rglob("*_test_results.csv")))
    assert str(df.loc[0, "accuracy"]) == str(own.loc[0, "accuracy"])

    # Per dataset too, mirroring metrics.csv.
    per_dataset = next(experiment_dir.glob("*/*_models.csv"))
    assert not pd.read_csv(per_dataset).empty


# ── the per-output (multi-label) formulation ──────────────────────────────────


def _multi_label_config(cache_path: Path, results_rel: str, target: str) -> dict:
    """Build a Synth_Multilabel_Freq run with one output per frequency."""
    return {
        "experiment_name": "test_run",
        "cache_path": str(cache_path),
        "results_rel_path": results_rel,
        "general": {"seed": 7, "reproducible": True, "device": "cpu"},
        "data_config": {
            "default_settings": {"train_split": 0.7, "val_split": 0.15},
            "datasets": [
                {
                    "dataset": "Synth_Multilabel_Freq",
                    "encode": "multihot",
                    "encode_params": {"classes": ["7Hz", "19Hz", "37Hz"]},
                    "use_predefined_splits": True,
                    # Restrict rather than regenerate: the generated signals are
                    # bit-pinned by the digest test, and the bursts have to stay
                    # intact for Time-Frequency AUC to have anything to localize.
                    "max_samples": 150,
                }
            ],
        },
        "train_config": {
            "default_settings": {
                "hyperparams": {
                    "epochs": 1,
                    "batchsize": 16,
                    "loss_func": "bcewithlogits",
                    "optimizer": "adam",
                },
                "train": True,
            },
            "models": [{"model": "FCN", "init_params": {"filters": [4, 8, 4]}}],
        },
        "explanation_config": {
            "default_settings": {
                "target": target,
                "samples": {"type": "random", "count": 6},
            },
            "explainers": [
                {
                    "method": "FreqRISE",
                    "domain": "fft",
                    "num_batches": 4,
                    "batch_size": 4,
                    "num_cells": 8,
                }
            ],
        },
        "evaluation_config": {
            "default_settings": {"metric_call_params": {"softmax": False}},
            "metrics": [{"metric": "Time-Frequency AUC"}],
        },
    }


@pytest.mark.integration
@pytest.mark.parametrize("target", ["all", "label"])
def test_per_output_formulation_end_to_end(tmp_path, monkeypatch, target):
    """One output per frequency, trained with BCE and scored against its own mask."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = f"tests/_runner_multilabel_{target}"

    config = _multi_label_config(cache_root, results_rel, target)
    main(str(_write_config(config, tmp_path)))

    experiment_dir = _experiment_dir(results_rel)
    df = pd.read_csv(experiment_dir / "metrics.csv")

    # One row per output rather than their mean: the per-frequency spread is the
    # diagnostic this formulation exists to expose, and averaging hid it.
    scored = df[df["status"] == "ok"]
    assert not scored.empty, f"no scored rows: {df.to_dict('records')}"
    assert set(scored["target"]) <= {"7Hz", "19Hz", "37Hz"}
    assert len(df) == 3, "one row per output, not one for the dataset"
    # Every scored row has to say what its denominator was: under a per-output
    # encoding it varies per output, so two scores are not otherwise comparable.
    assert (scored["samples"] > 0).all()
    # Time-Frequency AUC needs both a frequency explainer and ground truth, so a
    # number here is what proves the ground-truth scope reached the metric.
    assert not scored["score"].isna().any()

    explained = 6  # samples.count
    if target == "all":
        # Nothing is masked: every output is explained for every sample.
        assert set(scored["samples"]) == {explained}
    else:
        # Each output is scored only on the samples carrying it. A sample
        # carries 1.5 of the 3 frequencies on average, so the totals cannot all
        # reach the explained-sample count - and if they did, the mask would not
        # be doing anything.
        assert scored["samples"].sum() < 3 * explained
        assert (scored["samples"] <= explained).all()


@pytest.mark.integration
def test_a_per_output_checkpoint_reloads_with_its_activation(tmp_path, monkeypatch):
    """The reason output_activation lives on the constructor, end to end."""
    monkeypatch.chdir(Path(__file__).parent.parent.parent)
    cache_root = Path(os.environ.get("XAI4TSC_TEST_CACHE", "./tests/cache"))
    results_rel = "tests/_runner_multilabel_ckpt"

    config = _multi_label_config(cache_root, results_rel, "all")
    main(str(_write_config(config, tmp_path)))

    experiment_dir = _experiment_dir(results_rel)
    sidecars = list(experiment_dir.rglob("*_config.json"))
    assert sidecars, "no checkpoint sidecar written"
    recorded = json.loads(sidecars[0].read_text())["init_params"]
    # Three frequencies, so three independent outputs - not eight powerset
    # classes - and a sigmoid rather than a competing softmax.
    assert recorded["num_classes"] == 3
    assert recorded["output_activation"] == "sigmoid"

    checkpoint = str(sidecars[0]).replace("_config.json", ".pt")
    reloaded = load_model({"model": "FCN", "model_ckpt_path": checkpoint}, device="cpu")
    assert reloaded.output_activation == "sigmoid"
