# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Experiment-runner reliability release. The `metrics.csv` schema changes, so this
is a minor version bump; the importable `xai4tsc` package API is unchanged.

### Added

- **Summary rows**: Added a summary to an experiment log, listing successful, failed
  and skipped datasets/models/explainers/metrics.
- **Result identities**: repeating a `datasets:`, `models:`, `explainers:` or
  `metrics:` entry is now supported. Entries that share a display name each get a
  `_1`, `_2`, ... suffix in config order, used for output directories, log labels
  and `metrics.csv` rows alike. Names that occur once are unchanged, so existing
  configs write to exactly the same paths as before. Previously, repeated entries
  shared one directory and silently overwrote each other's checkpoints,
  evaluation plots and saved attribution arrays.
- **`status` and `detail` columns in `metrics.csv`**: every combination the run
  attempts now gets a row. `status` is `ok`, `nan` (the metric ran but produced
  no number), `error` (the metric raised) or `skipped-domain`, with `score` empty
  unless `ok` and `detail` carrying the reason. Metrics that returned all-NaN
  previously vanished from the results without a trace.
- **Per-component failure isolation**: a raising metric, explainer or model no
  longer aborts the rest of its dataset. Each failure is recorded and the run
  continues.
- The runner now **exits non-zero** when anything failed, and logs a closing
  summary naming the datasets completed, skipped and failed. A fully failed run
  previously exited 0 and was indistinguishable from a successful one.

### Changed

- The global `metrics.csv` is always written, header-only when nothing
  succeeded, so a finished run always leaves a file with a known schema.
- `experiment_runner.main.main()` returns the total number of failures instead of
  `None`.
- `docs/source/framework.rst` now documents the `metrics.csv` column set, which
  was previously undocumented.

### Fixed

- `general.reproducible: false` raised `UnboundLocalError` and silently skipped
  every dataset; the seeded and unseeded paths now both work.

## [1.1.0] - 2026-07-03

Feature release for public API to enable users to easily access the packages internal
logging system.

### Added

- **ssrc/xai4tsc/logging_config.py**: Provides top-level package functions for package
  users to access the packages internal logging system. This is now also used by the
  experiment runner.
- **Minor Changes**:
  - Refactored the getting_started example of the package to a python script
  - Added F1 and MCC scores for model evaluation
  - Unification of the save_path parameter types
  - Fixes a bug in UcrUeaDatatset path handling

[1.1.0]: https://github.com/TimeXAIgroup/XAI4TSC/releases/tag/v1.1.0

## [1.0.0] - 2026-06-30

First stable public release. The public API under the top-level `xai4tsc`
namespace is now considered stable and follows semantic versioning.

### Added

- **Dual-use design**: standalone experiment runner (`python -m experiment_runner.main`)
  and an importable, config-agnostic `xai4tsc` package.
- **Data** (`xai4tsc.data`): UCR/UEA loaders, pre-split local datasets, and a
  synthetic-dataset framework (`SyntheticDataset` ABC) with a shipped
  `freq_shapes` dataset, label/multi-hot encoders, and ground-truth
  localization metadata.
- **Models** (`xai4tsc.models`): `FCN`, `LeNet`, `ResNet` (Wang et al. 2017),
  `LSTM`, `XLSTM` (Beck et al. 2024), and `PatchTST` (Nie et al. 2023), all
  clean-room PyTorch implementations behind a common `ModelBase` wrapper.
- **Explainers** (`xai4tsc.xai`): Captum-backed time-domain attributions,
  frequency / time-frequency explainers (incl. a paper-faithful FreqRISE), and
  TSHAP, all behind the `Explainer` ABC with declared `data_applicability`.
- **Evaluation** (`xai4tsc.evaluation`): a unified `EvaluatorBase` design — the
  Quantus library is adapted through a single `QuantusEvaluator`, alongside native
  frequency / time-frequency perturbation metrics (`FrequencyEvaluator`,
  `TimeFrequencyEvaluator`) and the ground-truth `TimeFrequencyAUCEvaluator`
  localization metric, all with declared domain applicability.
- **Extensibility**: runtime registries (`MODELS`, `EXPLAINERS`, `METRICS`) with
  `register_*` hooks.
- Documentation, contribution guide, and a research disclaimer.

[1.0.0]: https://github.com/TimeXAIgroup/XAI4TSC/releases/tag/v1.0.0
