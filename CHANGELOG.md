# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
