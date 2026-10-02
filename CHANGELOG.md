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
- **Label encoding**: `split()` now always returns 1-D `int64` class
  indices for every `encode` value, and the fitted encoder is kept solely to map
  those indices back to class names. `encode` describes *how to read the label
  column*, not what the target array looks like. `encode: "onehot"` and
  `"ordinal"` previously produced a scipy sparse matrix and an `(n, 1)` float
  column respectively, because they used scikit-learn's **feature** encoders in a
  **target** slot; both crashed when the split cache was written, auto-detected
  `num_classes` wrongly, and left the index-to-name map empty. The encoders are
  still stock scikit-learn, adapted by two new helpers,
  `xai4tsc.data.decode_labels` and `xai4tsc.data.class_names`.
- **`classes` is now an ordered class list**, used with `encode: "ordinal"` to
  declare a meaningful order, e.g. `classes: ["low", "medium", "high"]`. Without
  it, classes order lexicographically, which for ordered string labels is wrong
  and is now warned about. Numeric labels need no order. The key was previously
  declared in `master.yaml` and never read.
- **`layout` on a local dataset** (`"auto"` | `"bct"` | `"btc"`) states the axis
  order instead of guessing it. `"auto"` keeps the previous heuristic and its
  warning; `"bct"` is the way to load data that genuinely has more channels than
  timesteps, which the heuristic would otherwise transpose into nonsense.
- **A config file beside every checkpoint**: `save_model()` now writes
  `<checkpoint>_config.json` next to each checkpoint it saves (e.g.
  `FCN_epoch_10.pt` -> `FCN_epoch_10_config.json`), recording the model name, its
  resolved constructor arguments, the `epoch` the weights represent, and
  `resumed_from` when the run continued an earlier checkpoint. `load_model`
  already read such a file, so a checkpoint can now be reloaded from its path
  alone, without restating the architecture (including the runner's auto-detected
  `in_channels` / `num_classes`) in the reload config. Being named after its
  checkpoint means the pairing is unambiguous when a directory holds several.
- **`load_model` now logs what it loaded**: the checkpoint path, which of the
  three resolution routes produced the class (the checkpoint's own `model_class`,
  the `MODELS` registry, or `class_name`/`class_path`), and how many epochs the
  weights have had. It also reports what the checkpoint's config file
  contributed, and warns when that file overrides a value the config set
  explicitly - previously the architecture actually used could differ from the
  one in the user's YAML with nothing recording it.
- **Resuming continues the epoch count.** `train_model` gained `start_epoch`,
  defaulting to the loaded checkpoint's epoch, so fine-tuning `FCN_epoch_10.pt`
  for 5 more epochs logs `Epoch 11/15` and writes `FCN_epoch_11.pt` through
  `FCN_epoch_15.pt`. Previously numbering restarted at 1, so the epoch in a
  filename no longer said how much training the weights had had, and a resume
  into an existing directory would overwrite the original run's checkpoint.
- **`format` and `path` are part of the split-cache key.** Previously a cached
  split was reused for a different loader or a different local directory, so a
  `format: csv` request could be served numpy data from another dataset that
  happened to share its name. Existing cache folder names are unchanged: the new
  tokens are emitted only for non-default values.

### Changed

- **Parameter defaults now live in the code, not in `master.yaml`.** The master
  config documents each model's, explainer's and metric's parameters in comments
  but no longer supplies them, and `config_resolve` no longer backfills per-entry
  values. A value you omit therefore comes from the class that consumes it, so the
  importable package and the runner can no longer disagree about what a method is.
  Two defaults had already diverged and are reconciled in the code:
  `Occlusion.strides` is now `1` (was `4` in code, `1` in master) and
  `FreqRISE.num_cells` is now `8` (was `50`, which raised for any STFT grid
  smaller than 100 bins). `master.yaml`'s `hyperparams` block now mirrors
  `t_con_defaults` exactly, and a test enforces all of this, including that the
  documenting comments stay true.
- **`master.yaml` is no longer runnable.** It is the annotated reference and the
  source of settings defaults; its four component lists are a catalogue of what is
  available. `experiment_name` and `results_rel_path` are `null`, and running it
  directly is refused with a pointer to `configs/example.yaml`.
- **The four component lists are required.** `data_config.datasets`,
  `train_config.models`, `explanation_config.explainers` and
  `evaluation_config.metrics` must be supplied, along with `experiment_name` and
  `results_rel_path`; a missing one raises before anything is written to disk,
  naming every setting that is absent. Setting a list to `[]` still skips that
  level deliberately and warns. Previously an omitted list was inherited from
  master, so leaving out `metrics:` silently ran all 41 of them and leaving out
  `models:` silently ran none.
- The global `metrics.csv` is always written, header-only when nothing
  succeeded, so a finished run always leaves a file with a known schema.
- `experiment_runner.main.main()` returns the total number of failures instead of
  `None`.
- `docs/source/framework.rst` now documents the `metrics.csv` column set, which
  was previously undocumented.
- **`train` is a switch**, documented as `train: true | false`. Its two former
  values, `"till_epoch"` and `"for_epochs"`, were never read by anything - they
  had no effect since the initial release - so both are now accepted as `true`
  with a deprecation warning, and any other string is a config error. The shipped
  configs use `train: true`. `hyperparams.epochs` is the number of epochs to run
  **in this call**; a resumed run continues its epoch count rather than adding to
  the budget.
- The module-level `save_model()` writes `model_dict_config.json` instead of
  `config.json`, so one naming convention covers every checkpoint. `load_model`
  still reads a directory-level `config.json` when present, so existing result
  trees keep loading.
- The runner detaches its run-level log handler when a run ends. Driving
  `main()` more than once in a single process previously left the handler
  attached, so every later run also wrote into the earlier runs' `experiment.log`.

### Fixed

- **One configuration report, repeated at the end.** Every config check now feeds a
  single collector, so unknown settings-block keys, deprecated values, unknown entry
  keys and unregistered names appear as one block with an honest count, directly
  after file logging is attached rather than scattered through the setup output. The
  warnings are restated after `--- Experiment Finished ---`, because a long run
  scrolls them out of view and the point of a warning is that it gets read. The
  empty-list warnings now reach `experiment.log` too: they are found before that file
  exists, so they are deferred rather than lost. Unknown settings keys also gained a
  did-you-mean suggestion (`'trainsplit'` -> `'train_split'`).
- **Console logging is colourised** when writing to a terminal: lines above `INFO`
  are tinted. The escape codes sit only at a line's edges, so output stays greppable 
	this includes patterns that span fields, such as `grep "WARNING:"`. Follows the `NO_COLOR`
  and `FORCE_COLOR` conventions, and log files are never coloured.
- **Config entries are validated, split by consequence.** Nothing checked the keys
  or names *inside* a `datasets` / `models` / `explainers` / `metrics` entry, which
  is where most real typos land: `nsteps` instead of `n_steps` was accepted,
  dropped, and the explainer silently ran with its own default. Entries are now
  checked against the keys that are actually read, and problems are collected into
  one report so a config with several mistakes takes one run to fix:
  - naming a component that is not registered is an **error** - including a
    misspelled *metric*, which previously only warned and silently shortened
    `metrics.csv`. The message suggests the nearest registered name, and for a
    case-only mismatch gives the exact spelling (metric names are case-sensitive
    display names, unlike model and explainer names).
  - a key the runner will never read is a **warning**: the entry still runs, just
    not as written. A parameter in the wrong place says where it belongs -
    explainers take theirs at the entry's top level, models under `init_params`,
    metrics under `metric_class_params` / `metric_call_params`.
- `general.reproducible: false` raised `UnboundLocalError` and silently skipped
  every dataset; the seeded and unseeded paths now both work. An unseeded run
  additionally **bypasses the split cache** in both directions, since a random
  draw cannot be described by a cache key - previously it would either have
  reused another run's splits or written a new cache directory per run. Seeded
  runs are unaffected: cache folder names and split outcomes are unchanged.
- **The metric GIF keys are documented.** `animate`, `animate_sample` and
  `animate_fps` are read by the runner but appeared nowhere in `master.yaml`.
- **Evaluating a pre-trained checkpoint without retraining now works.** The
  workflow was unreachable in both of its halves. `load_model` only built a model
  when the checkpoint carried a `model_class` key, which only the module-level
  `save_model()` writes - so the `<Model>_epoch_<N>.pt` files the runner actually
  produces failed to load, with an error naming class options the user had not
  used. A bare state dict is now resolved against the `MODELS` registry (and the
  sidecar above), a missing checkpoint path is reported with the checkpoints that
  do exist beside it, and an unresolvable one names what was tried. Separately,
  *omitting* `train` could not mean "do not retrain", because the entry inherited
  `train: till_epoch` from `default_settings`; a model entry supplying
  `model_ckpt_path` no longer inherits it, and the run logs that it is not
  training.

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
