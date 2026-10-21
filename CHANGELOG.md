# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Experiment-runner reliability release. The `metrics.csv` schema changes,
`xai4tsc.Explanation` gains a required field, and label encoding, the loss and the
optimizer now take library objects instead of strings.

**Results from earlier versions will not reproduce exactly.** Randomness is now
derived per result rather than once per run, so model weight init, the split and
the explained-sample selection all draw differently. Statuses and row sets are
unchanged; scores move once. Split-cache folder names change once with them.

**Migration**: code that constructs an `Explanation` by hand must pass
`explainer_params` (use `{}` if there is nothing to record). Explanations produced
by `generate_explanation()` are unaffected. `train_model()` and `evaluate_model()`
now require a destination: pass `save_path=`, or build the model through
`load_model(..., save_path=...)`. Code that relied on the old fall-back to the
working directory has to name a directory. `ModelBase.load_from_checkpoint` is
removed: build the class and load the weights
(`MyCNN(...).configure(device=...)` then `load_state_dict`), or reload through
`load_model({"model": ..., "model_ckpt_path": ...})`, which also resolves the
class and reads the constructor arguments back from the checkpoint's config. The
`use_predefined_splits` moved from `UcrUeaDataset.__init__` to `split()`:
`UcrUeaDataset(name, use_predefined_splits=True)` becomes
`UcrUeaDataset(name).split(..., use_predefined_splits=True)`, and passing it to
`load_dataset()` now raises `TypeError` naming the new home rather than returning a
dataset that would split the other way. The
`rel_type_kwargs` key `bubble_size` is now `point_size`, matching the style it
sizes. `train_model()` and `evaluate_model()` take their training
hyperparameters as keyword arguments instead of a `hyperparams` dict
(`train_model(data, labels, epochs=50, batchsize=32, ...)`,
`evaluate_model(data, labels, batchsize=32)`), and `xai4tsc.models.base.t_con_defaults`
is removed with them - the defaults are now the signature. YAML configs keep
their `hyperparams:` block, which the runner unpacks. `generate_explanation()`
renames `samples` to `n_samples`, and `visualization_type` now defaults to the
tuple `("points",)` instead of `None`. `split()` and `load_saved_splits()` take
an `encoder` object in place of the `encode` string and `class_order`:
`split(encode="label")` becomes `split(encoder=LabelEncoder())`, or
`split()` - the default is unchanged. In YAML, `encode: "onehot"` and
`encode: "ordinal"` become `encode: "label"`, which reproduces their results, and
`data_config.classes` is removed. `train_model()` takes the loss and the optimizer
as objects: `train_model(loss_func="crossentropy", optimizer="adam")` becomes
`train_model(loss_func=nn.CrossEntropyLoss(), optimizer=optim.Adam)`, or
`train_model()` - the defaults are unchanged. YAML configs keep their strings,
including the mixed-case `"CrossEntropy"` the shipped configs use.

### Changed

- **Label encoding takes a scikit-learn encoder object, not a string.**
  `DatasetBase.split(encode="label", class_order=None)` becomes
  `split(encoder=None)`, and `load_saved_splits(directory, encode=...)` becomes
  `load_saved_splits(directory, encoder=...)`. `None` keeps the previous default
  (`LabelEncoder`), so behaviour is unchanged. The string stays a runner concept:
  `experiment_runner/encoders.py` maps `encode` to an object through a small table
  plus a dotted-path escape hatch, so `encode:
  "sklearn.preprocessing.LabelBinarizer"` works with no table entry, and an
  importing user can pass any encoder at all - which was previously impossible,
  since the four schemes were hard-coded in the package. `MultiHotLabelEncoder` is
  now exported from `xai4tsc` and `xai4tsc.data`.
  The encoder's output rank is now the target contract: 1-D means one class index
  per sample, 2-D means one column per model output. The integer dtype is
  checked rather than cast, so an encoder emitting floats is refused instead of
  silently truncated.
- **The loss and the optimizer are torch objects, not names.**
  `train_model(loss_func="crossentropy", optimizer="adam")` becomes
  `train_model(loss_func: nn.Module | None, optimizer: Callable[..., Optimizer] |
  Optimizer | None)`, both defaulting to `None`, which keeps
  `nn.CrossEntropyLoss` and `torch.optim.Adam`. The two closed whitelists - one
  loss and one optimizer, which raised on anything else - are deleted, so every
  `nn.*Loss`, every `torch.optim` optimizer, and any custom module or factory now
  works without editing the package. An optimizer is accepted as a class, as a
  callable (a `functools.partial` carrying constructor arguments), or as an
  instance built over parameter groups; in the last case `learn_rate` does not
  apply, and that is logged with the learning rate actually in force rather than
  silently ignored. The loss is moved to the model's device, so a `weight` tensor
  built on the CPU travels with it.
  The runner owns the strings, as it does for the encoder:
  `experiment_runner/losses.py` and `optimizers.py` hold small tables
  (`crossentropy`, `bcewithlogits`, `nll`; `adam`, `adamw`, `sgd`, `rmsprop`) plus
  a dotted-path escape hatch, so `loss_func: "torch.nn.MultiMarginLoss"` or
  `optimizer: "torch.optim.NAdam"` need no table entry. Names are matched
  case-insensitively, which is what keeps the shipped `"CrossEntropy"` configs
  working; a dotted path is not, because it has to import. An unresolvable name is
  reported during the config check, naming the model entry, before anything is
  downloaded. `experiment_runner/lookups.py` holds the resolution the three
  families share.
- **New `loss_params` and `optimizer_params` hyperparameters** carry constructor
  arguments: `loss_params: {weight: [1.0, 3.2, 0.8]}`,
  `optimizer_params: {momentum: 0.9}`. List values become `float32` tensors,
  because torch registers `weight` and `pos_weight` as buffers and refuses a list
  outright (`cannot assign 'list' object to buffer 'weight'`), and a `float64`
  tensor makes the loss raise at the first call. Class weighting therefore needs
  no framework feature of its own - it is a constructor argument of the loss.
  Note that torch's `weight` and `pos_weight` default to `None`, which means
  **unweighted, not balanced**: a majority class dominates in proportion to its
  frequency unless weights are given.
  These two are the first `hyperparams` keys that are not `train_model`
  parameters: the runner consumes them when it builds the objects. The accepted
  key list is now derived from both pointer tables rather than from
  `_PACKAGE_BACKED` alone.
- **The loss is checked against the targets before the first epoch.** Two layers,
  because torch will not do it: a smoke call on one batch, re-raised with both
  shapes and dtypes named, which turns a mid-training traceback into a startup
  error (including for a `weight` of the wrong length); and a specific rule for
  the one case a smoke call cannot catch - `nn.CrossEntropyLoss` **accepts** a 2-D
  float target, reading each row as soft class probabilities over a competitive
  softmax, so two co-occurring labels can each reach at most 0.5. It converges,
  the curve looks healthy and the objective is wrong, so an index-target loss
  against per-output targets now raises and names `bcewithlogits`. The check runs
  in eval mode under `no_grad` and is handed tensors rather than a loader - a
  `DataLoader` iterator seeds itself from the global torch generator, which would
  have shifted every later weight initialisation.
- **Targets follow their rank at the tensor boundary**: `int64` for 1-D class
  indices, `float32` for a 2-D array with one column per output, since
  `BCEWithLogitsLoss` rejects `int64` and `CrossEntropyLoss` rejects `float32`
  indices. The per-epoch accuracy follows too: a 2-D target is scored per output
  (`1 - Hamming loss`, logged as `bit_acc`) instead of being compared against an
  argmax, which raised when the batch size differed from the column count and
  silently reported 133% when they matched.
- **A free-form parameter mapping keeps the keys `master.yaml` cannot list.**
  `_merge_warn_drop` filtered every nested level against master and dropped what
  it did not find, so a Quantus parameter set in `default_settings` vanished:
  `metric_class_params: {nr_samples: 7}` resolved to master's five declared keys
  alone. The merge now stops its descent at the mappings listed as runner-owned -
  `metric_class_params`, `metric_call_params`, `loss_params`, `optimizer_params` -
  which is the rule `default_settings_leaves` already used. Master's own entries
  still apply underneath as defaults. The trade is deliberate: inside those four
  mappings a typo is no longer caught at merge time, because master cannot
  enumerate the parameters of ~37 metric classes or of an arbitrary torch loss.
- **`encode: "onehot"` and `encode: "ordinal"` are removed**, and both raise with a
  message naming `"label"`. They put scikit-learn's *feature* encoders in the
  label slot, which the library separates deliberately (`LabelEncoder`,
  `LabelBinarizer` and `MultiLabelBinarizer` are "not intended to be used on
  features, only on supervised learning targets"). `"onehot"` was also broken on
  the input it documented - a column of one-hot rows gave three class indices per
  sample - and on the scalar labels it actually received it produced codes
  identical to `"label"`, verified on ECG200, so `"label"` reproduces every result
  either of them produced.
- **An unknown `encode` is an error.** It used to fall back to `LabelEncoder` in
  silence, so `encode: "multhot"` trained a different problem and reported
  plausible numbers for it. The name is now resolved during the config sanity
  check, before anything is downloaded, and the message lists the available keys.
- **`use_predefined_splits` is a `split()` parameter, and selects any dataset's**
	**own splits.** It was a constructor argument of `UcrUeaDataset` alone, read in one
  place inside `split()` and nowhere else, while every other key that shapes a split
  - `train_split`, `val_split`, `stratify`, `encode` - is a `DatasetBase.split`
  parameter. It now joins them, so one signature owns the default and any dataset can
  build on it. `Synth_Multilabel_Freq` does: with the flag it applies the source
  project's published partition as before, and without it splits by the configured
  fractions like any other dataset, scoping its ground-truth regions either way.
  Previously it applied its own partition unconditionally and the flag never reached
  it at all. The flag defaults to `True` on that dataset, because its partition is
  part of the published data it reproduces and already includes a validation split,
  and to `False` on UCR/UEA, where the archive partition has none. `master.yaml`
  keeps `false`, so the shipped `example.yaml` and `example_frequency.yaml` now state
  `use_predefined_splits: true` on the dataset entry and behave exactly as before. A
  dataset with no partition of its own reports the request and splits normally, which
  also covers synthetic and custom datasets - the warning used to reach local
  datasets only.
- **Sample restriction no longer leaves per-sample metadata behind.**
  `DatasetBase.split` applied `max_samples` to the data and labels but not to
  `metadata`, which is one row per sample and paired with the data by position, so a
  dataset carrying metadata failed with `ValueError: zip() argument 2 is longer than
  argument 1` - a message naming neither `max_samples` nor metadata. The selected
  indices now apply to all three. Only `Synth_Multilabel_Freq` escaped it, by
  restricting in its own override; that override is gone and both share one path.
- **A predefined-split cache folder names changed holds.** The split-cache key
  interpolated `train_split`, `val_split` and the seed even under
	`use_predefined_splits: true`, which ignores all three - so a directory called
	`GunPoint_tr0.8_v0.1_s42_label_official` held the archive's own 50/150 split,
	and editing one of those dead keys minted a new key that re-split the data and
	wrote it back byte-identical. Those folders are now `GunPoint_label_official`.
	The seed is kept, as `GunPoint_s42_label_…`, when `max_samples` is set, because
	there it decides which rows were stored. Existing `*_official` caches no longer
	resolve and are rebuilt once; the old directories are left on disk and can be
	deleted. Random and stratified split names are unchanged, so every other cache
	stays valid.
- **`channels` and `length` are gone from `data_config.default_settings`.** Both
  were declared in `master.yaml` and read by nothing: the channel count is
  detected from the data and the series length comes from the dataset, so setting
  `length: 150` changed nothing while looking like it would truncate. Because
  master is also the key whitelist, a config that still sets either one is now
  warned about rather than silently ignored. A test holds the pointer tables and
  master to each other in both directions, so a key cannot go dead this way again.
- **`--help` names an invocation that exists.** The usage line read
  `usage: xai4tsc`, a console script this project does not install; it is now
  `usage: python -m experiment_runner`.
- **Defaults have one source: the package signature that consumes them.** Every
  `default_settings` value existed two to five times - in `master.yaml`, in the
  package, and often again as a `.get(key, literal)` fallback in the runner that
  could never fire - and the copies had drifted: `val_split` was `0.1` in master
  and `0.0` in `DatasetBase.split`, and `["points"]` was written in five places.
  The runner now derives each of those values from the parameter that consumes
  it, through a table naming the owner, so a renamed parameter fails at startup
  instead of leaving a stale literal behind. Keys the package never sees - the
  `skip_*` switches, `path`, `is_presplit`, the Quantus blocks - are listed as
  runner-owned, and `master.yaml` stays their source. A test requires every key
  to be in one list or the other.`master.yaml` keeps its values as documentation
	and gains descriptions for the keys that had none; a test holds them to the
	signatures. The one default that had to move to make this true is `val_split`,
	which is `0.1` everywhere now: a package caller who omits it gets a 10% 
	validation split where previously they got none. Every call site in the project
	already passed it, so no shipped config or test changes.
- **The config is resolved once, before anything validates or runs it.**
  `default_settings` used to be merged onto each entry only when the pipeline
  reached it, in four places, plus twice more in validators that needed an
  effective value early - and one of those merged in place, so validating a
  config rewrote it and left every entry sharing one `samples` dict with the
  defaults block. `config_resolve` now does it once: after it returns, every
  entry carries its effective settings, deep-copied, and the four merges in the
  pipeline and the two in the validators are gone. The resolved config logged at
  the start of a run therefore shows each entry exactly as it ran. One decision
  genuinely needs to tell a stated setting from an inherited one - a checkpoint
  entry must not retrain just because `default_settings` says `train: true` - and
  that now happens during resolution, which is the only place it is knowable.
- **A mistyped training hyperparameter is a config error.** `hyperparams:
  {epochss: 50}` reached `hyperparams.get("epochs", default)` and trained the
  default number of epochs without a word. The block is now validated against
  `train_model`'s signature, naming the entry and suggesting the near-miss, and
  an unknown key would raise at the call even if it got that far.

### Added

- **Per-call explainer options, recorded and replayed.** `ExplainerBase.explain`
  had declared a `**kwargs` bag documented as "additional explainer-specific
  options" since the first release, but nothing could ever reach it: no call site
  passed one, `generate_explanation` had no parameter to carry one, and the
  runner's entry validation accepted only constructor arguments. A contributor
  who declared a per-call option could therefore never receive a value for it.
  `generate_explanation` now takes `call_params`, an explainer entry takes a
  nested `call_params` block, wrapper explainers forward them to the method they
  wrap, and `Explanation` records them in the new `explainer_call_params` field.
  The re-explaining metrics replay that record, so an attribution is re-computed
  exactly as it was produced: without this, the first use of a call option would
  have made MPRT, Smooth MPRT, Random Logit and the rest silently compare two
  differently-configured runs of the same explainer and report the difference as
  a property of the method. Construction options stay where they are, at an
  entry's top level and in `explainer_params`. Nothing shipped reads a call
  option, so no existing result moves.
- **`general.allow_kwargs`, and parameter checking for every component.** A
  component whose signature ends in `**kwargs` accepts any key, so no inspection
  can say whether one is read: a misspelled explainer option was swallowed in
  silence and the attribution computed with a default, while a misspelled Quantus
  parameter raised inside the dataset loop after the model had trained. Such a key
  is now reported before the run starts, and `general.allow_kwargs: true` passes
  them through unchecked for configs that need to reach a catch-all. The check is
  keyed on **named** parameters rather than the presence of `**kwargs`, because
  every Quantus metric declares both, and it reaches blocks that had no validation
  at all: `init_params` on a model or synthetic dataset, and
  `metric_class_params` / `metric_call_params`, the last two checked against the
  Quantus class rather than the adapter that wraps it. Keys inherited from a
  section's `default_settings` are not blamed on the entries they do not apply to,
  which is how `softmax` reaches every metric although only the perturbation ones
  read it. Every shipped config passes with the flag at its default, and a test
  holds that.
- **The runner is executable by its package name.** `python -m experiment_runner`
  failed with `No module named experiment_runner.__main__`, so the only working
  form was the longer `python -m experiment_runner.main`, which is what every
  document taught. A `__main__.py` now delegates to a `cli()` extracted from
  `main.py`, the documentation names the short form, and the longer one keeps
  working. No console script: `experiment_runner` is deliberately not part of the
  installed package, so an entry point in `[project.scripts]` would ship a command
  that fails for everyone who installed from a wheel.
- **The synthetic dataset is now generated instead of shipped, and loads through the
  package API.** `freq_shapes` could only be read by the runner: its generator was
  a placeholder that raised, and the data was committed to the repository (11 MB)
  where a package install could never find it, so the documented
  `load_dataset(...).split(...)` path raised and told the caller to set a runner
  config key. The dataset is now `Synth_Multilabel_Freq`, generated on first use
  and cached, with its generator and its own stratified partition ported from
  [multilabel-ts-xai-benchmark](https://github.com/TimeXAIgroup/multilabel-ts-xai-benchmark)
  (MIT, recorded in `THIRD_PARTY_LICENSES.md`). At its defaults it reproduces that
  project's published dataset bit-for-bit - 4000 samples, 2800/600/600 - which a
  digest test pins. The committed copy and its `.gitignore` negation chain are
  gone.
- **Ground-truth regions can be annotated in Hz.** A region frequency is read as
  cycles per series, the unit a coefficient grid is indexed by. A dataset recording
  physical frequencies now states the `sampling_rate` they are relative to, on the
  region or once on the sample row, and `GroundTruthRegion.cycles()` converts.
  Omitting it keeps the previous meaning, so existing metadata is unaffected.
- **`ModelBase.configure()` sets a model's device and output directory.** These
  describe the machine and the run rather than the architecture, so they are not
  constructor arguments; until now the only thing that set them was a private
  helper reached through `load_model`, which left a model built with its own
  constructor no supported way to say where it should write.
  `FCN(...).configure(device="cpu", save_path=out)` returns the model, so it
  chains onto the constructor. `load_model` calls it on the caller's behalf, and
  an omitted argument leaves the current value alone.

### Fixed

- **FreqRISE follows its paper: the estimator weights each mask by the logit**
  , not the softmax probability. Brüsch et al. specify the score "prior to the softmax
  activation, since these show higher sensitivity to changes in the input" -
  softmax saturates, so on a confident model masking a genuinely important
  coefficient barely moves the probability while it moves the logit a lot, which
  compresses exactly the signal the method measures. The relevance is therefore
  signed now; that is harmless, because each map is min-max scaled to `[0, 1]`
  before it is returned and min-max is order-preserving, so the ranking metrics and
  plots read is unchanged.
- **`FreqRISE.probability_of_drop` is renamed `probability_of_keep`.** It always
  named the opposite of what it did: a coefficient is *kept* with that probability,
  and the estimator divides by the same value as `E[M]`. Invisible at the 0.5
  default, so a user asking to drop 90% of coefficients was keeping 90% of them
  instead.
- **The runner's own entry keys no longer reach an explainer's constructor.** An
  explainer takes its parameters at the entry's top level, so the dict handed to
  `build_explainer` is the whole resolved entry. `dict_to_args` filtered the
  runner's keys out against a named signature but returns the dict unchanged for a
  constructor declaring `**kwargs`, which then received `method`, `target`,
  `samples`, `visualization_type`, `save_exp_values` and the derived seed as if
  they had been configured for it. Latent, since no built-in explainer declares a
  catch-all constructor, but the new `allow_kwargs` would otherwise have made it
  arrive silently where an unknown-key warning used to hint at it. The keys the
  runner owns are now removed first, derived from the identity key, the structural
  extras and the resolved `default_settings` rather than listed; a key the class
  itself names is kept, and a wrapper's nested `base` keeps its `method`, which
  `WrapperExplainer` pops to build the method it wraps.
  `Explanation.explainer_params` consequently records the parameters the explainer
  was built with rather than the entry it came from, which is what it always
  documented. No score moves: the shipped explainers all declare named signatures,
  so those keys were already being dropped.
- **Three leftovers from the `bubbles` -> `points` rename.**
  `generate_explanation` still defaulted `visualization_type` to `["bubbles"]`
  while its own docstring, `plot_exp`, `_get_explanation` and the runner all said
  `"points"` - and since that default wins whenever the argument is omitted,
  every plot written through the defaulted package path was named
  `..._bubbles.png`. `add_relevance` defaulted to `"bubble"`, a third spelling,
  and the `rel_type_kwargs` key was still `bubble_size`. All three now say
  points. `"bubbles"` was never a value the dispatch matched, in any version, so
  nothing that worked before behaves differently.
  An unrecognised style still renders the points overlay rather than raising -
  that is deliberate and now documented at the dispatch and in both docstrings,
  with a test pinning it: a style name only selects an overlay, so a wrong one
  costs a re-render and touches no attribution, metric or saved array, and the
  name is echoed into the filename so a typo will be visible.
- **Impossible split fractions are refused at config time.**
  `train_split + val_split >= 1` leaves nothing for the test split:
  `_split_dataset` derived a zero or negative `rel_test`, sklearn rejected it,
  and the `except ValueError` around that call reported it as a stratification
  problem and retried before raising again - so the error named the wrong cause,
  after the dataset had been loaded. The fractions are now checked before
  anything is downloaded, naming the dataset and the arithmetic. Entries using
  the archive partition or an already-split directory are exempt, since they
  never reach the splitter.
- **Out-of-range sample indices are refused before the model trains.**
  `explanation_config.samples.indices` reached numpy as
  `IndexError: index 999999 is out of bounds for axis 0`, from inside the
  explainer, so after training had completed. The runner now checks the
  indices against the test split as soon as the split exists and before any
  model is built, and the package raises a named error at the point of use so a
  direct `generate_explanation` call is covered too. Config validation catches
  the data-independent part - a non-empty list of non-negative integers - up
  front, since it runs before any dataset is loaded and cannot know the split
  size.
- **An unknown `samples.type` silently explained the entire test split.** The
  dispatch fell through to an undocumented `"all"` branch, and because metrics
  are computed only on the explained samples, a typo quietly changed which
  subset every number in the run was scored over - in the most expensive
  direction. `"all"` is now documented in `master.yaml` and the three accepted
  values are validated, so a typo is a config error naming the near-miss.
- **The getting-started example trained a classifier worse than chance, and**
  **taught a call signature that did nothing.** It asked for an 80/10 split with
  `use_predefined_splits=True`, where `train_split`, `val_split` and `stratify`
  are ignored and no validation set is produced, so early stopping fell back to
  training loss and the shipped result was `spec=0.0000`, `TN=0`, `mcc=-0.083` -
  a one-class predictor, whose explanations the example then scored. It now
  takes a real 80/10/10 split, passes the validation split to `train_model`,
  trains to a usable accuracy, builds its model with the constructor and
  `configure()`, caches the download, and prints the `Complexity` score it
  computes instead of discarding it. An integration test runs it and asserts the
  model is not degenerate; nothing executed `examples/` before.
- **Split fractions the archive partition ignores are now reported.**
  `use_predefined_splits: true` discards `train_split`, `val_split` and
  `stratify` and produces no validation set, which was documented in a docstring
  and silent at runtime. Passing a non-default value for any of the three now
  warns, names the values, and states the consequence - early stopping monitors
  training loss without a validation split. `random_state` is deliberately not
  reported: it still seeds sample restriction. `uea_benchmark.yaml` asked for
  `val_split: 0.1` on this path and so implied a validation split it never got.
- **`load_model` blamed a missing checkpoint for an unregistered model name.**
  Its final branch raised `"No model checkpoint path supplied!"`, but it is
  reached whenever the `model` name is not in `MODELS` *and* no checkpoint was
  given - so a typo in the name was reported as a missing checkpoint. The error
  now names the unresolved value, lists the registered models, and says how to
  register or supply a class, matching what the checkpoint branch already did.
- **A checkpoint written by a directly built model could not be reloaded.** Only
  `load_model` recorded the constructor arguments, so a model built with its own
  constructor had nothing to write into the `_config.json` beside its weights, and
  `load_model({"model_ckpt_path": ...})` could not rebuild it without the caller
  restating `init_params`. Each built-in now records its own arguments, defaults
  included, with `self._record_init_params(locals())` as the first statement of its
  constructor - so the record is what the model was actually built with, and an old
  checkpoint still rebuilds the same architecture if a default changes later. The
  constructor is the only writer of that record now; `configure()` does not touch
  it. A custom model that omits the call still trains and saves weights, but gets
  no sidecar.
- **An unencodable constructor argument created a truncated checkpoint config.**
  The config was streamed straight to the open file, so a value `json`
  could not encode failed mid-write and left a partial file that reloads worse than
  no file at all. The config is now serialised in full before the file is opened,
  and the warning names the offending argument instead of reporting a bare
  `TypeError`.
- **A model built with its own constructor wrote its files into the working**
  **directory, under the name `None`.** `name` was set only by `load_model()`, so
  `FCN(in_channels=1, num_classes=2)` kept the class default of `None`, and
  `train_model()` fell back to the current directory when given no `save_path`.
  `name` now defaults to the class name for every model, including one reloaded 
	through `load_from_checkpoint()`, and the two methods that write raise a
	`ValueError` naming both ways to supply a destination rather than choosing one.
	The checkpoint sidecar is keyed on the recorded constructor arguments rather
	than on `name`, since an empty `init_params` would send `load_model` to a
	constructor call that raises. `load_model` is also exported at the top level
	now, next to `load_dataset`: the public namespace offered the model classes 
	but hid the factory that configures them.
- **An external model class lost the arguments needed to rebuild it.** The
  `class_path` branch of `load_model` overwrote the record `load_ext_model` had
  just made, so a checkpoint written from an externally loaded class carried no
  usable config. Setting the operational attributes no longer touches that record
  at all, so there is nothing left to overwrite.
- **`Time-Frequency AUC` scored against the wrong frequency bins, and so reported
  chance for every explainer.** The shipped ground truth records its frequencies in
  Hz (7/19/37 at 100 Hz over 500 samples) while the mask builder read them as
  cycles per series, putting the mask on bins 7/19/37 where the true bins are
  35/95/185.
- **Every result is now reproducible on its own.** `general.reproducible: true`
  used to seed numpy and torch once per run, after which the RNG advanced through
  every split, model and explanation - so a result depended on which entries ran
  before it, and re-running one dataset to check a number gave a different number.
  Each dataset, model, explainer and metric now derives its own seed from the run
  seed and its position in the config, logs it, and `metrics.csv` carries it in a
  new `seed` column so a row can be re-run from the row alone. Explained-sample
  selection derives from the dataset rather than the explainer, so every explainer
  in one run now scores the same samples and their rows are comparable.
  `reproducible: false` is unchanged: unseeded, split cache bypassed, seed column
  empty. `general.seed` and `general.reproducible` are now type-checked - a quoted
  `reproducible: "false"` is a non-empty string and was read as true.
- **`DEFAULT_SEED`, `derive_seed()` and `resolve_rng()`** are exported from the
  package, replacing three inconsistent conventions for an unspecified seed
  (`None`, `random_state=42`, `seed=0`) and four unseeded `default_rng()`
  fallbacks. `None` still means OS entropy. Each seed used can be found in the
	log files.
- **Robustness and randomisation metrics now work**: the twelve Quantus metrics that
  re-explain perturbed or randomised inputs failed because the framework never
  supplied an `explain_func`. `Explanation` now records `explainer_params`, so the
  evaluator rebuilds the explainer that produced the attribution and hands Quantus a
  callable for it.
- **Ground-truth localization metrics now work**: the six Quantus `LOCALISATION`
  metrics need per-sample masks marking where the class signal is, and nothing
  supplied them. `GroundTruthRegion` is now the schema for a discriminative region -
  one interval, with every other field a scope broadcasting over it - parsed from
  whatever a dataset stored, so committed datasets are not rewritten. A region needs
  only `pos` and `len`, which is all a time-domain dataset has; `freqs`/`phases` are
  the optional augmentation the frequency metrics use. Samples with no region are
  excluded from a score rather than counted as zero. A localization metric on a
  dataset that declares no ground truth is reported as `skipped-no-ground-truth`.
- **An attribution is scored against the class it explains.** `Explanation` records
  `explained_targets`, the class each attribution is for, and only the ground-truth
  regions scoped to that class are used. This matters for multi-label data, where a
  sample carries evidence for several classes: previously every class was scored
  against the union, so each got an identical mask. `freq_shapes` derives each
  region's class scope from its frequency, which is what its labels mean.
- **Unsupported metrics are refused with a reason** instead of raising from inside
  the library, and are reported as `skipped-unsupported`. Six index a second spatial
  axis a `(B, C, T)` series does not have (`Continuity Test`, `InputInvariance`,
  `Focus`, `IROF`, `Infidelity`, `ROAD`); `Smooth MPRT` calls the library's own
  explain function internally, so it would score an explainer it never ran. The only
  metrics still raising are the six localisation ones that need ground-truth masks.
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

### Removed

- **`ModelBase.load_from_checkpoint`.** It had no callers anywhere in the
  project, no test coverage, and left `self.device` as `None` while moving the
  weights onto that device, so the attribute disagreed with where the model
  actually was. It was `load_model` minus class resolution and minus reading the
  checkpoint's config, and the one case it covered - reloading an unregistered
  class from plain Python values - is two lines of public API. See the migration
  note above.

### Changed

- **An explanation is now a pure function of `(model, data, params, seed)`.** Every
  stochastic explainer rebuilds its generator inside `explain()` instead of holding
  one on the instance, so explaining the same input twice gives the same
  attribution. This is what the re-explaining metrics need, since they build one
  explainer and call it repeatedly. Vary the seed, or raise the sample count, to
  measure a method's own spread.
- **`TSHAP` and the two random baselines are reproducible at all for the first
  time.** They built `np.random.default_rng(seed)` with `seed=None`, which draws
  from OS entropy and cannot be pinned from outside - not even by
  `np.random.seed`, so the run seed never reached them.
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

- **`FreqRISE` no longer reseeds the global torch stream.** It called
  `torch.manual_seed` inside `explain()`, which rewound the process-wide generator
  for every later explainer, metric perturbation and the next dataset's weight
  init; `example_frequency.yaml` ships `seed: 0`, so one entry pinned the rest of
  the run. It now uses a local `torch.Generator`, and its own output for a given
  seed is unchanged.
- **A split-cache folder now describes the split stored under it.** The key
  carried a value drawn once per run while the split seed was drawn per dataset,
  so the same dataset in two positions wrote different splits under one key - and
  a later run could be served a split its own config would never have produced.

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
