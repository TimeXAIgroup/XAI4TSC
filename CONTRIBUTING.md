# Contributing to XAI4TSC

Thanks for your interest in improving XAI4TSC! This guide covers how to report
problems, set up a development environment, and add a new model, explainer, or
metric.

XAI4TSC is research software that re-implements published methods. Please also
read the [Disclaimer](https://github.com/TimeXAI-group/XAI4TSC/blob/main/docs/source/disclaimer.md)
— it explains why implementation correctness matters here and what we ask
contributors to do about it.

## Reporting bugs and requesting features

Before opening an issue, please search the
[existing issues](https://github.com/TimeXAI-group/XAI4TSC/issues) to avoid
duplicates.

- **Bugs** — use the *Bug report* issue template. A minimal reproducer (a small
  YAML config or a short package snippet), the expected vs. actual behaviour, and
  your environment (OS, Python version, install mode) make a bug far easier to
  fix.
- **Features** — use the *Feature request* template. Describe the use case and
  which area it touches (model / explainer / metric / runner / docs).

## Development setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv sync --group dev --group runner   # runtime, runner and dev tools in .venv
source .venv/bin/activate
```

To build the documentation locally, also install the docs group
(`uv sync --group dev --group docs --group runner`) and run `make html` from
`docs/`.

If you only want the package (no experiment runner), an editable install is
enough:

```bash
pip install -e .
```

### Dependencies

`pyproject.toml` is the single source of truth for dependencies: runtime
dependencies under `[project.dependencies]`, and the runner's own (pyyaml,
packaging), the dev and the docs tools as groups under `[dependency-groups]`.
The runner is not part of the published package, so its dependencies are not
either. `uv.lock` pins the exact resolved versions. After
changing a dependency, run `uv lock` and commit `uv.lock`; CI installs with
`uv sync --locked` and fails if the lockfile is out of date.

### Code style and the quality gate

- **Formatting & linting** are handled by [Ruff](https://docs.astral.sh/ruff/).
  The configuration lives in `pyproject.toml`.
- **Docstrings** follow the **numpy** convention and are required for public
  classes, methods, and functions; document all parameters.
- **Type hints** are expected on the public API.

Run the quality gate before every commit:

```bash
# Auto-fix what can be fixed, then format
ruff check --fix src/ experiment_runner/ tests/
ruff format src/ experiment_runner/ tests/

# Verify (this is what CI runs)
ruff check src/ experiment_runner/ tests/
ruff format --check src/ experiment_runner/ tests/
```

### Tests

The suite uses two pytest markers, declared in `pyproject.toml`:

- `unit` — fast tests with **no I/O and no training**; your constant development
  loop. They live in `tests/unit/`.
- `integration` — full-pipeline tests that **download real data and train
  models**. They live in `tests/integration/`.

```bash
pytest -m unit           # fast loop — run constantly
pytest -m integration    # slow — downloads data, trains
pytest                   # the whole suite
```

Coverage is **opt-in** so the fast loop stays fast; pass `--cov` when you want a
report:

```bash
pytest -m unit --cov=xai4tsc --cov=experiment_runner --cov-report=term-missing
```

Every change to behaviour needs a matching test. New code should come with a
`@pytest.mark.unit` test that follows the existing patterns in
`tests/unit/test_*.py`. Coverage is a guide, not a target — a covered line with a
weak assertion still proves nothing.

### The runner / package boundary

XAI4TSC keeps a strict separation between the standalone runner and the
importable package. Respecting it is the single most important architectural
rule:

- **`experiment_runner/`** owns all config parsing, orchestration, logging setup,
  and path construction. It is **never** imported by the package. Run it as a
  module from the repository root:

  ```bash
  python -m experiment_runner --conf experiment_runner/configs/example.yaml
  ```

  Modules inside the runner use **relative** imports (`from .config import ...`)
  and import the library with **absolute** imports (`from xai4tsc... import ...`).

- **`src/xai4tsc/`** is the config-agnostic package. Functions here take plain
  Python values, never YAML config dicts. Anything that unpacks a config key or
  builds a path from `config["results_rel_path"]` belongs in a runner adapter, 
	not the package.

## Adding a model, explainer, or metric

All three component types use the same idea: subclass the right base and pass an
instance to the package API. Nothing has to be registered. A built-in also gets an
entry in the fitting catalog `MODELS`/`EXPLAINERS`/`METRICS`, a plain name-to-class
dict for discovery and the docs, which the runner's lookup tables
(`experiment_runner/lookups.py`) are built from. A runner config reaches a class
outside the package through `class_name` + `class_path` on its entry, behind
`general.allow_external_code: true`. Write a unit test either way.

### A new model

- Subclass `ModelBase` (`src/xai4tsc/models/base.py`) and implement
  `forward(x)` taking a `(B, C, T)` tensor and returning `(B, num_classes)`
  **raw logits**. Training, prediction, evaluation, and checkpointing are
  inherited.
- Take `in_channels` and `num_classes` as constructor parameters (the runner
  auto-detects and injects them).
- A built-in goes into the `MODELS` catalog (`src/xai4tsc/models/models.py`),
  keyed lowercase.

### A new explainer

- Subclass an explainer base in `src/xai4tsc/xai/` (e.g. `ExplainerBase`,
  `GradientExplainer`, `PerturbationExplainer`). For a Captum method, subclass
  `GradientExplainer` and implement `_get_captum_attribution()`.
- Set the class attributes: `explanation_type`, `data_applicability`
  (a `DataType` set), and `explanation_domains` (a `Domain` set).
- Implement `explain(...)`, returning an array shaped `(n_samples, C, T)` — or
  `(n_classes, n_samples, C, T)` when explaining all classes.
- Declare options that configure the method as `__init__` arguments; the
  instance is kept on `Explanation.explainer`, and a metric that re-explains runs
  a deep copy of it, so the explainer must survive `copy.deepcopy`. Options a method takes per call go in
  `explain`'s `**kwargs`, which callers supply via `generate_explanation`'s
  `call_params` (a `call_params` block on a runner entry) and which are recorded
  on `Explanation.explainer_call_params` and replayed on re-explanation. Prefer
  named parameters: the runner can verify those, while a key reaching a `**kwargs`
  catch-all cannot be checked and so requires `general.allow_kwargs: true`.
- A built-in goes into the `EXPLAINERS` catalog (`src/xai4tsc/xai/explain.py`),
  keyed lowercase.

### A new metric

- Either subclass a Quantus metric directly (pass the class to
  `QuantusEvaluator`, e.g. `QuantusEvaluator(MyQuantusMetric)`) or subclass
  `EvaluatorBase` (`src/xai4tsc/evaluation/base.py`) for a custom evaluator.
  `evaluate()` takes the evaluator instance.
- The catalog key is the **human-readable capitalized display name**
	(e.g. `"AUC"`,  `"Time-Frequency AUC"`).
- Domain-aware metrics receive `explanation.transform` and/or
  `explanation.metadata` automatically **if** those names appear in the metric's
  `__init__` signature; a `**kwargs` catch-all is *not* enough.
- A built-in goes into the `METRICS` catalog
  (`src/xai4tsc/evaluation/evaluate.py`).

> **Re-implementing a published method?** Implement it from source unless it is 
> not license-compatible (the project is MIT). Cite the paper and repository
> in the docstring, and note any known deviation from the original — see
> the [Disclaimer](https://github.com/TimeXAI-group/XAI4TSC/blob/main/docs/source/disclaimer.md).

## Pre-PR checklist

Before opening a pull request, confirm:

- [ ] `ruff check` and `ruff format --check` are clean.
- [ ] A `@pytest.mark.unit` test was added for the change.
- [ ] `pytest -m unit` passes (and `pytest -m integration` if you touched the
      pipeline).
- [ ] The documentation is updated.
- [ ] If dependencies changed: `uv.lock` is updated (`uv lock`).
- [ ] For a re-implemented method: the paper is cited and any known deviation
      from the original is noted.

## Opening a pull request

1. Branch from the latest `main`.
2. Keep the change focused; write a clear summary and link any related issues.
3. Open the PR — the template pre-fills the checklist above.
4. CI must pass: the `Tests` workflow runs Ruff lint + format and the unit
   suite on every push and pull request.

## Releasing

Releases go through GitHub only: set the version in `pyproject.toml`, then
publish a GitHub Release with a matching tag (`v<version>`). The `Publish to
PyPI` workflow builds the package with `uv build`, checks that the tag matches
the version, and uploads it via Trusted Publishing.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](https://github.com/TimeXAI-group/XAI4TSC/blob/main/LICENSE).
