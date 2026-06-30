# Contributing to XAI4TSC

Thanks for your interest in improving XAI4TSC! This guide covers how to report
problems, set up a development environment, and add a new model, explainer, or
metric. By participating you agree to abide by our
[Code of Conduct](https://github.com/TimeXAI-group/XAI4TSC/blob/main/CODE_OF_CONDUCT.md).

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

Dependencies are managed with [Poetry](https://python-poetry.org/) inside a Conda
environment. From the repository root:

```bash
conda env create            # uses environment.yml -> creates the "xai4tsc" env
conda activate xai4tsc
python -m venv .venv
source .venv/bin/activate
poetry install --with dev   # runtime + dev tools (ruff, pytest-cov)
```

To build the documentation locally, also install the docs group
(`poetry install --with dev,docs`) and run `make html` from `docs/`.

If you only want the package (no experiment runner), an editable install is
enough:

```bash
pip install -e .
```

After adding new source files under `src/xai4tsc/`, reinstall in editable mode
(`pip install -e .`) so they are importable.

### Dependencies

`pyproject.toml` is the single source of truth for dependencies; `poetry.lock`
pins the exact resolved versions. `requirements.txt` is a **generated**
convenience for plain-pip users — it mirrors the direct runtime dependencies in
`[project.dependencies]` as version ranges (it is *not* a pinned lockfile
export).

If you change a dependency in `pyproject.toml`, regenerate `requirements.txt`
rather than editing it by hand:

```bash
python scripts/generate_requirements.py
```

The `Tests` workflow's `deps` job enforces this: it fails if `poetry.lock` is
out of sync with `pyproject.toml` (run `poetry lock`) or if `requirements.txt`
no longer matches `[project.dependencies]` (rerun the script above).

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
  python -m experiment_runner.main --conf experiment_runner/configs/example.yaml
  ```

  Modules inside the runner use **relative** imports (`from .config import ...`)
  and import the library with **absolute** imports (`from xai4tsc... import ...`).

- **`src/xai4tsc/`** is the config-agnostic package. Functions here take plain
  Python values, never YAML config dicts. Anything that unpacks a config key or
  builds a path from `config["results_rel_path"]` belongs in a runner adapter, 
	not the package.

## Adding a model, explainer, or metric

All three component types use the same idea: a plain-dict **registry** plus a
`register_*` function so new components can be added without editing framework
internals. Subclass the right base, register it, write a unit test, and (for
documentation) add it to the fitting registry `MODELS`/`EXPLAINERS`/`METRICS`.

### A new model

- Subclass `ModelBase` (`src/xai4tsc/models/base.py`) and implement
  `forward(x)` taking a `(B, C, T)` tensor and returning `(B, num_classes)`
  **raw logits**. Training, prediction, evaluation, and checkpointing are
  inherited.
- Take `in_channels` and `num_classes` as constructor parameters (the runner
  auto-detects and injects them).
- Register it: `register_model("my_model", MyModel)` — it lands in the `MODELS`
  registry (`src/xai4tsc/models/models.py`).

### A new explainer

- Subclass an explainer base in `src/xai4tsc/xai/` (e.g. `ExplainerBase`,
  `GradientExplainer`, `PerturbationExplainer`). For a Captum method, subclass
  `GradientExplainer` and implement `_get_captum_attribution()`.
- Set the class attributes: `explanation_type`, `data_applicability`
  (a `DataType` set), and `explanation_domains` (a `Domain` set).
- Implement `explain(...)`, returning an array shaped `(n_samples, C, T)` — or
  `(n_classes, n_samples, C, T)` when explaining all classes.
- Register it: `register_explainer("my_method", MyExplainer)` — keyed lowercase
  in the `EXPLAINERS` registry (`src/xai4tsc/xai/explain.py`).

### A new metric

- Either subclass a Quantus metric directly (the `QuantusEvaluator` wraps it
  automatically) or subclass `EvaluatorBase` (`src/xai4tsc/evaluation/base.py`)
  for a custom evaluator.
- The registry key is the **human-readable capitalized display name**
	(e.g. `"AUC"`,  `"Time-Frequency AUC"`).
- Domain-aware metrics receive `explanation.transform` and/or
  `explanation.metadata` automatically **if** those names appear in the metric's
  `__init__` signature; a `**kwargs` catch-all is *not* enough.
- Register it: `register_metric("My Metric", MyMetric)` — it lands in the
  `METRICS` registry (`src/xai4tsc/evaluation/evaluate.py`).

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
- [ ] If dependencies changed: `poetry.lock` is updated (`poetry lock`) and
      `requirements.txt` is regenerated (`python scripts/generate_requirements.py`).
- [ ] For a re-implemented method: the paper is cited and any known deviation
      from the original is noted.

## Opening a pull request

1. Branch from the latest `main`.
2. Keep the change focused; write a clear summary and link any related issues.
3. Open the PR — the template pre-fills the checklist above.
4. CI must pass: the `Tests` workflow runs Ruff lint + format and the unit
   suite on every push and pull request.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](https://github.com/TimeXAI-group/XAI4TSC/blob/main/LICENSE).
