# XAI4TSC

An evaluation framework for eXplainable AI (XAI) methods applied to Time Series Classification
(TSC), developed by the TimeXAI Research Group.

XAI4TSC has two independent use cases:

- **Standalone experiment runner** — clone the repo, write a YAML config, run experiments from
  the command line.
- **Importable Python package** — `pip install -e .` and use the public API in your own code,
  notebooks, or scripts.

The documentation can be found here: https://timexaigroup.github.io/XAI4TSC/

---

## Installation

### Standalone (experiment runner)

We use Poetry for package management inside a Conda environment:

1. Install the Conda environment from `environment.yml`:
   ```bash
   conda env create   # environment.yml is picked up automatically
   ```
2. Activate the environment:
   ```bash
   conda activate xai4tsc
   ```
3. Create a local virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate the local environment:
   ```bash
   source .venv/bin/activate
   ```
5. Confirm setup — check which `python` and `poetry` your shell uses:
   ```bash
   which python   # should point to the local venv
   which poetry   # should point to the conda environment
   ```
6. Install dependencies with Poetry:
   ```bash
   poetry install   # use --dry-run to preview safely
   ```

### Package only

```bash
pip install xai4tsc
```

Or, for a local/editable install from a clone:

```bash
pip install -e PATH/TO/REPOSITORY
```

---

## Project layout

```
xai4tsc/
├── experiment_runner/          # Standalone CLI — owns all config and orchestration logic
│   ├── main.py                 # Entry point: python -m experiment_runner.main --conf ...
│   ├── config.py               # Config loading, validation, defaults
│   ├── cache.py                # Runner-level split caching helpers
│   ├── explain.py              # Runner adapter for xai4tsc.xai
│   ├── evaluate.py             # Runner adapter for xai4tsc.evaluation
│   ├── log_setup.py            # Logging setup for the standalone runner
│   └── configs/
│       ├── master.yaml         # Annotated reference — all available options and defaults
│       ├── example.yaml        # Go-to demo: synthetic data, time-domain explainers + metrics
│       ├── example_frequency.yaml  # Frequency / time-frequency showcase (FreqRISE, freq metrics)
│       ├── ucr_benchmark.yaml  # Full UCR sweep (LeNet + Integrated Gradients + Complexity)
│       └── uea_benchmark.yaml  # Full UEA sweep (skips OOM-risky datasets)
│
├── src/xai4tsc/                # Importable package
│   ├── data/
│   │   ├── datasets.py         # UcrUeaDataset, LocalDataset, SyntheticDataset
│   │   └── ...                 # base classes, data loaders
│   ├── models/                 # ModelBase, built-in models, registry
│   ├── xai/                    # Explainer ABC, generate_explanation(), built-in explainers
│   ├── evaluation/             # evaluate(), Quantus metric registry
│   └── utils/                  # Shared utilities (dict_to_args, merge_dicts, rescale_array, plot)
│
└── tests/
    ├── conftest.py             # Session-scoped fixtures (GunPoint download, split, model)
    ├── fixtures/
    │   └── test_config.yaml    # Minimal runner config for integration tests
    ├── unit/                   # Fast, no-I/O tests  (@pytest.mark.unit)
    └── integration/            # Full pipeline tests (@pytest.mark.integration)
```

---

## Running an experiment

Run the runner as a module from the repository root:

```bash
python -m experiment_runner.main --conf experiment_runner/configs/example.yaml
python -m experiment_runner.main --conf experiment_runner/configs/example.yaml --debug
```

Four ready-to-run configs ship under `experiment_runner/configs/`:

- `example.yaml` — the go-to demo (and the default when `--conf` is omitted):
  the synthetic `freq_shapes` dataset, two models, and time-domain explainers
  and metrics. Needs no download.
- `example_frequency.yaml` — the same dataset explained with FreqRISE in the
  frequency and time-frequency domains, scored with the frequency metrics.
- `ucr_benchmark.yaml` / `uea_benchmark.yaml` — full-archive sweeps with a
  minimal model/explainer/metric stack.

Edit or copy any of them to configure datasets, models, explainers, and metrics.
`master.yaml` contains annotated documentation for every available option.

---

## Testing

The test suite uses two pytest markers to separate fast unit tests from slow
integration tests that require a live dataset and a training run.

```bash
# Unit tests only — fast, no I/O, no training
pytest -m unit

# Integration tests only — downloads GunPoint on first run, trains a model
pytest -m integration

# Full suite
pytest
```

GunPoint is downloaded automatically on the first integration run and cached in
`tests/cache/` (override with the `XAI4TSC_TEST_CACHE` environment variable).

---

## Using xai4tsc as a package

```python
import xai4tsc
from xai4tsc.data import load_dataset, LocalDataset
from xai4tsc.models.models import load_model
from xai4tsc.xai.explain import generate_explanation
from xai4tsc.evaluation.evaluate import evaluate

# ── Load data (UCR download or local numpy files) ─────────────────────────────
ds = load_dataset("GunPoint")                           # UcrUeaDataset — downloads on first use
# ds = LocalDataset("/path/to/data", name="MyDataset")  # local data.npy + labels.json

splits, encoder = ds.split(
    train_split=0.8, val_split=0.1, random_state=42, encode="label"
)
train_data, train_labels, _ = splits[0]
test_data,  test_labels,  _ = splits[1]

# ── Train a model ─────────────────────────────────────────────────────────────
model = load_model(
    {"model": "FCN", "init_params": {"in_channels": 1, "num_classes": 2}},
    device="cpu",
)
model.train_model(
    train_data, train_labels,
    hyperparams={"epochs": 50, "batchsize": 32, "loss_func": "CrossEntropy",
                 "optimizer": "adam", "learn_rate": 0.001, "patience": 10},
    save_path="results",   # best checkpoint + training plots land here
)

# ── Generate explanations ─────────────────────────────────────────────────────
exp = generate_explanation(
    method="integrated_gradients",
    model=model,
    data=test_data,
    labels=test_labels,
    encoder=encoder,
    indices=[0, 1, 2],
    device="cpu",
)
# exp.exp_values — numpy array, same shape as test_data[[0, 1, 2]]

# ── Evaluate ──────────────────────────────────────────────────────────────────
score = evaluate(
    model=model,
    metric="Complexity",
    explanation=exp,
    data=test_data[exp.indices],
    labels=test_labels[exp.indices],
    metric_class_params={"normalise": True, "abs": True, "disable_warnings": True},
    device="cpu",
)

# ── Register a custom explainer ───────────────────────────────────────────────
from captum.attr import Saliency
from xai4tsc import GradientExplainer, register_explainer

class SaliencyExplainer(GradientExplainer):
    def _get_captum_attribution(self, model):
        return Saliency(model)

register_explainer("saliency", SaliencyExplainer)
exp2 = generate_explanation(
    "saliency", model=model, data=test_data,
    labels=test_labels, encoder=encoder, indices=[0], device="cpu",
)
```

---

## Built on

- **Computational backends:** PyTorch, scikit-learn
- **Explanation backend:** Captum
- **Evaluation backend:** Quantus

---

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
development setup, code style and quality gate, testing conventions, and the
pattern for adding a new model, explainer, or metric. By participating you agree
to the [Code of Conduct](CODE_OF_CONDUCT.md). The project is released under the
[MIT License](LICENSE).

---

## Disclaimer

XAI4TSC bundles **clean-room re-implementations** of models, explainers, and
metrics from the research literature, alongside thin wrappers around third-party
libraries (Captum, Quantus). These implementations have **not** been verified by
the original authors and may differ from the source papers or reference code in
ways that affect results.

Evaluation scores are an empirical, sometimes contested, proxy for explanation
quality — treat them as guidance, not ground truth. When using XAI4TSC for
research, cite the original papers, state which implementation you used, and,
where possible, validate against a reference implementation. See the
[full disclaimer](docs/source/disclaimer.md) in the documentation for details.
