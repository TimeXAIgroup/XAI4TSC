# XAI4TSC

An evaluation framework for eXplainable AI (XAI) methods applied to Time Series Classification
(TSC), developed by the TimeXAI Research Group.

XAI4TSC has two independent use cases:

- **Standalone experiment runner**: Clone the repo, choose or adapt a YAML config, run experiments from
  the command line.
- **Importable Python package**: `pip install xai4tsc` and use the public API in your own code,
  notebooks, or scripts.

The documentation can be found here: https://timexaigroup.github.io/XAI4TSC/

---

## Installation

### Standalone (experiment runner)

We use Poetry for package management inside a Conda environment:

1. Clone the repository:
	```bash
	git clone https://github.com/TimeXAI-group/XAI4TSC.git
	cd XAI4TSC
	```
2. Create a python environment (choose one of the following):  
	2.1. Using python:
   ```bash
   python -m venv .venv			 # Use python 3.12 or 3.13
   source .venv/bin/activate
   ```
	2.2. Using conda:
	 ```bash
   conda env create          # picks up environment.yml
   conda activate xai4tsc
   ```

3. Install the dependencies (choose one of the following):  
	3.1. Using poetry:
   ```bash
   pip install poetry # only needed if the local python .venv is used
   poetry install
   ```
	3.2. Using pip and PyPI:
   ```bash
   pip install xai4tsc
   ```
	3.3. Using pip and a local installation:
   ```bash
   pip install -e . 
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
└── src/xai4tsc/                # Importable package
    ├── data/
    │   ├── datasets.py         # UcrUeaDataset, LocalDataset, SyntheticDataset
    │   └── ...                 # base classes, data loaders
    ├── models/                 # ModelBase, built-in models, registry
    ├── xai/                    # Explainer ABC, generate_explanation(), built-in explainers
    ├── evaluation/             # evaluate(), Quantus metric registry
    └── utils/                  # Shared utilities 
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

## Using XAI4TSC as a package

See our [Getting Started Example](examples/getting_started.py).

---

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
development setup, code style and quality gate, testing conventions, and the
pattern for adding a new model, explainer, or metric. By participating you agree
to the [Code of Conduct](CODE_OF_CONDUCT.md). The project is released under the
[MIT License](LICENSE).

---

## Disclaimer

XAI4TSC bundles **re-implementations** of models, explainers, and
metrics from the research literature, alongside thin wrappers around third-party
libraries (Captum, Quantus). These implementations have **not** been verified by
the original authors and may differ from the source papers or reference code in
ways that affect results.

Evaluation scores are an empirical, sometimes contested, proxy for explanation
quality — treat them as guidance, not ground truth. When using XAI4TSC for
research, cite the original papers, state which implementation you used, and,
where possible, validate against a reference implementation. See the
[full disclaimer](docs/source/disclaimer.md) in the documentation for details.
