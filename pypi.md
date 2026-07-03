# XAI4TSC

An evaluation framework for eXplainable AI (XAI) methods applied to Time Series Classification
(TSC), developed by the TimeXAI Research Group.

XAI4TSC provides an importable Python package: `pip install xai4tsc` and use the public API in
your own code, notebooks, or scripts.

The documentation can be found here: https://timexaigroup.github.io/XAI4TSC/

---

## Installation

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
src/xai4tsc/                # Importable package
├── data/
│   ├── datasets.py         # UcrUeaDataset, LocalDataset, SyntheticDataset
│   └── ...                 # base classes, data loaders
├── models/                 # ModelBase, built-in models, registry
├── xai/                    # Explainer ABC, generate_explanation(), built-in explainers
├── evaluation/             # evaluate(), Quantus metric registry
└── utils/                  # Shared utilities
```

---

## Using XAI4TSC as a package

See our [Getting Started Example](https://github.com/TimeXAIgroup/XAI4TSC/blob/main/examples/getting_started.py).

---

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](https://github.com/TimeXAIgroup/XAI4TSC/blob/main/CONTRIBUTING.md)
for the development setup, code style and quality gate, testing conventions, and the
pattern for adding a new model, explainer, or metric. By participating you agree
to the [Code of Conduct](https://github.com/TimeXAIgroup/XAI4TSC/blob/main/CODE_OF_CONDUCT.md).
The project is released under the [MIT License](https://github.com/TimeXAIgroup/XAI4TSC/blob/main/LICENSE).

---

## Disclaimer

XAI4TSC bundles **re-implementations** of models, explainers, and
metrics from the research literature, alongside thin wrappers around third-party
libraries (Captum, Quantus). These implementations have **not** been verified by
the original authors and may differ from the source papers or reference code in
ways that affect results.

Evaluation scores are an empirical, sometimes contested, proxy for explanation
quality - treat them as guidance, not ground truth. When using XAI4TSC for
research, cite the original papers, state which implementation you used, and,
where possible, validate against a reference implementation. See the
[full disclaimer](https://timexaigroup.github.io/XAI4TSC/disclaimer.html) in the documentation for details.
