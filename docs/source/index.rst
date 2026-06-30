.. XAI4TSC documentation master file, created by
   sphinx-quickstart on Thu May  7 11:45:44 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

XAI4TSC documentation
=====================

**XAI4TSC** is a Python framework for benchmarking eXplainable AI (XAI) methods on
time series classification (TSC) models. It covers the full evaluation pipeline:

- **Data** — load UCR/UEA datasets or local files, split, encode, and cache
- **Models** — train PyTorch classifiers with a unified ``ModelBase`` interface
- **Explanations** — generate feature attributions via Captum (Integrated Gradients, DeepLIFT, Occlusion, and more)
- **Evaluation** — quantify explanation quality with 38 Quantus metrics

XAI4TSC is designed for two use cases: as an importable **package** for programmatic
use in notebooks and scripts, and as a YAML-driven **experiment runner** for
large-scale reproducible benchmarks.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   framework
   package
   autoapi/index
   Contribute to XAI4TSC <contributing>
   disclaimer

.. note::
	This project is under active development.
