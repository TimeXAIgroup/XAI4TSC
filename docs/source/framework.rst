Framework
=========

The experiment runner is the standalone use case for xai4tsc. Clone the
repository, write a YAML config, and run a full evaluation pipeline from the
command line — no Python scripting required. Results are saved automatically
to CSV files grouped by dataset and model.

Installation
------------

.. code-block:: bash

   git clone https://github.com/TimeXAI-group/XAI4TSC.git
   cd XAI4TSC
   conda env create          # picks up environment.yml
   conda activate xai4tsc
   python -m venv .venv
   source .venv/bin/activate
   poetry install

Getting Started
---------------

1. Copy ``experiment_runner/configs/example.yaml`` and edit it for your
   datasets, models, explainers, and metrics.
2. Run the experiment as a module from the repository root:

   .. code-block:: bash

      python -m experiment_runner.main --conf path/to/your/config.yaml
      python -m experiment_runner.main --conf path/to/your/config.yaml --debug

3. Results are written under ``results_rel_path/<experiment_name>/`` (the
   ``experiment_name`` from your config), organised as::

      results_rel_path/
      └── <experiment_name>/
          ├── metrics.csv            # all results
          ├── experiment.log         # run log
          └── <dataset>/
              ├── metrics.csv        # per-dataset results
              └── <model>/
                  ├── metrics.csv    # per-model results
                  ├── <model>_epoch_<n>.pt   # best checkpoint
                  └── explanations/  # per-sample relevance plots

   Every ``metrics.csv`` has the same columns:

   .. list-table::
      :header-rows: 1
      :widths: 14 86

      * - Column
        - Meaning
      * - ``dataset``
        - Dataset result identity
      * - ``model``
        - Model result identity
      * - ``explainer``
        - Explainer result identity (``WRAPPER-BASE`` for wrapper methods)
      * - ``metric``
        - Metric result identity
      * - ``status``
        - ``ok`` (a score was produced), ``nan`` (the metric ran but could not
          produce a number), ``error`` (the metric raised) or
          ``skipped-domain`` (the metric needs an explanation domain this
          explainer does not produce)
      * - ``score``
        - The aggregated score; empty unless ``status`` is ``ok``
      * - ``detail``
        - Why, for a non-``ok`` status

   Every combination the run attempts gets a row, so the file accounts for the
   whole configuration rather than only its successes. A finished run always
   writes the global ``metrics.csv``, header-only if nothing succeeded.

   **Result identities.** A row and its output directory are named after the
   config entry that produced them. Repeating an entry is supported: when two or
   more entries at the same level share a name, each gets a ``_1``, ``_2``, ...
   suffix in config order, so their results and artifacts stay separate. Names
   that occur once are used unchanged.

See ``experiment_runner/configs/master.yaml`` for a fully annotated reference
of every available config key and its default value.

Example
-------

The shipped ``experiment_runner/configs/example.yaml`` is the go-to demo and the
config the runner uses by default (``python -m experiment_runner.main``). It
needs no download: the synthetic ``freq_shapes`` dataset is shared with the
repository. It trains two model architectures (LeNet and FCN), explains them
with three time-domain methods (Integrated Gradients, Guided Backpropagation,
and TSHAP), and scores those explanations with the Complexity and Pixel-Flipping
metrics:

.. literalinclude:: ../../experiment_runner/configs/example.yaml
   :language: yaml

For the frequency / time-frequency counterpart — FreqRISE and the frequency
metrics on the same dataset — see
``experiment_runner/configs/example_frequency.yaml``. Full archive sweeps are in
``ucr_benchmark.yaml`` and ``uea_benchmark.yaml``.

API Reference
-------------

The experiment runner is built around a small set of modules:

- :mod:`experiment_runner.main` — entry point, orchestrates the pipeline
- :mod:`experiment_runner.config` — config resolution and validation
- :mod:`experiment_runner.cache` — caching layer for results
- :mod:`experiment_runner.evaluate` — evaluation routines
- :mod:`experiment_runner.explain` — explanation generation
- :mod:`experiment_runner.log_setup` — logging configuration

See the full :doc:`autoapi/experiment_runner/index` for complete details.
