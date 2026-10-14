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
                  ├── <model>_epoch_<n>_config.json  # how to rebuild it
                  └── explanations/  # per-sample relevance plots

   Every checkpoint is written with a config file named after it, so the two are
   unambiguously paired. It records the model name, the resolved ``init_params``
   - including the auto-detected ``in_channels`` and ``num_classes`` - the
   ``epoch`` the weights represent, and ``resumed_from`` when the run continued
   from an earlier checkpoint:

   .. code-block:: json

      {"model": "FCN",
       "init_params": {"in_channels": 1, "num_classes": 2},
       "epoch": 10}

   To score a trained model again without retraining it, omit the ``train`` key
   from the model entry and point ``model_ckpt_path`` at one of those
   ``<model>_epoch_<n>.pt`` files; the architecture comes from the config file
   beside it and does not have to be restated. Setting ``train`` on that entry
   fine-tunes instead, and a fine-tuning run **continues** the epoch count, so
   resuming ``FCN_epoch_10.pt`` with ``epochs: 5`` writes ``FCN_epoch_11.pt``
   through ``FCN_epoch_15.pt`` rather than restarting at 1.

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

   **Labels.** Whatever ``encode`` is set to, a split carries 1-D ``int64`` class
   indices, and the fitted encoder is kept only to map those indices back to class
   names. ``encode`` therefore describes how to *read* the label column:
   ``"label"`` for unordered categories, ``"ordinal"`` for ordered ones (declare
   the order with ``classes``), ``"onehot"`` for a column of one-hot rows, and
   ``"multihot"`` for multi-hot rows, which collapse to one index per distinct
   combination.

   **Required settings.** ``experiment_name``, ``results_rel_path`` and the four
   component lists -- ``data_config.datasets``, ``train_config.models``,
   ``explanation_config.explainers`` and ``evaluation_config.metrics`` -- must be
   supplied. The settings blocks describe *how* an experiment runs; the lists
   describe *what* it runs, which has no sensible default. Omitting one fails with
   a message naming it, before anything is written to disk. Set a list to ``[]`` to
   skip that level deliberately: the run then warns and completes.

   **Config checks.** Problems are split by consequence and reported together, so
   one run surfaces all of them. Naming a component that is not registered is an
   **error**: the entry cannot run, and the message suggests the closest registered
   name. Supplying a key the runner will never read is a **warning**: the entry
   runs, but not as written. That second case is where most typos land, because a
   parameter name that does not match is dropped and the component silently uses
   its own default. Note where parameters belong, which differs by entry kind:
   explainers take theirs at the entry's top level, models under ``init_params``,
   and metrics under ``metric_class_params`` or ``metric_call_params``. A parameter
   in the wrong place is reported with the place it belongs.

   **Where defaults live.** Every default belongs to the package parameter that
   consumes it -- a model's or explainer's constructor, the arguments of
   :meth:`~xai4tsc.models.base.ModelBase.train_model` for training
   hyperparameters, or those of :meth:`~xai4tsc.data.DatasetBase.split` and
   :func:`~xai4tsc.generate_explanation` for the pipeline's own settings, so a
   value you omit behaves identically whether you drive the pipeline from a YAML
   config or import the package.

   A component's parameters reach it through
   :func:`~xai4tsc.utils.dict_to_args`, which simply does not pass what you did
   not set. The settings in a ``default_settings`` block are read by the runner
   before it calls anything, so for those it keeps a table ``_PACKAGE_BACKED``
   in ``experiment_runner/defaults.py``, naming the parameter that owns each
   key, and reads the value off that signature at startup. A renamed parameter
   therefore fails immediately instead of leaving a stale copy behind. Keys the
   package never sees, such as the ``skip_*`` switches and the Quantus parameter
   blocks, are listed alongside as runner-owned, and for those ``master.yaml`` is
   the source. The two lists must together cover every key, which a test enforces.

   ``master.yaml`` documents all of them in comments but supplies only the
   runner-owned ones, so the two entry points cannot diverge; a test compares
   every documented value against the signature it came from.

``experiment_runner/configs/master.yaml`` is the annotated reference for every
available config key. It is **not runnable**: it lists each component as a
catalogue of what is available, so running it directly is refused with a pointer
to ``configs/example.yaml``. Copy the entries you need into your own config.

Example
-------

The shipped ``experiment_runner/configs/example.yaml`` is the go-to demo and the
config the runner uses by default (``python -m experiment_runner.main``). It
needs no download: the synthetic ``Synth_Multilabel_Freq`` dataset is generated
on first use and cached. It trains two model architectures (LeNet and FCN), explains them
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
