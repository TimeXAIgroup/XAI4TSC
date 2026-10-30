Package
=======

The xai4tsc package is the importable use case. Install it with pip and use
the public API in your own code, notebooks, or scripts to load datasets, train
classifiers, generate explanations, and evaluate them quantitatively. The
catalogs (``MODELS``, ``EXPLAINERS``, ``METRICS``, ``SYNTHETIC_DATASETS``) list
the built-in components by name for discovery; the API itself takes the
objects.

Installation
------------

.. code-block:: bash

   # editable install from a local clone (not yet published to PyPI)
   pip install -e PATH/TO/XAI4TSC

Getting Started
---------------

The typical workflow follows four steps: load data, train a model, generate
explanations, and evaluate them. The example below shows the full sequence
using a UCR dataset and a built-in model.

Example
-------
.. literalinclude:: ../../examples/getting_started.py
   :language: python

API Reference
-------------

The package is organised into the following submodules:

- :mod:`xai4tsc.data` — dataset loading and splitting
- :mod:`xai4tsc.models` — model base class and implementations
- :mod:`xai4tsc.xai` — explainer base classes and methods
- :mod:`xai4tsc.evaluation` — evaluation metrics

Key functions
~~~~~~~~~~~~~

The three functions a package user reaches for, in pipeline order. Each is also
re-exported at the top level (e.g. ``from xai4tsc import generate_explanation``)
and from its submodule (e.g. ``from xai4tsc.models import load_model``).

==========================================================  ===============================================================
Function                                                     Purpose
==========================================================  ===============================================================
:func:`~xai4tsc.models.models.load_model`                   Reload a model from a checkpoint (or build one by registry name).
:func:`~xai4tsc.xai.explain.generate_explanation`           Produce an :class:`~xai4tsc.xai.Explanation` for chosen samples.
:func:`~xai4tsc.evaluation.evaluate.evaluate`               Score an explanation with an evaluator.
==========================================================  ===============================================================

Data comes first and has no loader function: construct the dataset class for
the data at hand and call its ``split`` method.

.. code-block:: python

   ds = xai4tsc.UcrUeaDataset("GunPoint", cache_dir="cache/datasets")
   # ds = xai4tsc.LocalDataset("data/my_dataset", name="my_dataset", layout="bct")
   # ds = xai4tsc.SynthMultilabelFreqDataset(cache_dir="cache/datasets")
   splits, encoder = ds.split(train_split=0.8, val_split=0.1)

Building and reloading models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build a **new** model, call its constructor and then
:meth:`~xai4tsc.models.base.ModelBase.configure`:

.. code-block:: python

   model = xai4tsc.FCN(in_channels=1, num_classes=2).configure(
       device="cpu", save_path="results/run_1"
   )

``device`` and ``save_path`` are properties of the machine and the run, not of
the architecture, so they are not constructor arguments. ``configure`` returns
the model, so it chains, and an omitted argument leaves the current value
alone. The methods that write
(:meth:`~xai4tsc.models.base.ModelBase.train_model`,
:meth:`~xai4tsc.models.base.ModelBase.evaluate_model`) raise rather than guess a
destination, so nothing lands in the working directory by accident.

To **reload** a model, use :func:`~xai4tsc.models.models.load_model`. It resolves
the class the weights belong to and reads the constructor arguments back out of
the ``config.json`` written beside the checkpoint, so the architecture does not
have to be restated:

.. code-block:: python

   model = load_model(
       {"model": "FCN", "model_ckpt_path": "results/run_1/FCN_epoch_12.pt"},
       device="cpu",
       save_path="results/run_1",
   )

``load_model`` can also build a new model from a registry name, which is the
route the YAML-driven experiment runner takes. From Python the constructor is
the more direct one. For a class that is not in ``MODELS`` and has no config
entry, load the weights yourself:

.. code-block:: python

   model = MyCNN(in_channels=1, num_classes=2).configure(device="cpu")
   model.load_state_dict(
       torch.load(path, weights_only=True, map_location="cpu")
   )

Built-in components
~~~~~~~~~~~~~~~~~~~~~

These are the concrete classes you select at runtime — by **registry key** in a
YAML config or in the package API (``load_model``), or, for explainers and metrics,
by constructing the class and passing the instance to ``generate_explanation`` or
``evaluate``. The class is where its parameters and behaviour are
documented. A custom component is a subclass of the matching base class,
passed as an instance; the experiment runner loads one from a source file
named by an entry's ``class_name`` and ``class_path``.

**Models** (``xai4tsc.MODELS``)

=============  ==========================================
Registry key   Class
=============  ==========================================
``fcn``        :class:`~xai4tsc.models.models.FCN`
``lenet``      :class:`~xai4tsc.models.models.LeNet`
``resnet``     :class:`~xai4tsc.models.models.ResNet`
``lstm``       :class:`~xai4tsc.models.models.LSTM`
``patchtst``   :class:`~xai4tsc.models.models.PatchTST`
``xlstm``      :class:`~xai4tsc.models.models.XLSTM`
=============  ==========================================

**Explainers** (``xai4tsc.EXPLAINERS``)

===========================  =========================================================================  ============
Registry key                 Class                                                                      Data
===========================  =========================================================================  ============
``integrated_gradients``     :class:`~xai4tsc.xai.feature_attribution.IntegratedGradientsExplainer`     agnostic
``guided_backpropagation``   :class:`~xai4tsc.xai.feature_attribution.GuidedBackpropagationExplainer`   agnostic
``deconvolution``            :class:`~xai4tsc.xai.feature_attribution.DeconvolutionExplainer`           agnostic
``deeplift``                 :class:`~xai4tsc.xai.feature_attribution.DeepLiftExplainer`                agnostic
``occlusion``                :class:`~xai4tsc.xai.feature_attribution.OcclusionExplainer`               agnostic
``tshap``                    :class:`~xai4tsc.xai.feature_attribution.TSHAPExplainer`                   time series
``sign``                     :class:`~xai4tsc.xai.wrappers.SignExplainer`                               as its base
``freqrise``                 :class:`~xai4tsc.xai.freqrise.FreqRISEExplainer`                           time series
``frequency``                :class:`~xai4tsc.xai.explanation_domains.FrequencyExplainer`               time series
``timefrequency``            :class:`~xai4tsc.xai.explanation_domains.TimeFrequencyExplainer`           time series
``random_frequency``         :class:`~xai4tsc.xai.random_baseline.RandomFrequencyExplainer`             time series
``random_timefrequency``     :class:`~xai4tsc.xai.random_baseline.RandomTimeFreqExplainer`              time series
===========================  =========================================================================  ============

**Metrics** (``xai4tsc.METRICS``)

Every ``METRICS`` value is a callable that produces an
:class:`~xai4tsc.evaluation.base.EvaluatorBase`. Most registry keys map to
`Quantus <https://github.com/understandable-machine-intelligence-lab/Quantus>`_ metric
classes (listed in ``xai4tsc.QUANTUS_METRICS``, e.g. ``"Complexity"``,
``"Faithfulness Correlation"``, ``"ROAD"``), adapted by the single
:class:`~xai4tsc.evaluation.base.QuantusEvaluator`, which takes the Quantus class
(``QuantusEvaluator(quantus.Complexity, disable_warnings=True)``); see the Quantus
documentation for those. The xai4tsc-native metrics are:

=========================================  =======================================================================================  ============
Registry key                               Class                                                                                    Data
=========================================  =======================================================================================  ============
``Frequency Perturbation``                 :class:`~xai4tsc.evaluation.frequency_evaluate.FrequencyEvaluator`                       time series
``Time-Frequency Perturbation``            :class:`~xai4tsc.evaluation.timefrequency_perturbation.TimeFrequencyEvaluator`           time series
``Time-Frequency Perturbation Gaussian``   :class:`~xai4tsc.evaluation.timefrequency_perturbation.TimeFrequencyEvaluatorGaussian`   time series
``Time-Frequency AUC``                     :class:`~xai4tsc.evaluation.timefrequency_auc.TimeFrequencyAUCEvaluator`                 time series
=========================================  =======================================================================================  ============

The Data column is the class's ``data_applicability`` (``xai4tsc.DataType``):
``agnostic`` methods work on any input type, ``time series`` ones only on time
series; it is declared, not enforced.

Data domain
~~~~~~~~~~~

A dataset declares the signal domain its arrays are *expressed in* via
:attr:`xai4tsc.data.DatasetBase.data_domain`, and that is the domain the model
trained on it consumes - the **model-native domain**. It is the time domain for raw
series, which is every dataset shipped here; a dataset of (real-valued) spectra
declares ``"frequency"``. A :class:`~xai4tsc.data.LocalDataset` takes it as a
constructor argument, because only the caller knows what is in the file, while
UCR/UEA data is always raw series and a synthetic dataset declares its own on the
class. Unlike ``layout``, which describes a file and is resolved away at load time,
this one travels: pass it to
:func:`~xai4tsc.xai.generate_explanation` and it lands on
:attr:`xai4tsc.Explanation.data_domain`, where it decides two things.

First, what a plain attributor's explanation domain *is*: relevance is assigned to
the features the model was given, so it comes out in the data's own domain, and
``Integrated_Gradients`` on a spectra-trained model yields a frequency explanation.
Explainers that deliberately change domain (``Frequency``, ``TimeFrequency``,
``FreqRISE``, the ``Random_*`` baselines) set
:attr:`~xai4tsc.xai.ExplainerBase.explains_in_data_domain` to ``False`` and name
their output instead.

Second, which explainers can read the data at all:
:attr:`~xai4tsc.xai.ExplainerBase.accepts_data_domains` declares the domains an
explainer can consume (empty means any). The domain-lifting explainers consume
time-domain data only, so pointing one at frequency data is refused rather than
transforming it a second time.

Metric domains
~~~~~~~~~~~~~~

Which explanation domains a metric can score is declared on the evaluator, and
:func:`~xai4tsc.evaluation.required_domains_for` is the reader. A metric defined
*for* a domain names it in ``required_domains`` (an empty set means any domain); one
that perturbs the input and re-queries the model instead sets
``requires_native_domain``, which ties it to the **model-native domain** - the domain
the model's inputs live in, :attr:`xai4tsc.Explanation.data_domain`. That is the time
domain for a model trained on raw series and the frequency domain for one trained on
spectra, so these metrics follow the model rather than being fixed to time.

For the Quantus metrics the choice follows from what each one consumes, recorded per Quantus class in
:data:`~xai4tsc.evaluation.quantus_groups.QUANTUS_GROUPS`:

=======================  ===========================  ===  ===========================
Group                    Its own code reads           n    Explanation domain
=======================  ===========================  ===  ===========================
``ATTRIBUTION_ONLY``     ``a_batch``                  3    any
``MODEL_PERTURBATION``   ``x_batch`` + ``model``      16   model-native
``LOCALISATION``         ``x_batch`` + ``s_batch``    6    model-native
``RE_EXPLAIN``           ``+ explain_func``           12   model-native
=======================  ===========================  ===  ===========================

So ``"Complexity"``, ``"Effective Complexity"`` and ``"Sparseness"`` score an
explanation in any domain, including one lifted out of the data's own (the adapter
supplies inputs on the same coefficient grid), while a metric that perturbs the input
needs the explanation in the domain the model predicts in.

The ``RE_EXPLAIN`` metrics re-run the explainer on perturbed or randomised inputs,
so they need a callable rather than just the stored attribution. The adapter builds
one from a deep copy of :attr:`xai4tsc.Explanation.explainer`, the instance that
produced the explanation, which is why the explanation keeps it: a re-explanation
has to use the *same* method with the *same* settings, or the metric measures a
configuration difference instead of the property it claims to. The copy leaves the
stored explainer untouched. :attr:`xai4tsc.Explanation.explainer_call_params` closes the same gap
for options a method takes per call rather than at construction: the adapter
replays them, and the ``explain_func_kwargs`` Quantus itself injects cannot
override them.

Seven metrics across the groups are refused rather than scored, each with its
reason in :data:`~xai4tsc.evaluation.quantus_groups.BLOCKED_BY_BACKEND`. Six of them
index a second spatial axis a ``(B, C, T)`` series does not have - super-pixels,
quadrants, 2-D padding, pixel neighbourhoods - and raise inside the library, so
refusing them replaces an obscure ``IndexError`` with a stated reason. The seventh,
``"Smooth MPRT"``, is the one worth knowing about: it *returns a number*, because it
calls the library's own explain function instead of the one it is given, so without
the entry it would report a plausible score for an explainer it never ran.

Ground-truth localization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``LOCALISATION`` metrics compare attributions to a supplied ground truth (GT).
That means the dataset has to supply the GT, as per-sample metadat. A localization 
metric on a dataset without GT is reported as ``skipped-no-ground-truth`` with the
reason, via :func:`~xai4tsc.evaluation.evaluate.ground_truth_missing`.

A dataset that does know records a list of **regions** under ``ground_truth`` in 
each sample's metadata. A region is **one interval**: ``pos`` and ``len`` define it,
and every other field is a scope that broadcasts over the whole of it.

====================  ================================================================
Field                 Meaning
====================  ================================================================
``pos``, ``len``      The interval, in samples. All a time-domain dataset needs
``channels``          Channels it applies to; omitted means all
``classes``           Classes it discriminates; omitted means all
``freqs``             Components in cycles per series - frequency / time-frequency only
``phases``            Paired with ``freqs``; recorded only
====================  ================================================================

Every field is optional and an omitted one means all, so a time-domain dataset
writes ``{"pos": ..., "len": ...}`` and the classes. A region must still localize
something, and an unrecognized field is an error rather than ignored - a misspelled
``len`` would otherwise read as *no* interval, which means "the whole series", and
quietly widen the mask over the entire sample.

Stored metadata is never rewritten to match this schema:
:meth:`~xai4tsc.evaluation._ground_truth.GroundTruthRegion.from_mapping` is the one
place that parses it, and it accepts the singular ``channel`` / ``freq`` / ``phase``
spellings and the per-class ``{class: [region, ...]}`` mapping alongside the forms
above.

An attribution is always for a class, so it is scored against the evidence for
that class. :attr:`xai4tsc.Explanation.explained_targets` records which class each
attribution explains, and
:func:`~xai4tsc.evaluation._ground_truth.regions_for_class` keeps only the regions
scoped to it; a region that declares no ``classes`` still applies to every class,
which is the right default for a single-label dataset whose sample label already
says which class it is. This matters most for a multi-label dataset: without it
every class would be scored against the union of the sample's regions, so each
class would get an identical mask.

Three consequences worth knowing when reading a score. Quantus requires its masks
to have a single channel, so a region's ``channels`` scope is unioned away on that
path and the score cannot say whether an attribution found the right channel, it
**warns** when that collapse discards something, which needs multivariate ground
truth. Samples with no region for the class in question are **dropped** rather 
than scored: an all-``False`` mask passes Quantus' own guard, which counts array
dimensions rather than elements, and would score 0 like a genuine result.

That dropping has a consequence for ``targets: predicted``: a sample carrying no
evidence for the class the model predicted is excluded, so the scored subset depends
on model accuracy. Use ``targets: label`` when the sample set has to be fixed across
models.

:meth:`~xai4tsc.evaluation.base.QuantusEvaluator.evaluate` also takes
``channel_first``, the axis order of the inputs, which defaults to ``True`` for the
framework's canonical ``(B, C, T)``. It is passed on to the Quantus metric; left
unset there it is inferred from the shape, and the guess is wrong for an input with
more channels than timesteps. Pass it through ``call_params`` when calling
:func:`~xai4tsc.evaluation.evaluate`, or set it under
``evaluation_config.default_settings.metric_call_params`` in a runner config.

Extending the package
~~~~~~~~~~~~~~~~~~~~~~~

If you want to add a custom dataset, model, explainer or evaluator, subclass the
relevant base class:

- :class:`xai4tsc.data.base.DatasetBase` — subclass to add a dataset
- :class:`xai4tsc.models.base.ModelBase` — subclass to add a model
- :class:`xai4tsc.xai.base.ExplainerBase` — subclass to add an explainer
- :class:`xai4tsc.evaluation.base.EvaluatorBase` — subclass to add a metric

See the full :doc:`autoapi/xai4tsc/index` for complete details.
