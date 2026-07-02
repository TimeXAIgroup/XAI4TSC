Package
=======

The xai4tsc package is the importable use case. Install it with pip and use
the public API in your own code, notebooks, or scripts to load datasets, train
classifiers, generate explanations, and evaluate them quantitatively. All
three registries (``MODELS``, ``EXPLAINERS``, ``METRICS``) support runtime
extension via the ``register_*`` functions.

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

The four functions a package user reaches for, in pipeline order. Each is also
re-exported at the top level (e.g. ``from xai4tsc import generate_explanation``)
and from its submodule (e.g. ``from xai4tsc.models import load_model``).

==========================================================  ===============================================================
Function                                                     Purpose
==========================================================  ===============================================================
:func:`~xai4tsc.data.datasets.load_dataset`                 Load a UCR/UEA, local, or synthetic dataset.
:func:`~xai4tsc.models.models.load_model`                   Instantiate a model by name or from a checkpoint.
:func:`~xai4tsc.xai.explain.generate_explanation`           Produce an :class:`~xai4tsc.xai.Explanation` for chosen samples.
:func:`~xai4tsc.evaluation.evaluate.evaluate`               Score an explanation with a metric from the registry.
==========================================================  ===============================================================

Built-in components
~~~~~~~~~~~~~~~~~~~~~

These are the concrete classes you select at runtime — by **registry key** in a
YAML config or in the package API (``load_model``,
``generate_explanation(method=...)``, ``evaluate(metric=...)``). The key is what
you pass; the class is where its parameters and behaviour are documented. Extend
any registry at runtime with the matching ``register_*`` function.

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

==========================  ==================================================================================
Registry key                 Class
==========================  ==================================================================================
``integrated_gradients``    :class:`~xai4tsc.xai.feature_attribution.IntegratedGradientsExplainer`
``guided_backpropagation``  :class:`~xai4tsc.xai.feature_attribution.GuidedBackpropagationExplainer`
``deconvolution``           :class:`~xai4tsc.xai.feature_attribution.DeconvolutionExplainer`
``deeplift``                :class:`~xai4tsc.xai.feature_attribution.DeepLiftExplainer`
``occlusion``               :class:`~xai4tsc.xai.feature_attribution.OcclusionExplainer`
``tshap``                   :class:`~xai4tsc.xai.feature_attribution.TSHAPExplainer`
``sign``                    :class:`~xai4tsc.xai.wrappers.SignExplainer`
``freqrise``                :class:`~xai4tsc.xai.freqrise.FreqRISEExplainer`
``frequency``               :class:`~xai4tsc.xai.explanation_domains.FrequencyExplainer`
``timefrequency``           :class:`~xai4tsc.xai.explanation_domains.TimeFrequencyExplainer`
``random_frequency``        :class:`~xai4tsc.xai.random_baseline.RandomFrequencyExplainer`
``random_timefrequency``    :class:`~xai4tsc.xai.random_baseline.RandomTimeFreqExplainer`
==========================  ==================================================================================

**Metrics** (``xai4tsc.METRICS``)

Every ``METRICS`` value is a callable that produces an
:class:`~xai4tsc.evaluation.base.EvaluatorBase`. Most registry keys map to
`Quantus <https://github.com/understandable-machine-intelligence-lab/Quantus>`_ metric
classes (listed in ``xai4tsc.QUANTUS_METRICS``, e.g. ``"Complexity"``,
``"Faithfulness Correlation"``, ``"ROAD"``), adapted by the single
:class:`~xai4tsc.evaluation.base.QuantusEvaluator` (bound to the metric name); see the
Quantus documentation for those. The xai4tsc-native metrics are:

===========================================  ============================================================================
Registry key                                  Class
===========================================  ============================================================================
``Frequency Perturbation``                   :class:`~xai4tsc.evaluation.frequency_evaluate.FrequencyEvaluator`
``Time-Frequency Perturbation``              :class:`~xai4tsc.evaluation.timefrequency_perturbation.TimeFrequencyEvaluator`
``Time-Frequency Perturbation Gaussian``     :class:`~xai4tsc.evaluation.timefrequency_perturbation.TimeFrequencyEvaluatorGaussian`
``Time-Frequency AUC``                       :class:`~xai4tsc.evaluation.timefrequency_auc.TimeFrequencyAUCEvaluator`
===========================================  ============================================================================

Extending the package
~~~~~~~~~~~~~~~~~~~~~~~

If you want to add a custom dataset, model, explainer or evaluator, subclass the
relevant base class:

- :class:`xai4tsc.data.base.DatasetBase` — subclass to add a dataset
- :class:`xai4tsc.models.base.ModelBase` — subclass to add a model
- :class:`xai4tsc.xai.base.ExplainerBase` — subclass to add an explainer
- :class:`xai4tsc.evaluation.base.EvaluatorBase` — subclass to add a metric

See the full :doc:`autoapi/xai4tsc/index` for complete details.
