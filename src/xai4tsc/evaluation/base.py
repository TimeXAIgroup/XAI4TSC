"""Evaluator base classes: ``EvaluatorBase`` ABC and ``QuantusEvaluator``."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..utils.defaults import _fraction_params_T, eval_metric_defaults
from ..utils.perturbation import resolve_perturb_func
from ..utils.utils import dict_to_args
from ..xai._types import DataType, Domain

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from ..xai._types import Explanation

logger = logging.getLogger(__name__)

# Callable-typed params that can be specified as strings in config.
# A string value is resolved via _resolve_callable_perturb_func and dropped
# (with a warning) if unresolvable so the metric uses its own default.
# TODO: extend resolution to other callable params once a canonical resolution
#       strategy and signature tests are in place:
#       similarity_func, norm_numerator, norm_denominator, aggregate_func,
#       normalise_func, complexity_func, distance_func, discretise_func,
#       output_func, default_plot_func
_RESOLVABLE_CALLABLE_PARAMS: frozenset[str] = frozenset({"perturb_func"})


def _resolve_callable_perturb_func(value: str) -> Callable | None:
    """
    Resolve a string name to a perturbation callable.

    Thin wrapper over :func:`xai4tsc.utils.perturbation.resolve_perturb_func`
    (the single, shared resolver): a built-in baseline name, the top-level
    ``quantus`` namespace, then a dotted import path (e.g. ``"mymodule.fn"``).
    Returns ``None`` if unresolvable so the caller can drop the param and let
    the metric use its own default.

    Parameters
    ----------
    value : str
        Baseline name, function name, or dotted import path.

    Returns
    -------
    callable or None
    """
    return resolve_perturb_func(value)


def _apply_fraction_params(class_name: str, params: dict, data: np.ndarray) -> dict:
    """
    Convert fraction params (float in 0.0-1.0) to integers using series length T.

    Parameters
    ----------
    class_name : str
        Quantus class name, used to look up which params are fraction params.
    params : dict
        Param dict to process (not mutated).
    data : np.ndarray
        Input batch of shape ``(B, C, T)``.

    Returns
    -------
    dict
        Copy of *params* with fraction values replaced by integers.
    """
    fraction_keys = _fraction_params_T.get(class_name, set())
    if not fraction_keys:
        return params
    t = data.shape[-1]
    result = {}
    for k, v in params.items():
        if k in fraction_keys and isinstance(v, float) and 0.0 < v <= 1.0:
            result[k] = max(1, int(v * t))
        else:
            result[k] = v
    return result


def _resolve_string_callables(params: dict, class_name: str) -> dict:
    """
    Replace string values for callable params with resolved callables.

    Unresolvable strings are dropped and a warning is logged so the metric
    falls back to its own default.

    Parameters
    ----------
    params : dict
        Param dict to process (not mutated).
    class_name : str
        Quantus class name, used in warning messages.

    Returns
    -------
    dict
        Copy of *params* with strings resolved (or dropped).
    """
    result = {}
    for k, v in params.items():
        if k in _RESOLVABLE_CALLABLE_PARAMS and isinstance(v, str):
            resolved = _resolve_callable_perturb_func(v)
            if resolved is not None:
                result[k] = resolved
            else:
                logger.warning(
                    "Could not resolve '%s' for param '%s' of %s — param "
                    "dropped, metric uses its own default.",
                    v,
                    k,
                    class_name,
                )
        else:
            result[k] = v
    return result


class EvaluatorBase(ABC):
    """
    Base class for all XAI evaluation metrics.

    Subclass this and implement :meth:`evaluate`, then register the subclass with
    :func:`xai4tsc.register_metric` to make it available by name. The registry value
    is a *callable producing an evaluator*, so a subclass (instantiated with its
    constructor params) satisfies the contract directly.

    Example::

        class MyEvaluator(EvaluatorBase):
            metric_name = "my_metric"

            def evaluate(self, model, explanation, data, labels, device="cpu", **kw):
                return float(np.mean(np.abs(explanation.exp_values)))

        xai4tsc.register_metric("MyMetric", MyEvaluator)

    Domain applicability
    --------------------
    Like explainers, evaluators declare where they apply via Quantus-style class
    attributes: :attr:`data_applicability` (a set of ``DataType`` members) and, on two
    axes, which explanation domains they can score.

    - :attr:`required_domains` names the domains outright (an empty set means any),
      which is what a metric defined *for* a domain uses, e.g. a frequency
      perturbation curve.
    - :attr:`requires_native_domain` instead ties the metric to the **model-native
      domain**: the domain the model's inputs live in
      (:attr:`xai4tsc.Explanation.data_domain`). A metric that perturbs the input and
      re-queries the model needs this, because that is the space it can predict in -
      the time domain for a model trained on raw series, the frequency domain for one
      trained on spectra.

    Read the two together via
    :func:`~xai4tsc.evaluation.evaluate.required_domains_for`, which resolves the
    native rule against a given data domain; the runner skips a metric whose result
    does not contain the explanation's domain.

    Contract — attributions are real-valued
    ---------------------------------------
    Metric backends operate on **real** attributions: Quantus preprocessing and the
    perturbation metrics' ``np.argsort`` are undefined on complex arrays. Frequency /
    time-frequency explanations may be **complex** (the explanation-space wrappers
    transform the relevance itself), while others in the same domain are real
    (FreqRISE, random baselines). Backend evaluators must therefore pass attributions
    through :meth:`real_attribution` before handing them to the underlying metric.
    """

    metric_name: str = ""
    """Human-readable name of the metric."""

    data_applicability: ClassVar[set[DataType]] = {DataType.AGNOSTIC}
    """Data domains this evaluator applies to (Quantus-style; defaults to agnostic)."""

    required_domains: ClassVar[set[Domain]] = set()
    """Explanation domains this evaluator requires; empty means any domain."""

    requires_native_domain: ClassVar[bool] = False
    """Whether the explanation must be in the model-native domain (see above)."""

    @staticmethod
    def real_attribution(a_batch: np.ndarray | None) -> np.ndarray | None:
        """
        Reduce a complex attribution to its magnitude; pass real arrays through.

        Frequency / time-frequency explanations may be complex; metric backends
        operate on real values, so complex attributions are reduced to magnitude
        (``np.abs``). Real attributions are returned unchanged so the metric's own
        ``abs`` handling still governs them.

        Parameters
        ----------
        a_batch : np.ndarray or None
            The attribution array (possibly complex), or ``None``.

        Returns
        -------
        np.ndarray or None
            A real-valued attribution, or ``None`` if *a_batch* was ``None``.
        """
        if a_batch is not None and np.iscomplexobj(a_batch):
            return np.abs(a_batch)
        return a_batch

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        explanation: Explanation,
        data: np.ndarray,
        labels: np.ndarray,
        device: str = "cpu",
        **kwargs: object,
    ) -> float | np.ndarray:
        """
        Run the evaluation metric.

        Parameters
        ----------
        model :
            The PyTorch model being explained (an :class:`~torch.nn.Module`).
        explanation :
            :class:`~xai4tsc.xai.Explanation` dataclass containing
            ``exp_values``, ``data``, ``labels``, and ``encoder``.
        data : np.ndarray
            Input samples to evaluate on.
        labels : np.ndarray
            Ground-truth labels.
        device : str
            Compute device.
        **kwargs :
            Additional metric-specific options.

        Returns
        -------
        float or np.ndarray
            Metric score(s).
        """


class QuantusEvaluator(EvaluatorBase):
    """
    The single adapter for every `Quantus`_ metric.

    .. _Quantus: https://github.com/understandingai/Quantus

    Initialized with a metric **name** (a key of the ``QUANTUS_METRICS`` lookup table),
    not a class — so the registry can map every Quantus metric in via one class, e.g.
    ``partial(QuantusEvaluator, name)``.

    Example::

        evaluator = QuantusEvaluator(
            "Faithfulness Correlation",
            normalise=True,
            abs=True,
            disable_warnings=True,
        )
        score = evaluator.evaluate(model, explanation, data, labels)

    Domain applicability
    --------------------
    Which explanation domains a Quantus metric can score depends on what the metric
    reads: one that reads only the attribution applies in any
    domain, while one that perturbs the input and re-queries the model needs the
    model-native domain. :attr:`requires_native_domain` is therefore set per metric
    *name* from :data:`~xai4tsc.evaluation.quantus_groups.QUANTUS_GROUPS`; the class
    attribute is the conservative ``True`` default.
    """

    requires_native_domain: ClassVar[bool] = True
    """Conservative default; the instance relaxes it per metric name."""

    def __init__(self, name: str, **metric_class_params: object) -> None:
        """
        Instantiate the evaluator for a named Quantus metric.

        Parameters
        ----------
        name : str
            A key of the ``QUANTUS_METRICS`` lookup table (the metric's display name).
        **metric_class_params
            Keyword arguments forwarded to the resolved Quantus class ``__init__``.
            Float values between 0.0 and 1.0 for params listed in
            ``_fraction_params_T`` are treated as fractions of series
            length T and converted to integers at evaluation time.
            String values for ``perturb_func`` are resolved to callables
            from the ``quantus`` namespace or via dotted import path.

        Raises
        ------
        ValueError
            If *name* is not a registered Quantus metric.
        """
        from .evaluate import QUANTUS_METRICS
        from .quantus_groups import quantus_requires_native_domain

        if name not in QUANTUS_METRICS:
            raise ValueError(
                f"Unknown Quantus metric '{name}'. "
                f"Valid names: {sorted(QUANTUS_METRICS)}."
            )
        self._metric_class = QUANTUS_METRICS[name]
        self._metric_class_params = metric_class_params
        self.metric_name = name
        self.requires_native_domain = quantus_requires_native_domain(name)

    def evaluate(
        self,
        model: nn.Module,
        explanation: Explanation,
        data: np.ndarray,
        labels: np.ndarray,
        device: str = "cpu",
        channel_first: bool = True,
        **call_params: object,
    ) -> float | np.ndarray:
        """
        Instantiate the Quantus metric and run it.

        The metric is reinstantiated on each call to avoid state leakage
        between evaluations.

        Parameters
        ----------
        model :
            The model being explained.
        explanation : Explanation
            The explanation to score; supplies the attribution and its domain.
        data : np.ndarray
            Input samples in the framework's canonical ``(B, C, T)`` order.
        labels : np.ndarray
            Target labels.
        device : str
            Compute device.
        channel_first : bool
            Axis order of *data*, passed straight to the Quantus metric. The default
            matches the framework's canonical ``(B, C, T)``; set it to ``False`` for
            channel-last inputs. Left to Quantus it is guessed from the shape, and
            the guess is wrong for an input with more channels than timesteps.
        **call_params
            Further keyword arguments for the Quantus metric's ``__call__``,
            filtered to its signature.

        Returns
        -------
        float or np.ndarray
            The metric's score(s).
        """
        class_name = self._metric_class.__name__

        # 1. Fill in framework defaults for params the user didn't specify.
        raw_defaults = eval_metric_defaults.get(class_name, {})
        auto_params = {
            k: v(data) if callable(v) else v
            for k, v in raw_defaults.items()
            if k not in self._metric_class_params
        }
        auto_params = _apply_fraction_params(class_name, auto_params, data)

        # 2. Process user-supplied params: convert fractions and resolve callables.
        user_params = _apply_fraction_params(
            class_name, dict(self._metric_class_params), data
        )
        user_params = _resolve_string_callables(user_params, class_name)

        merged = {**auto_params, **user_params}
        init_params = dict_to_args(merged, self._metric_class.__init__)
        metric = self._metric_class(**(init_params or {}))

        # The axis order is stated rather than left to Quantus, which infers it from
        # the shape and gets it wrong for an input with more channels than timesteps.
        filtered_call = (
            dict_to_args(
                {"channel_first": channel_first, **call_params}, metric.__call__
            )
            or {}
        )

        # Metric backends operate on real attributions; reduce complex (freq/TF)
        # explanations to magnitude, leaving real ones untouched.
        a_batch = self.real_attribution(explanation.exp_values)
        return metric(
            model=model,
            x_batch=self._domain_inputs(explanation, data),
            y_batch=labels,
            a_batch=a_batch,
            device=device,
            **filtered_call,
        )

    @staticmethod
    def _domain_inputs(explanation: Explanation, data: np.ndarray) -> np.ndarray:
        """
        Return the inputs expressed in the explanation's own domain.

        Quantus asserts that the attribution and the input correspond, so an
        attribution on a coefficient grid needs inputs on that same grid. The values
        are never read by the metrics this path serves (they are the attribution-only
        group, see :mod:`~xai4tsc.evaluation.quantus_groups`); the grid is.

        *data* is already in the right domain whenever the explanation is in the
        model-native one (``explanation_domain == data_domain``), which covers both a
        time-domain explanation of a time-domain model and a frequency explanation of
        a model trained on spectra. Only an explanation lifted *out* of the data's
        domain needs transforming.

        Parameters
        ----------
        explanation : Explanation
            The explanation being scored; supplies both domains and the transform.
        data : np.ndarray
            Inputs in the model-native domain, shape ``(B, C, T)`` in the time domain.

        Returns
        -------
        np.ndarray
            *data* unchanged when the explanation is in the data's own domain, else
            the magnitude of its coefficients in the explanation's domain.

        Raises
        ------
        ValueError
            If the explanation is in another domain and carries no transform, or if
            its domain is not reachable from the data's by a forward transform.
        """
        import torch

        if explanation.explanation_domain is explanation.data_domain:
            return data
        transform = explanation.transform
        if transform is None:
            raise ValueError(
                "Scoring a "
                f"{explanation.explanation_domain.value} explanation requires a "
                "domain transform; none was available on the explanation."
            )
        if explanation.data_domain is not Domain.TIME:
            raise ValueError(
                "Inputs can only be lifted from the time domain, but the data is "
                f"{explanation.data_domain.value} and the explanation is "
                f"{explanation.explanation_domain.value}."
            )
        coefficients = transform.forward(torch.as_tensor(np.asarray(data)))
        return np.abs(coefficients.detach().cpu().numpy()).astype(np.float32)
