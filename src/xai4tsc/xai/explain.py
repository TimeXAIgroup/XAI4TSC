"""``generate_explanation()`` entry point, ``EXPLAINERS`` registry, plot helpers."""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .._random import DEFAULT_SEED, resolve_rng
from ..data.base import class_names
from ..models.base import ModelBase, predict_probs
from ..utils.plot import plot_relevance, plot_relevance_f, plot_relevance_tf
from ..utils.utils import dict_to_args
from . import (
    explanation_domains,
    feature_attribution,
    freqrise,
    random_baseline,
    wrappers,
)
from ._types import Domain, Explanation, is_multi_target, resolve_data_domain
from .base import ExplainerBase

logger = logging.getLogger(__name__)


def generate_explanation(
    method: str,
    model: ModelBase,
    data: np.ndarray,
    labels: np.ndarray | None = None,
    targets: str = "predicted",
    params: dict | None = None,
    call_params: dict | None = None,
    encoder: object | None = None,
    indices: list | None = None,
    n_samples: int = 5,
    prng: np.random.Generator | None = None,
    device: str = "cpu",
    save_path: Path | str | None = None,
    visualization_type: Sequence[str] = ("points",),
    data_domain: str | Domain = Domain.TIME,
) -> Explanation | None:
    """
    Generate an explanation for the given indices or for random samples.

    Set *indices* to an empty list or ``None`` to choose random samples.

    Parameters
    ----------
    method : str
        Explanation method to use. Have a look at the support.yaml file.
    model : ModelBase
        Model to explain.
    data : np.ndarray
        Dataset to generate an explanation for.
    labels : np.ndarray, optional
        Labels of the dataset, by default None
    targets : str, optional
        Which class to explain: ``"predicted"`` (default), ``"label"``
        (ground truth), or ``"all"`` (one explanation per class).
    params : dict, optional
        Parameters for the explanation method, by default None
    call_params : dict, optional
        Per-call options given into the explainer's ``explain()`` as keyword
        arguments, by default None. Construction options go in *params*; use this
        for options an explainer takes per call. Recorded on the returned
        explanation so the re-explaining metrics replay the same call.
    encoder : object, optional
        Fitted label encoder, used to name the explained classes. ``None`` by
        default.
    indices : list, optional
        Indices of samples to explain, by default None
    n_samples : int, optional
        Number of random samples to explain if no indices supplied, by default 5
    prng : np.random.Generator, optional
        Generator used to pick random sample indices. When ``None`` a generator
        seeded with :data:`~xai4tsc.DEFAULT_SEED` is used, so the selection is
        reproducible; the runner passes a per-dataset generator so every
        explainer scores the same samples.
    device : str, optional
        Device to calculate on, by default ``"cpu"``.
    save_path : Path | str, optional
        Path to save the explanation visualisation to, by default None
    visualization_type : Sequence[str], optional
        Plot styles to render when saving, by default ``("points",)``.
    data_domain : str or Domain, optional
        Signal domain *data* is expressed in, i.e. the domain the model consumes
        (``"time"`` by default; see :attr:`xai4tsc.data.DatasetBase.data_domain`).
        A plain attributor explains in this domain; a domain-changing explainer is
        refused on data it cannot transform from.

    Returns
    -------
    Explanation
        An object of the Explanation dataclass.
    """
    prng = resolve_rng(DEFAULT_SEED, prng)

    model.eval()
    # Samples to explain are in list form
    if indices is None and n_samples > 0:
        indices = np.arange(len(data))
        prng.shuffle(indices)
        indices = indices[:n_samples]
    elif len(indices) == 0:
        if n_samples <= 0:
            logger.warning(
                "Supplied no indices to explain nor a sample count for "
                "random explanations, aborting."
            )
            return None
        indices = np.arange(len(data))
        prng.shuffle(indices)
        indices = indices[:n_samples]
    else:
        # Explicitly requested indices are the one path numpy would fail on, and
        # it fails with a bare IndexError naming an axis rather than the setting
        # that produced it.  Both data[indices] below and the metadata lookup in
        # _get_explanation index by these, so one check covers both.
        out_of_range = [int(i) for i in indices if not 0 <= int(i) < len(data)]
        if out_of_range:
            raise ValueError(
                f"indices {out_of_range} are out of range for {len(data)} "
                f"samples (valid: 0-{len(data) - 1})."
            )

    explanation = _get_explanation(
        method=method,
        model=model,
        data=data[indices],
        labels=labels[indices],
        encoder=encoder,
        params=params,
        call_params=call_params,
        targets=targets,
        device=device,
        orig_indices=indices,
        save_path=save_path,
        visualization_type=visualization_type,
        data_domain=data_domain,
    )
    return explanation


def _get_explanation(
    method: str,
    model: ModelBase,
    data: np.ndarray,
    labels: np.ndarray,
    encoder: object,
    params: dict,
    call_params: dict | None = None,
    targets: str = "predicted",
    device: str = "cpu",
    orig_indices: np.ndarray | None = None,
    save_path: Path | None = None,
    visualization_type: Sequence[str] = ("points",),
    data_domain: str | Domain = Domain.TIME,
) -> Explanation:
    """
    Use the supplied parameters to explain the given samples.

    Parameters
    ----------
    method : str
        Explanation method to use, must be contained in EXPLAINERS.
    model : ModelBase
        Model to use.
    data : np.ndarray
        Samples to use.
    labels : np.ndarray
        Target labels.
    encoder : object
        Fitted label encoder for the labels
    params : dict
        Parameters for the explanation method
    call_params : dict, optional
        Per-call options given into the explainer's ``explain()`` as keyword
        arguments, by default None. Construction options go in *params*; use this
        for options an explainer takes per call. Recorded on the returned
        explanation so the re-explaining metrics replay the same call.
    targets : str, optional
        Whether to explain predictions for all classes or only the
        predicted class, by default "predicted"
    device : str, optional
        Device to calculate on, by default "cpu"
    orig_indices : np.ndarray, optional
        Original indices of the supplied samples, by default None
    save_path : Path, optional
        Path to save visualisation to, by default None
    visualization_type : Sequence[str], optional
        Plot styles to render when saving, by default ``("points",)``.
    data_domain : str or Domain, optional
        Signal domain *data* is expressed in, i.e. the domain the model consumes
        (``"time"`` by default; see :attr:`xai4tsc.data.DatasetBase.data_domain`).
        A plain attributor explains in this domain, so it is what the resulting
        ``explanation_domain`` becomes; a domain-changing explainer declares which
        domains it can consume and is refused on the others.

    Returns
    -------
    Explanation
        A dataclass containing the explanation and other relevant information.

    Raises
    ------
    NotImplementedError
        Thrown for not implemented explanation methods.
    ValueError
        If the explainer cannot consume data in *data_domain*.
    """
    # Instantiate the explainer, filtering config params to its __init__.
    explainer_key = method.lower()
    explainer_instance = build_explainer(explainer_key, params)

    data_domain = resolve_data_domain(data_domain)

    # A domain-changing explainer transforms its input, so it can only start from a
    # domain it knows how to transform; anything else would transform data that is
    # already in the target domain. An empty set means "any domain".
    accepted = explainer_instance.accepts_data_domains
    if accepted and data_domain not in accepted:
        raise ValueError(
            f"Explainer '{explainer_key}' consumes "
            f"{sorted(d.value for d in accepted)} data, but the data is "
            f"{data_domain.value}."
        )

    # Derive the realized explanation domain. A plain attributor assigns relevance
    # to the features it was given, so its explanation is in the data's own domain.
    # For a domain-changing explainer it is the sole declared member; multi-capability
    # ones (e.g. FreqRISE) must set exp.explanation_domain themselves inside
    # explain() and are validated below.
    domains = explainer_instance.explanation_domains
    if explainer_instance.explains_in_data_domain:
        realized_domain = data_domain
    else:
        realized_domain = next(iter(domains)) if len(domains) == 1 else Domain.TIME

    # Create explanation data structure
    exp = Explanation(
        explainer=explainer_key,
        explainer_params=dict(params or {}),
        explainer_call_params=dict(call_params or {}),
        explanation_type=explainer_instance.explanation_type,
        exp_values=None,
        data=data,
        labels=labels,
        encoder=encoder,
        indices=orig_indices,
        meta=None,
        data_domain=data_domain,
        explanation_domain=realized_domain,
        transform=getattr(explainer_instance, "transform", None),
    )

    # Check which targets to generate explanations for
    if targets == "all":
        # Explain all classes, stacking one attribution per class on a leading
        # axis. The container is derived from what the explainer actually
        # returns, not from `data`: frequency is (n, C, F), time-frequency is
        # (n, C, n_freq, n_time), and freq/TF outputs may be complex.
        classes = encoder.classes_
        exp_values = None
        for ind in range(len(classes)):
            relevance = explainer_instance.explain(
                model, exp, device, [ind] * len(data), **exp.explainer_call_params
            )
            if exp_values is None:
                exp_values = np.empty(
                    (len(classes), *relevance.shape), dtype=relevance.dtype
                )
            exp_values[ind] = relevance
        exp.exp_values = exp_values
    elif targets == "label":
        # Captum targets must be per-sample integer class indices. Encoded labels
        # may be 2-D (one-hot or ordinal); collapse them back to class indices.
        label_arr = np.asarray(labels)
        if label_arr.ndim > 1:
            label_arr = (
                label_arr.argmax(axis=1)
                if label_arr.shape[1] > 1
                else label_arr.ravel()
            )
        method_targets = label_arr.astype(int).tolist()
        exp.exp_values = explainer_instance.explain(
            model, exp, device, method_targets, **exp.explainer_call_params
        )
        exp.explained_targets = np.asarray(method_targets, dtype=int)
    else:
        # Explain the model's predicted class for each sample.
        method_targets = predict_probs(model, exp.data).argmax(axis=1).tolist()
        exp.exp_values = explainer_instance.explain(
            model, exp, device, method_targets, **exp.explainer_call_params
        )
        exp.explained_targets = np.asarray(method_targets, dtype=int)

    # Save results as specified
    if save_path is not None:
        plot_exp(exp, save_path=save_path, visualization_type=visualization_type)
    return exp


def _class_label(names: list[str], index: int) -> str:
    """
    Return the display name of class *index*, or the index itself when unnamed.

    An encoder exposing no classes gives an empty name list, in which case the
    index is the only identifier available.
    """
    return names[index] if 0 <= index < len(names) else str(index)


def plot_exp(
    exp: Explanation,
    norm: bool = True,
    save_path: Path | str | None = None,
    visualization_type: Sequence[str] = ("points",),
) -> None:
    """
    Render and save relevance plots for every sample in *exp*.

    Plots are written flat under *save_path*, one file per sample (and, for
    ``targets="all"`` explanations, per class), named
    ``relevance_index_<idx>_class_<name>...`` so samples sharing a directory do
    not overwrite one another. ``<name>`` is the class the attribution explains
    (``exp.explained_targets``, falling back to ``exp.labels``), spelled as
    :func:`~xai4tsc.data.base.class_names` spells it - so a multi-hot class reads
    ``0-0-1`` rather than ``[0 0 1]``, which is not a usable filename.

    The renderer is chosen by ``exp.explanation_domain``:
    time-domain explanations use the 1-D
    :func:`~xai4tsc.utils.plot.plot_relevance` overlay styles; frequency and
    time-frequency explanations use :func:`~xai4tsc.utils.plot.plot_relevance_f`
    and :func:`~xai4tsc.utils.plot.plot_relevance_tf` respectively. Frequency and
    time-frequency explanations plot against the transformed signal, derived on
    demand via ``exp.transform``.

    Parameters
    ----------
    exp : Explanation
        The explanation to render. Multi-target ("all") explanations are detected
        via :func:`~xai4tsc.xai.is_multi_target` (domain-aware), so a single-target
        time-frequency explanation is not mistaken for a multi-target one.
    norm : bool, optional
        Whether to normalize explanation values, by default True.
    save_path : Path | str | None, optional
        Directory to save the plots to, by default None.
    visualization_type : Sequence[str], optional
        Overlay styles for time / frequency plots: any of ``"points"`` (default),
        ``"background"``, ``"intensity"``, ``"graph"``, or ``"bar"``. Ignored for
        time-frequency, which renders a single fixed heatmap view.

    Raises
    ------
    ValueError
        If *exp* is a frequency / time-frequency explanation but carries no
        ``exp.transform`` to derive the transformed signal from.
    """
    save_path = Path(save_path) if isinstance(save_path, str) else save_path
    save_path.mkdir(parents=True, exist_ok=True)

    is_transformed = exp.explanation_domain in (Domain.FREQUENCY, Domain.TIME_FREQUENCY)
    if is_transformed and exp.transform is None:
        raise ValueError(
            f"Cannot plot a {exp.explanation_domain.value} explanation without "
            "exp.transform (the domain transform); none is set."
        )

    # Frequency / time-frequency explanations plot against the transformed signal,
    # derived on demand; time-domain explanations plot against the raw series.
    if is_transformed:
        signal = (
            exp.transform.forward(torch.as_tensor(exp.data, dtype=torch.float32))
            .cpu()
            .numpy()
        )
    else:
        signal = exp.data

    multi_target = is_multi_target(exp)
    names = class_names(exp.encoder)
    if multi_target:
        explained = None
    else:
        explained = (
            exp.explained_targets if exp.explained_targets is not None else exp.labels
        )

    for sample_ind in range(exp.data.shape[0]):
        signal_i = signal[sample_ind : sample_ind + 1]
        idx = exp.indices[sample_ind]
        # Normalise both target modes to a list of (label, relevance) pairs so the
        # per-domain dispatch below is written once. Multi-target stacks a class
        # axis first: exp_values[class, sample]; single-target is exp_values[sample].
        if multi_target:
            items = [
                (
                    _class_label(names, class_ind),
                    exp.exp_values[class_ind, sample_ind : sample_ind + 1],
                )
                for class_ind in range(exp.exp_values.shape[0])
            ]
        else:
            items = [
                (
                    _class_label(names, int(explained[sample_ind])),
                    exp.exp_values[sample_ind : sample_ind + 1],
                )
            ]

        for label, relevance_i in items:
            base = save_path / f"relevance_index_{idx}_class_{label}"
            if exp.explanation_domain == Domain.TIME_FREQUENCY:
                # Time-frequency renders a single fixed heatmap view (no rel_type).
                fig_path = Path(f"{base}_timefrequency.png")
                plot_relevance_tf(signal_i, relevance_i, save_path=fig_path)
            else:
                plotter = (
                    plot_relevance
                    if exp.explanation_domain == Domain.TIME
                    else plot_relevance_f
                )
                infix = "" if exp.explanation_domain == Domain.TIME else "_frequency"
                for vtype in visualization_type:
                    fig_path = Path(f"{base}{infix}_{vtype}.png")
                    plotter(signal_i, relevance_i, rel_type=vtype, save_path=fig_path)

            logger.info(
                "Saved %s explanation for sample with index %s to:\n%s",
                exp.explanation_domain.value,
                idx,
                fig_path,
            )


EXPLAINERS = {
    "integrated_gradients": feature_attribution.IntegratedGradientsExplainer,
    "guided_backpropagation": feature_attribution.GuidedBackpropagationExplainer,
    "deconvolution": feature_attribution.DeconvolutionExplainer,
    "deeplift": feature_attribution.DeepLiftExplainer,
    "occlusion": feature_attribution.OcclusionExplainer,
    "tshap": feature_attribution.TSHAPExplainer,
    "sign": wrappers.SignExplainer,
    "freqrise": freqrise.FreqRISEExplainer,
    "frequency": explanation_domains.FrequencyExplainer,
    "timefrequency": explanation_domains.TimeFrequencyExplainer,
    "random_frequency": random_baseline.RandomFrequencyExplainer,
    "random_timefrequency": random_baseline.RandomTimeFreqExplainer,
}


def build_explainer(method: str, params: dict | None) -> ExplainerBase:
    """
    Instantiate a registered explainer, filtering *params* to its ``__init__``.

    Shared by :func:`_get_explanation` and by wrapper explainers
    (:class:`~xai4tsc.xai.base.WrapperExplainer`) that need to build their
    wrapped base method the same way the runner does.

    Parameters
    ----------
    method : str
        Explainer name; looked up case-insensitively in :data:`EXPLAINERS`.
    params : dict or None
        Parameters to pass to the explainer's ``__init__``. Keys not matching
        the constructor signature are dropped via :func:`dict_to_args`. If the
        class defines no ``__init__``, no parameters are passed.

    Returns
    -------
    ExplainerBase
        A ready-to-use explainer instance.

    Raises
    ------
    NotImplementedError
        If *method* is not registered in :data:`EXPLAINERS`.
    """
    explainer_key = method.lower()
    if explainer_key not in EXPLAINERS:
        raise NotImplementedError(f"Explainer '{explainer_key}' not supported.")
    # Only pass config params if the class defines its own __init__ — otherwise
    # object.__init__ is used, which would receive the full config dict and fail.
    explainer_class = EXPLAINERS[explainer_key]
    if "__init__" in explainer_class.__dict__:
        init_params = dict_to_args(params, explainer_class.__init__)
        return explainer_class(**(init_params or {}))
    return explainer_class()


def register_explainer(name: str, explainer_class: type) -> None:
    """
    Register a custom :class:`~xai4tsc.xai.ExplainerBase` subclass.

    After registration the explainer is available by *name* in experiment
    configs and via :func:`_get_explanation`.

    Parameters
    ----------
    name:
        Key used to look up the explainer (case-insensitive).
    explainer_class:
        A concrete subclass of :class:`~xai4tsc.xai.ExplainerBase` (or one of
        its mid-level subclasses).

    Raises
    ------
    TypeError
        If *explainer_class* is not an
        :class:`~xai4tsc.xai.ExplainerBase` subclass.
    """
    is_subclass = isinstance(explainer_class, type) and issubclass(
        explainer_class, ExplainerBase
    )
    if not is_subclass:
        raise TypeError(
            f"Explainer '{name}' must be an ExplainerBase subclass, "
            f"got {explainer_class!r}."
        )
    EXPLAINERS[name.lower()] = explainer_class
