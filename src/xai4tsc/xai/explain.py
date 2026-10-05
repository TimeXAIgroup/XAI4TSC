"""``generate_explanation()`` entry point, ``EXPLAINERS`` registry, plot helpers."""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from ..data.base import class_names, decode_labels
from ..models.base import ModelBase
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
    encoder: OneHotEncoder | LabelEncoder | OrdinalEncoder | None = None,
    indices: list | None = None,
    samples: int = 5,
    prng: np.random.Generator | None = None,
    device: str = "cpu",
    save_path: Path | str | None = None,
    visualization_type: list | None = None,
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
    encoder : OneHotEncoder | LabelEncoder | OrdinalEncoder, optional
        Encoder used to encode the labels, by default None
    indices : list, optional
        Indices of samples to explain, by default None
    samples : int, optional
        Number of random samples to explain if no indices supplied, by default 5
    prng : np.random.Generator, optional
        Random number generator to sample random indices, by default None
    device : str, optional
        Device to calculate on, by default ``"cpu"``.
    save_path : Path | str, optional
        Path to save the explanation visualisation to, by default None
    visualization_type : list, optional
        Plot styles to render when saving, by default ``["points"]``.
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
    if visualization_type is None:
        visualization_type = ["bubbles"]
    if prng is None:
        prng = np.random.default_rng()

    model.eval()
    # Samples to explain are in list form
    if indices is None and samples > 0:
        indices = np.arange(len(data))
        prng.shuffle(indices)
        indices = indices[:samples]
    elif len(indices) == 0:
        if samples <= 0:
            logger.warning(
                "Supplied no indices to explain nor a sample count for "
                "random explanations, aborting."
            )
            return None
        indices = np.arange(len(data))
        prng.shuffle(indices)
        indices = indices[:samples]

    explanation = _get_explanation(
        method=method,
        model=model,
        data=data[indices],
        labels=labels[indices],
        encoder=encoder,
        params=params,
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
    encoder: OneHotEncoder | LabelEncoder | OrdinalEncoder,
    params: dict,
    targets: str = "predicted",
    device: str = "cpu",
    orig_indices: np.ndarray | None = None,
    save_path: Path | None = None,
    visualization_type: list | None = None,
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
    encoder : OneHotEncoder | LabelEncoder | OrdinalEncoder
        Encoder for the labels
    params : dict
        Parameters for the explanation method
    targets : str, optional
        Whether to explain predictions for all classes or only the
        predicted class, by default "predicted"
    device : str, optional
        Device to calculate on, by default "cpu"
    orig_indices : np.ndarray, optional
        Original indices of the supplied samples, by default None
    save_path : Path, optional
        Path to save visualisation to, by default None
    visualization_type : list, optional
        Plot styles to render when saving, by default ``["points"]``.
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
    if visualization_type is None:
        visualization_type = ["points"]
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
                model, exp, device, [ind] * len(data)
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
        exp.exp_values = explainer_instance.explain(model, exp, device, method_targets)
    else:
        # Explain the model's predicted class for each sample.
        method_targets = model.predict(exp.data)[0].tolist()
        exp.exp_values = explainer_instance.explain(model, exp, device, method_targets)

    # Save results as specified
    if save_path is not None:
        plot_exp(exp, save_path=save_path, visualization_type=visualization_type)
    return exp


def plot_exp(
    exp: Explanation,
    norm: bool = True,
    save_path: Path | str | None = None,
    visualization_type: list | None = None,
) -> None:
    """
    Render and save relevance plots for every sample in *exp*.

    Plots are written flat under *save_path*, one file per sample (and, for
    ``targets="all"`` explanations, per class), named
    ``relevance_index_<idx>_class_<label>...`` so samples sharing a directory do
    not overwrite one another. The renderer is chosen by ``exp.explanation_domain``:
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
    visualization_type : list | None, optional
        Overlay styles for time / frequency plots: any of ``"points"`` (default),
        ``"background"``, ``"intensity"``, ``"graph"``, or ``"bar"``. Ignored for
        time-frequency, which renders a single fixed heatmap view.

    Raises
    ------
    ValueError
        If *exp* is a frequency / time-frequency explanation but carries no
        ``exp.transform`` to derive the transformed signal from.
    """
    if visualization_type is None:
        visualization_type = ["points"]
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
    class_labels = None if multi_target else decode_labels(exp.encoder, exp.labels)

    for sample_ind in range(exp.data.shape[0]):
        signal_i = signal[sample_ind : sample_ind + 1]
        idx = exp.indices[sample_ind]
        # Normalise both target modes to a list of (label, relevance) pairs so the
        # per-domain dispatch below is written once. Multi-target stacks a class
        # axis first: exp_values[class, sample]; single-target is exp_values[sample].
        if multi_target:
            items = [
                (label, exp.exp_values[class_ind, sample_ind : sample_ind + 1])
                for class_ind, label in enumerate(class_names(exp.encoder))
            ]
        else:
            items = [
                (class_labels[sample_ind], exp.exp_values[sample_ind : sample_ind + 1])
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
