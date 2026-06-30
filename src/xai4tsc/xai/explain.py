"""``generate_explanation()`` entry point, ``EXPLAINERS`` registry, plot helpers."""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

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
from ._types import Domain, Explanation
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
    save_path: Path | None = None,
    visualization_type: list | None = None,
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
    save_path : Path, optional
        Path to save the explanation visualisation to, by default None
    visualization_type : list, optional
        Plot styles to render when saving, by default ``["bubbles"]``.

    Returns
    -------
    Explanation
        An object of the Explanation dataclass.
    """
    if visualization_type is None:
        visualization_type = ["bubbles"]
    if prng is None:
        prng = np.random.default_rng()
    # Attribution must run with Dropout/BatchNorm in inference mode, otherwise
    # the explained forward pass is stochastic and uses batch statistics.
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
        Plot styles to render when saving, by default ``["bubbles"]``.

    Returns
    -------
    Explanation
        A dataclass containing the explanation and other relevant information.

    Raises
    ------
    NotImplementedError
        Thrown for not implemented explanation methods.
    """
    if visualization_type is None:
        visualization_type = ["bubbles"]
    # Instantiate the explainer, filtering config params to its __init__.
    explainer_key = method.lower()
    explainer_instance = build_explainer(explainer_key, params)

    # Derive the realized explanation domain. For single-capability explainers
    # it is the sole member; multi-capability explainers (e.g. FreqRISE) must set
    # exp.explanation_domain themselves inside explain() and are validated below.
    domains = explainer_instance.explanation_domains
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
        explanation_domain=realized_domain,
        transform=getattr(explainer_instance, "transform", None),
    )

    # Check which targets to generate explanations for
    if targets == "all":
        # Explain all classes
        classes = encoder.classes_
        # Stack array to hold explanations for each specified class
        exp.exp_values = np.vstack(
            [np.expand_dims(np.zeros_like(data), axis=0)] * len(classes)
        )
        for ind in range(len(classes)):
            exp.exp_values[ind, ...] = explainer_instance.explain(
                model, exp, device, [ind] * len(data)
            )
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

    # Invariant: the realized domain must be one the explainer declares it can
    # produce. Catches a multi-capability explainer that forgot to set it.
    if exp.explanation_domain not in explainer_instance.explanation_domains:
        raise ValueError(
            f"Explainer {explainer_key!r} produced explanation_domain "
            f"{exp.explanation_domain} not in its declared explanation_domains "
            f"{explainer_instance.explanation_domains}."
        )

    # Save results as specified
    if save_path is not None:
        plot_exp(exp, save_path=save_path, visualization_type=visualization_type)
    return exp


def plot_exp(
    exp: Explanation,
    norm: bool = True,
    save_path: Path | None = None,
    visualization_type: list | None = None,
) -> None:
    """
    Render and save relevance plots for every sample in *exp*.

    One sub-directory per sample index is created under *save_path*. The renderer
    is chosen by ``exp.explanation_domain``: time-domain explanations use the 1-D
    :func:`~xai4tsc.utils.plot.plot_relevance` overlay styles; frequency and
    time-frequency explanations use :func:`~xai4tsc.utils.plot.plot_relevance_f`
    and :func:`~xai4tsc.utils.plot.plot_relevance_tf` respectively. For
    multi-target (time-domain) explanations one plot per class is written.
    """
    if visualization_type is None:
        visualization_type = ["bubbles"]

    if exp.explanation_domain in (Domain.FREQUENCY, Domain.TIME_FREQUENCY):
        _plot_transformed_exp(exp, save_path, visualization_type)
        return

    n_samples = exp.data.shape[0]
    multi_target = len(exp.exp_values.shape) == 4  # (n_classes, n_samples, C, T)

    if multi_target:
        # "all" targets: exp.exp_values shape = (n_classes, n_samples, C, T)
        for class_ind, class_label in enumerate(exp.encoder.classes_):
            for sample_ind in range(n_samples):
                signal_i = exp.data[sample_ind : sample_ind + 1]
                relevance_i = exp.exp_values[class_ind, sample_ind : sample_ind + 1]
                sample_dir = save_path / f"index_{exp.indices[sample_ind]}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                for vtype in visualization_type:
                    fig_path = sample_dir / f"relevance_class_{class_label}_{vtype}.png"
                    plot_relevance(
                        signal_i, relevance_i, rel_type=vtype, save_path=fig_path
                    )
                    logger.info(
                        "Saved explanation for sample with index %s to:\n%s",
                        exp.indices[sample_ind],
                        fig_path,
                    )
    else:
        # Single target: exp.exp_values shape = (n_samples, C, T)
        class_labels = exp.encoder.inverse_transform(exp.labels)
        for sample_ind in range(n_samples):
            signal_i = exp.data[sample_ind : sample_ind + 1]
            relevance_i = exp.exp_values[sample_ind : sample_ind + 1]
            sample_dir = save_path / f"index_{exp.indices[sample_ind]}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for vtype in visualization_type:
                fig_path = (
                    sample_dir
                    / f"relevance_class_{class_labels[sample_ind]}_{vtype}.png"
                )
                plot_relevance(
                    signal_i, relevance_i, rel_type=vtype, save_path=fig_path
                )
                logger.info(
                    "Saved explanation for sample with index %s to:\n%s",
                    exp.indices[sample_ind],
                    fig_path,
                )


def _plot_transformed_exp(
    exp: Explanation,
    save_path: Path | None,
    visualization_type: list,
) -> None:
    """
    Render frequency / time-frequency explanations (single-target).

    The transformed input is *derived* on demand via ``exp.transform`` rather
    than stored on the Explanation, then handed to
    :func:`~xai4tsc.utils.plot.plot_relevance_f` (frequency) or
    :func:`~xai4tsc.utils.plot.plot_relevance_tf` (time-frequency).
    """
    if exp.transform is None:
        raise ValueError(
            f"Cannot plot a {exp.explanation_domain.value} explanation without "
            "exp.transform (the domain transform); none is set."
        )
    signal = (
        exp.transform.forward(torch.as_tensor(exp.data, dtype=torch.float32))
        .cpu()
        .numpy()
    )
    class_labels = (
        exp.encoder.inverse_transform(exp.labels)
        if exp.encoder is not None
        else exp.labels
    )
    is_tf = exp.explanation_domain == Domain.TIME_FREQUENCY

    for sample_ind in range(exp.data.shape[0]):
        signal_i = signal[sample_ind : sample_ind + 1]
        relevance_i = exp.exp_values[sample_ind : sample_ind + 1]
        sample_dir = save_path / f"index_{exp.indices[sample_ind]}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        label = class_labels[sample_ind]
        if is_tf:
            fig_path = sample_dir / f"relevance_class_{label}_timefrequency.png"
            plot_relevance_tf(signal_i, relevance_i, save_path=fig_path)
        else:
            for vtype in visualization_type:
                fig_path = sample_dir / f"relevance_class_{label}_frequency_{vtype}.png"
                plot_relevance_f(
                    signal_i, relevance_i, rel_type=vtype, save_path=fig_path
                )
        logger.info(
            "Saved %s explanation for sample with index %s to:\n%s",
            exp.explanation_domain.value,
            exp.indices[sample_ind],
            sample_dir,
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
