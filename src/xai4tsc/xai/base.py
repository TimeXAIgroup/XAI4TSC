"""Explainer ABCs: ``ExplainerBase``, ``GradientExplainer``, and friends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from ._types import DataType, Domain

if TYPE_CHECKING:
    from captum.attr import Attribution

    from ._types import Explanation


class ExplainerBase(ABC):
    """
    Base class for all XAI explainer methods.

    Subclass one of the mid-level classes (:class:`GradientExplainer`,
    :class:`PerturbationExplainer`, or :class:`SurrogateExplainer`) to add a
    custom explainer.  Register the subclass with
    :func:`xai4tsc.register_explainer` to make it available by name.

    Example::

        class MyExplainer(GradientExplainer):
            explanation_type = "feature_attribution"

            def _get_captum_attribution(self, model):
                from captum.attr import IntegratedGradients
                return IntegratedGradients(model)

        xai4tsc.register_explainer("my_method", MyExplainer)

    Use specific classes for explanation types, e.g. ``"feature_attribution"``
    or ``"counterfactual"`` explanations.
    """

    explanation_type: str
    """Type of explanation produced by this explainer."""

    data_applicability: ClassVar[set[DataType]] = {DataType.AGNOSTIC}
    """Data domains this explainer applies to. A set of :class:`~xai4tsc.xai.DataType`
    members — ``{DataType.AGNOSTIC}`` (any input) or ``{DataType.TIME_SERIES}``."""

    explanation_domains: ClassVar[set[Domain]] = {Domain.TIME}
    """Signal domains this explainer *can* produce explanations in. A set of
    :class:`~xai4tsc.xai.Domain` members (capability declaration, mirroring
    :attr:`data_applicability`). Checked statically by the runner's config sanity
    check before any explainer is instantiated. Defaults to ``{Domain.TIME}``;
    frequency/time-frequency explainers override it. The realized domain of a
    produced explanation (``Explanation.explanation_domain``) must be a member of
    this set."""

    @abstractmethod
    def explain(
        self,
        model: torch.nn.Module,
        exp: Explanation,
        device: str | torch.device,
        targets: list | None,
        **kwargs: object,
    ) -> np.ndarray:
        """
        Generate explanations for the samples in *exp*.

        Parameters
        ----------
        model :
            The PyTorch model to explain (an :class:`~torch.nn.Module`).
        exp :
            :class:`~xai4tsc.xai.Explanation` dataclass holding the data,
            labels, and encoder for the samples to explain.
        device :
            Compute device (e.g. ``"cpu"`` or ``"cuda"``).
        targets :
            Target class indices to explain, or ``None`` for all classes.
        **kwargs :
            Additional explainer-specific options.

        Returns
        -------
        np.ndarray
            Explanation values with the same leading dimensions as ``exp.data``.
        """


class GradientExplainer(ExplainerBase, ABC):
    """
    Base class for gradient-based attribution methods (Captum).

    Provides a shared :meth:`explain` implementation that calls the abstract
    :meth:`_get_captum_attribution` factory method.  Subclasses only need to
    return the appropriate Captum ``Attribution`` object.

    Suitable for: Integrated Gradients, DeepLIFT, Deconvolution,
    Guided Backpropagation, and any other Captum gradient method.

    Subclasses may set ``_attribute_kwargs`` in their ``__init__`` to pass
    extra keyword arguments to Captum's ``attribute()`` call.
    """

    explanation_type = "feature_attribution"
    _attribute_kwargs: ClassVar[dict] = {}

    @abstractmethod
    def _get_captum_attribution(self, model: torch.nn.Module) -> Attribution:
        """
        Return a Captum ``Attribution`` instance wrapping *model*.

        Parameters
        ----------
        model : nn.Module
            The model to explain.

        Returns
        -------
        captum.attr.Attribution
            A Captum attribution object ready to call ``.attribute()``.
        """

    def explain(
        self,
        model: torch.nn.Module,
        exp: Explanation,
        device: str | torch.device,
        targets: list | None,
        **kwargs: object,
    ) -> np.ndarray:
        """Compute Captum attributions for the samples in *exp*."""
        attribution = self._get_captum_attribution(model)
        data_tensor = torch.tensor(
            exp.data, dtype=torch.float32, device=device
        ).requires_grad_(True)
        return (
            attribution.attribute(data_tensor, target=targets, **self._attribute_kwargs)
            .detach()
            .cpu()
            .numpy()
        )


class PerturbationExplainer(ExplainerBase, ABC):
    """
    Base class for perturbation-based attribution methods.

    Suitable for: Occlusion, RISE, and any method that masks or replaces
    parts of the input to estimate feature importance.
    """

    explanation_type = "feature_attribution"


class SurrogateExplainer(ExplainerBase, ABC):
    """
    Base class for surrogate / proxy model attribution methods.

    Suitable for: LIME, SHAP (KernelSHAP), and any method that fits a
    simpler interpretable model locally around the prediction.

    Currently a placeholder — no concrete implementations yet.
    """

    explanation_type = "feature_attribution"


class WrapperExplainer(ExplainerBase, ABC):
    """
    Base class for explainers that wrap and extend another explainer.

    A wrapper takes a *base* explainer (any entry in
    :data:`~xai4tsc.xai.explain.EXPLAINERS`) and modifies its computation:
    either at the final input-multiplication step (e.g. SIGN for gradient-family
    bases) or by re-invoking it on transformed inputs (e.g. Temporal Saliency
    Rescaling, Time Forward Tunnel). Use :meth:`_run_base` to (re-)invoke the
    wrapped method. Subclasses needing a true in-pass modification (e.g. an LRP
    input-layer rule) may override :meth:`explain` without using
    :meth:`_run_base` at all.

    The base is instantiated via
    :func:`~xai4tsc.xai.explain.build_explainer`, so it is configured exactly
    as it would be when invoked directly by name.

    Parameters
    ----------
    base : dict or str
        The base explainer to wrap. Either a bare method name, or a config
        dict with a ``"method"`` key plus that method's hyperparameters.
    """

    def __init__(self, base: dict | str) -> None:
        # Lazy import avoids a circular import: explain.py imports the wrappers
        # module to populate EXPLAINERS, so wrappers/base cannot import
        # explain.py at module load (same pattern config.py uses).
        from .explain import build_explainer

        base_cfg = {"method": base} if isinstance(base, str) else dict(base)
        method = base_cfg.pop("method")
        self._base = build_explainer(method, base_cfg)
        # A wrapper reports the same explanation type, data applicability, and
        # explanation domains as the method it extends. Domain-changing wrappers
        # (e.g. Frequency/TimeFrequency) override explanation_domains *after*
        # calling super().__init__().
        self.explanation_type = self._base.explanation_type
        self.data_applicability = self._base.data_applicability
        self.explanation_domains = self._base.explanation_domains

    def _run_base(
        self,
        model: torch.nn.Module,
        exp: Explanation,
        device: str | torch.device,
        targets: list | None,
    ) -> np.ndarray:
        """
        Run the wrapped base explainer and return its relevance array.

        Parameters
        ----------
        model : nn.Module
            The model to explain.
        exp : Explanation
            Samples to explain (the wrapper may pass a transformed copy).
        device : str | torch.device
            Compute device.
        targets : list | None
            Target class indices, or ``None`` for all classes.

        Returns
        -------
        np.ndarray
            The base method's explanation values.
        """
        return self._base.explain(model, exp, device, targets)
