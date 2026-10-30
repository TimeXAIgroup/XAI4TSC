"""
Every config-string-to-object table of the runner.

The package takes objects - a :class:`torch.nn.Module` loss, a
:class:`torch.optim.Optimizer` subclass, an sklearn-style target encoder, a
:class:`~xai4tsc.utils.DomainTransform`, an explainer, an evaluator - so the only
gap is that YAML cannot hold a Python object. This module closes it in one place.

Two kinds of table live here:

- Component tables (:data:`MODELS`, :data:`EXPLAINERS`, :data:`METRICS`,
  :data:`SYNTHETIC_DATASETS`), copied from the package catalogs and extended with
  the ``class_name`` / ``class_path`` components of a config by
  :func:`load_external_components`, which is gated by
  ``general.allow_external_code``.
- :class:`Lookup` families (encoder, loss, optimizer, transform): a table plus a
  dotted-path escape hatch, so a loss that has no table entry still works. A
  dotted path must resolve to a class of the family's kind before anything is
  constructed, so a config cannot name an arbitrary callable such as
  ``os.system``.

Each family then constructs in the way that suits it: an encoder and a transform
are built with their parameters, a loss with its list parameters as tensors, and
an optimizer is deferred into a :func:`functools.partial` because it needs
``model.parameters()``, which only exists inside ``train_model``.
"""

import inspect
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import quantus
import torch
from quantus.functions import perturb_func as _quantus_perturb
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim

import xai4tsc
from xai4tsc.data import MultiHotLabelEncoder, PowersetLabelEncoder
from xai4tsc.evaluation import EvaluatorBase
from xai4tsc.models.base import ModelBase
from xai4tsc.utils.fourier_transforms import (
    DomainTransform,
    FourierTransform,
    RFFTransform,
    STFTransform,
)
from xai4tsc.utils.perturbation import BASELINES
from xai4tsc.xai.base import ExplainerBase

logger = logging.getLogger("xai4tsc.runner.lookups")


# ── component tables ──────────────────────────────────────────────────────────

MODELS: dict[str, type] = dict(xai4tsc.MODELS)
"""Config ``model`` (lower case) -> model class."""

EXPLAINERS: dict[str, type] = dict(xai4tsc.EXPLAINERS)
"""Config ``method`` (lower case) -> explainer class."""

METRICS: dict[str, Callable] = dict(xai4tsc.METRICS)
"""Config ``metric`` (exact display name) -> callable building an evaluator."""

QUANTUS_METRICS: dict[str, type[quantus.Metric]] = dict(xai4tsc.QUANTUS_METRICS)
"""Config ``metric`` -> Quantus class, for the metrics :data:`METRICS` adapts."""

SYNTHETIC_DATASETS: dict[str, type] = dict(xai4tsc.SYNTHETIC_DATASETS)
"""Config ``dataset`` -> synthetic dataset class."""


def load_external_components(config: dict) -> None:
    """
    Add the config's ``class_name`` / ``class_path`` components to the tables.

    A configured model, explainer or metric whose name is absent from its table but
    whose entry supplies ``class_name`` and ``class_path`` is imported from that
    source file and added, so the rest of the pipeline treats it like a built-in.

    Gated by ``general.allow_external_code`` (default ``False``), because importing
    the file executes it. If a path is supplied while the flag is off, a warning is
    logged and the component is left out, so the config check then reports the
    unknown name.

    Parameters
    ----------
    config : dict
        Resolved configuration dict.

    Raises
    ------
    TypeError
        If a loaded class is not a subclass of its family's base class
        (:class:`~xai4tsc.models.base.ModelBase`,
        :class:`~xai4tsc.xai.ExplainerBase`,
        :class:`~xai4tsc.evaluation.EvaluatorBase`).
    """
    from xai4tsc.utils.utils import load_class_from_path

    allow = config.get("general", {}).get("allow_external_code", False)

    def _load(kind: str, base: type, name: str, item: dict) -> type | None:
        path, class_name = item.get("class_path"), item.get("class_name")
        if not (path and class_name):
            return None
        if not allow:
            logger.warning(
                "External source '%s' specified for %s '%s' but "
                "general.allow_external_code is false; not loading.",
                path,
                kind,
                name,
            )
            return None
        cls = load_class_from_path(path, class_name)
        if not (isinstance(cls, type) and issubclass(cls, base)):
            raise TypeError(
                f"External {kind} '{name}' must be a {base.__name__} subclass, "
                f"got {cls!r}."
            )
        logger.info("Loaded external %s '%s' from %s", kind, name, path)
        return cls

    # Models and explainers are looked up case-insensitively, metrics by their
    # exact display name.
    families = (
        ("train_config", "models", "model", ModelBase, MODELS, str.lower),
        (
            "explanation_config",
            "explainers",
            "method",
            ExplainerBase,
            EXPLAINERS,
            str.lower,
        ),
        ("evaluation_config", "metrics", "metric", EvaluatorBase, METRICS, str),
    )
    for section, list_key, id_key, base, table, fold in families:
        for entry in config.get(section, {}).get(list_key, []):
            key = fold(entry[id_key])
            if key in table:
                continue
            if (cls := _load(id_key, base, entry[id_key], entry)) is not None:
                table[key] = cls


# ── Lookup families ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Lookup:
    """
    One config-string-to-object family.

    Parameters
    ----------
    table : dict
        Config name (lower case) -> the class it denotes.
    key : str
        The config key this family is configured by, used in every message so an
        error names the YAML key the user wrote (``"encode"``, ``"loss_func"``).
    label : str
        The thing being named, article included (``"an encoder"``, ``"a loss"``),
        so the sentences read correctly.
    example : str
        A dotted path to show in the "available" line.
    kind : type, optional
        Base class a dotted path must resolve to a subclass of.
    requires : tuple of str, optional
        Methods a dotted-path class must have, for a family without a base class
        (sklearn-style encoders).
    params_key : str, optional
        The config key holding the constructor arguments, for the "parameters
        without a name" message.
    retired : dict, optional
        Names that used to work, mapped to the reason they no longer do. A retired
        name must say what to use instead: a config carrying one stops running.
    """

    table: dict[str, type]
    key: str
    label: str
    example: str
    kind: type | None = None
    requires: tuple[str, ...] = ()
    params_key: str = ""
    retired: dict[str, str] = field(default_factory=dict)

    def given(self, name: str | None, params: dict | None) -> bool:
        """
        Return whether *name* is set, refusing parameters without one.

        Parameters
        ----------
        name : str or None
            The configured name; ``None`` or a blank string means unset.
        params : dict or None
            The configured constructor arguments.

        Returns
        -------
        bool
            ``False`` when the family's default applies.

        Raises
        ------
        ValueError
            If *params* were given without a *name*; they would be discarded.
        """
        if name is not None and not (isinstance(name, str) and not name.strip()):
            return True
        if params:
            article = "an" if self.key[0] in "aeiou" else "a"
            raise ValueError(
                f"{self.params_key} {params!r} were given without {article} "
                f"{self.key} to apply them to. Name {self.label} as well, or drop "
                "the parameters."
            )
        return False

    def resolve(self, name: str) -> type:
        """
        Return the class that *name* denotes.

        Table lookups are case-folded, because configs in the wild spell a loss
        ``"CrossEntropy"``; a dotted path is left alone, since
        ``torch.nn.BCEWithLogitsLoss`` cannot resolve lower-cased.

        Parameters
        ----------
        name : str
            A key of :attr:`table`, or a dotted import path to a class.

        Returns
        -------
        type
            The class. Constructing it is the caller's business.

        Raises
        ------
        ValueError
            If *name* is not a string, is retired, resolves to nothing, or a dotted
            path resolves to something other than a class of the family's kind.
            Falling back to a default instead would run a different experiment
            and report plausible numbers for it.
        """
        if not isinstance(name, str):
            raise ValueError(
                f"{self.key} must be a string naming {self.label}, got "
                f"{type(name).__name__} ({name!r}). {self._available()}"
            )
        folded = name.lower()
        if folded in self.table:
            return self.table[folded]
        if folded in self.retired:
            raise ValueError(
                f"{self.key}: '{name}' is no longer available because "
                f"{self.retired[folded]}. {self._available()}"
            )
        if "." in name:
            return self._from_dotted_path(name)
        raise ValueError(f"Unknown {self.key}: '{name}'. {self._available()}")

    def construct(self, name: str, params: dict | None = None) -> object:
        """
        Resolve *name* and build it with *params*.

        Parameters
        ----------
        name : str
            As :meth:`resolve`.
        params : dict, optional
            Constructor keyword arguments.

        Returns
        -------
        object
            The constructed instance.

        Raises
        ------
        ValueError
            If construction fails. The parameters come from YAML, so a wrong name
            or a wrong shape is a config error and is reported as one.
        """
        factory = self.resolve(name)
        params = params or {}
        try:
            return factory(**params)
        except (TypeError, ValueError) as exc:
            if params:
                raise ValueError(
                    f"{self.key}: '{name}' could not be constructed from {params!r} "
                    f"({exc}). Check the parameter names and values against "
                    f"{getattr(factory, '__name__', factory)}."
                ) from exc
            raise ValueError(
                f"{self.key}: '{name}' could not be constructed without arguments "
                f"({exc}). Supply one that takes none, or pass the object directly "
                "from Python instead of via YAML."
            ) from exc

    def _available(self) -> str:
        """Return the sentence listing what this family accepts."""
        return (
            f"Available: {sorted(self.table)}, or a dotted path to {self.label} "
            f"class, e.g. '{self.example}'."
        )

    def _from_dotted_path(self, path: str) -> type:
        """
        Import the class named by a dotted *path* and check its kind.

        Parameters
        ----------
        path : str
            Dotted path, e.g. ``"torch.optim.NAdam"``.

        Returns
        -------
        type
            The imported class.

        Raises
        ------
        ValueError
            If the module or the attribute does not exist, or it is not a class of
            the family's kind. Nothing is called before this check passes.
        """
        try:
            candidate = pkgutil.resolve_name(path)
        except (ImportError, AttributeError, ValueError) as exc:
            raise ValueError(
                f"Unknown {self.key}: '{path}' could not be imported ({exc}). "
                f"{self._available()}"
            ) from exc
        if self.kind is not None:
            ok = isinstance(candidate, type) and issubclass(candidate, self.kind)
            wanted = f"a subclass of {self.kind.__name__}"
        else:
            ok = isinstance(candidate, type) and all(
                callable(getattr(candidate, method, None)) for method in self.requires
            )
            wanted = f"a class with {' and '.join(self.requires)}"
        if not ok:
            raise ValueError(
                f"{self.key}: '{path}' is not {wanted}, so it cannot be "
                f"{self.label}. {self._available()}"
            )
        logger.info("Resolved %s '%s' to %s.", self.key, path, candidate.__name__)
        return candidate


# ── encoders ──────────────────────────────────────────────────────────────────

ENCODERS: dict[str, type] = {
    "label": LabelEncoder,
    "multihot": MultiHotLabelEncoder,
    "powerset": PowersetLabelEncoder,
}
"""Config name -> an sklearn-style target encoder class, constructed per dataset."""

_RETIRED_ENCODERS: dict[str, str] = {
    "onehot": (
        "it used a scikit-learn feature encoder in the label slot; use 'label', "
        "which produced identical class indices on scalar labels"
    ),
    "ordinal": (
        "it used a scikit-learn feature encoder in the label slot; use 'label', "
        "since a class index is nominal and its order changes no result"
    ),
}
"""Names that used to work, with the reason they no longer do.

``"multihot"`` is deliberately absent. It used to select the label-powerset
reduction and now keeps the multi-hot rows, so it is *reused* rather than
retired, and a name cannot be both (a table hit is resolved before this map is
consulted). A stale config does not silently switch formulation: the rows are a
2-D target, and ``ModelBase._check_loss`` refuses an index-target loss such as
``CrossEntropy`` against one before the first epoch, naming the fix. A config
that only evaluates a checkpoint fails as loudly, on the output-width mismatch
when the state dict is loaded.
"""

_ENCODER_LOOKUP = Lookup(
    table=ENCODERS,
    key="encode",
    label="an encoder",
    example="sklearn.preprocessing.LabelBinarizer",
    requires=("fit", "transform"),
    params_key="encode_params",
    retired=_RETIRED_ENCODERS,
)


def build_encoder(name: str | None, params: dict | None = None) -> object:
    """
    Return a fresh encoder for the config value *name*.

    The runner keeps the string as well as the object, because the split-cache
    directory name is built from it
    (:func:`~experiment_runner.cache.get_split_cache_path`).

    Parameters
    ----------
    name : str or None
        A key of :data:`ENCODERS`, or a dotted import path to an encoder class
        with ``fit`` and ``transform`` (e.g.
        ``"sklearn.preprocessing.MultiLabelBinarizer"``). ``None`` or an empty
        string gives the default :class:`~sklearn.preprocessing.LabelEncoder`.
    params : dict, optional
        Constructor arguments, e.g. ``{"classes": ["7Hz", "19Hz", "37Hz"]}`` for
        :class:`~xai4tsc.data.MultiHotLabelEncoder`. Most encoders take none.

    Returns
    -------
    object
        An unfitted encoder, ready to hand to
        :meth:`~xai4tsc.data.base.DatasetBase.split`.

    Raises
    ------
    ValueError
        If *name* is not a string, is a retired name, or resolves to nothing;
        falling back to the default instead would train a different problem and
        report plausible numbers for it. Also if *params* were given without a
        *name* to apply them to.
    """
    if not _ENCODER_LOOKUP.given(name, params):
        return LabelEncoder()
    return _ENCODER_LOOKUP.construct(name, params)


# ── losses ────────────────────────────────────────────────────────────────────

LOSSES: dict[str, type[nn.Module]] = {
    "crossentropy": nn.CrossEntropyLoss,
    "bcewithlogits": nn.BCEWithLogitsLoss,
    "nll": nn.NLLLoss,
}
"""Config name -> loss class. Anything else resolves by dotted path."""

_LOSS_LOOKUP = Lookup(
    table=LOSSES,
    key="loss_func",
    label="a loss",
    example="torch.nn.BCEWithLogitsLoss",
    kind=nn.Module,
    params_key="loss_params",
)


def build_loss(name: str | None, params: dict | None = None) -> nn.Module | None:
    """
    Return the loss object for the config value *name*.

    The package takes the loss object itself (see
    :meth:`~xai4tsc.models.base.ModelBase.train_model`), so every ``nn.*Loss`` and
    every custom :class:`~torch.nn.Module` with a ``(logits, targets) -> scalar``
    call works without a table entry.

    Parameters
    ----------
    name : str or None
        A key of :data:`LOSSES` (matched case-insensitively, since configs spell
        it ``"CrossEntropy"``), or a dotted import path to a
        :class:`~torch.nn.Module` subclass. ``None`` returns ``None``, which
        leaves :meth:`~xai4tsc.models.base.ModelBase.train_model` on its own
        default so the default lives in one place.
    params : dict, optional
        Constructor arguments, e.g. ``{"weight": [1.0, 3.0, 1.0]}``. List values
        become ``float32`` tensors - see :func:`as_tensors`.

    Returns
    -------
    torch.nn.Module or None
        The constructed loss, or ``None`` when no name was given.

    Raises
    ------
    ValueError
        If *name* resolves to nothing, if construction fails, or if *params* were
        given without a *name* to apply them to.
    """
    if not _LOSS_LOOKUP.given(name, params):
        return None
    return _LOSS_LOOKUP.construct(name, as_tensors(params or {}))


def as_tensors(params: dict) -> dict:
    """
    Return *params* with every list value converted to a ``float32`` tensor.

    The tensor-valued arguments of the torch loss family - ``weight`` for
    :class:`~torch.nn.CrossEntropyLoss` and :class:`~torch.nn.NLLLoss`,
    ``weight`` and ``pos_weight`` for :class:`~torch.nn.BCEWithLogitsLoss` - are
    registered buffers, and a list cannot be assigned to one: the constructor
    raises ``cannot assign 'list' object to buffer 'weight'``. YAML has no tensor,
    so the conversion happens here. ``float32`` specifically, because a
    ``float64`` weight makes the loss raise ``expected scalar type Float but
    found Double`` at the first call.

    Parameters
    ----------
    params : dict
        Constructor arguments as they came from YAML.

    Returns
    -------
    dict
        The same mapping with list and tuple values as tensors.
    """
    converted = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            converted[key] = torch.tensor(value, dtype=torch.float32)
            logger.debug("Converted loss param '%s' to a float32 tensor.", key)
        else:
            converted[key] = value
    return converted


# ── optimizers ────────────────────────────────────────────────────────────────

OPTIMIZERS: dict[str, type[optim.Optimizer]] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}
"""Config name -> optimizer class. Anything else resolves by dotted path."""

_OPTIMIZER_LOOKUP = Lookup(
    table=OPTIMIZERS,
    key="optimizer",
    label="an optimizer",
    example="torch.optim.NAdam",
    kind=optim.Optimizer,
    params_key="optimizer_params",
)


def build_optimizer(name: str | None, params: dict | None = None) -> Callable | None:
    """
    Return the optimizer factory for the config value *name*.

    An optimizer cannot be fully constructed here: it needs
    ``model.parameters()``, which only exists inside
    :meth:`~xai4tsc.models.base.ModelBase.train_model`. So this returns the class,
    or a :func:`functools.partial` carrying the configured arguments, and the
    package calls it with ``(self.parameters(), lr=learn_rate)``.

    Parameters
    ----------
    name : str or None
        A key of :data:`OPTIMIZERS` (matched case-insensitively), or a dotted
        import path to a :class:`torch.optim.Optimizer` subclass. ``None``
        returns ``None``, which leaves
        :meth:`~xai4tsc.models.base.ModelBase.train_model` on its own default so
        the default lives in one place.
    params : dict, optional
        Constructor arguments beyond the parameters and the learning rate, e.g.
        ``{"momentum": 0.9, "nesterov": true}``. They are bound into a
        :func:`functools.partial`; ``learn_rate`` stays a ``train_model``
        parameter and is applied there.

    Returns
    -------
    Callable or None
        The class, a partial of it, or ``None`` when no name was given.

    Raises
    ------
    ValueError
        If *name* resolves to nothing, or if *params* were given without a *name*
        to apply them to.
    """
    if not _OPTIMIZER_LOOKUP.given(name, params):
        return None
    factory = _OPTIMIZER_LOOKUP.resolve(name)
    # Returning the bare class when there is nothing to bind keeps the run log
    # and any repr readable.
    return partial(factory, **params) if params else factory


# ── transforms ────────────────────────────────────────────────────────────────

TRANSFORMS: dict[str, type[DomainTransform]] = {
    "fft": FourierTransform,
    "rfft": RFFTransform,
    "stft": STFTransform,
}
"""Config ``transform.name`` -> transform class. Anything else resolves by path."""

_TRANSFORM_LOOKUP = Lookup(
    table=TRANSFORMS,
    key="transform",
    label="a transform",
    example="xai4tsc.utils.STFTransform",
    kind=DomainTransform,
)


def build_transform(entry: dict | None) -> DomainTransform | None:
    """
    Build the transform object an explainer entry's ``transform`` block names.

    Parameters
    ----------
    entry : dict or None
        ``{"name": ..., "params": {...}}`` from the config, or ``None`` to leave
        the explainer on its default.

    Returns
    -------
    DomainTransform or None
        The constructed transform, or ``None`` when *entry* is ``None``.

    Raises
    ------
    ValueError
        If *entry* is not a mapping, or its name or parameters do not build a
        transform.
    """
    if entry is None:
        return None
    if not isinstance(entry, dict):
        raise ValueError(
            f"transform must be a mapping with 'name' and 'params', got {entry!r}."
        )
    return _TRANSFORM_LOOKUP.construct(entry.get("name"), entry.get("params"))


# ── Quantus perturbation functions ────────────────────────────────────────────

# shortcut: Quantus' own perturbation functions only, no dotted path to a custom
# one (a function has no kind to check); add one if a config needs it.
PERTURB_FUNCS: dict[str, Callable] = {
    name: func
    for name, func in inspect.getmembers(_quantus_perturb, inspect.isfunction)
    if func.__module__ == _quantus_perturb.__name__
}
"""Config ``perturb_func`` name on a Quantus metric -> Quantus perturbation function."""


def build_perturb_func(name: str) -> Callable:
    """
    Return the Quantus perturbation function a metric's ``perturb_func`` names.

    Parameters
    ----------
    name : str
        A key of :data:`PERTURB_FUNCS`.

    Returns
    -------
    Callable
        The Quantus function.

    Raises
    ------
    ValueError
        If *name* is a TSHAP baseline name or not a Quantus perturbation function.
        A baseline name used to resolve to TSHAP's ``(C, T)`` function, which
        Quantus would call with ``(B, C*T)`` rows.
    """
    if name in BASELINES:
        raise ValueError(
            f"perturb_func {name!r} is a TSHAP baseline name, not a Quantus "
            "perturbation function. Drop perturb_func (the metric's default "
            "already replaces by baseline) and set "
            f"perturb_baseline: {name}."
        )
    if name not in PERTURB_FUNCS:
        raise ValueError(
            f"Unknown perturb_func {name!r}. Available: {sorted(PERTURB_FUNCS)}."
        )
    return PERTURB_FUNCS[name]
