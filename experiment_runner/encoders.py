"""
Label-encoder lookup: the config string to a scikit-learn target encoder.

The package takes an encoder *object* (see
:meth:`~xai4tsc.data.base.DatasetBase.split`), so the only gap is that YAML cannot
hold one. That is a serialization problem rather than an abstraction, which is why
this is a table and a dotted-path escape hatch instead of a registry: there is
nothing to curate, and an encoder that does not exist yet still works.

The runner keeps the string as well as the object, because the split-cache
directory name is built from it (:func:`~experiment_runner.cache.get_split_cache_path`).
"""

import importlib
import logging
from collections.abc import Callable

from sklearn.preprocessing import LabelEncoder

from xai4tsc.data import MultiHotLabelEncoder

logger = logging.getLogger("xai4tsc.runner.encoders")

ENCODERS: dict[str, Callable[[], object]] = {
    "label": LabelEncoder,
    "multihot": MultiHotLabelEncoder,
}
"""Config name -> callable returning a fresh, unfitted encoder."""

_RETIRED: dict[str, str] = {
    "onehot": (
        "it used a scikit-learn feature encoder in the label slot; use 'label', "
        "which produced identical class indices on scalar labels"
    ),
    "ordinal": (
        "it used a scikit-learn feature encoder in the label slot; use 'label', "
        "since a class index is nominal and its order changes no result"
    ),
}
"""Names that used to work, with the reason they no longer do."""


def build_encoder(name: str | None) -> object:
    """
    Return a fresh encoder for the config value *name*.

    Parameters
    ----------
    name : str or None
        A key of :data:`ENCODERS`, or a dotted import path to an encoder class
        (e.g. ``"sklearn.preprocessing.LabelBinarizer"``). ``None`` or an empty
        string gives the default :class:`~sklearn.preprocessing.LabelEncoder`.

    Returns
    -------
    object
        An unfitted encoder, ready to hand to
        :meth:`~xai4tsc.data.base.DatasetBase.split`.

    Raises
    ------
    ValueError
        If *name* is not a string, is a retired name, or resolves to nothing.
        Falling back to the default instead would train a different problem and
        report plausible numbers for it.
    """
    if name is None:
        return LabelEncoder()
    if not isinstance(name, str):
        raise ValueError(
            f"encode must be a string naming an encoder, got "
            f"{type(name).__name__} ({name!r}). Available: {sorted(ENCODERS)}, or "
            "a dotted path to an encoder class."
        )
    if not name.strip():
        return LabelEncoder()
    if name in ENCODERS:
        return ENCODERS[name]()
    if name in _RETIRED:
        raise ValueError(
            f"encode: '{name}' is no longer available because {_RETIRED[name]}. "
            f"Available: {sorted(ENCODERS)}, or a dotted path to an encoder class."
        )
    if "." in name:
        return _from_dotted_path(name)
    raise ValueError(
        f"Unknown encode: '{name}'. Available: {sorted(ENCODERS)}, or a dotted path "
        "to an sklearn-style target encoder class, e.g. "
        "'sklearn.preprocessing.LabelBinarizer'."
    )


def _from_dotted_path(path: str) -> object:
    """
    Import and instantiate the encoder class named by a dotted *path*.

    Parameters
    ----------
    path : str
        Dotted path to a class, e.g. ``"sklearn.preprocessing.LabelBinarizer"``.

    Returns
    -------
    object
        An unfitted instance.

    Raises
    ------
    ValueError
        If the module or attribute does not exist, or the attribute cannot be
        called with no arguments.
    """
    module_path, _, attr = path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"Unknown encode: '{path}' - module '{module_path}' could not be "
            f"imported ({exc}). Available: {sorted(ENCODERS)}."
        ) from exc
    candidate = getattr(module, attr, None)
    if candidate is None or not callable(candidate):
        raise ValueError(
            f"Unknown encode: '{path}' - '{module_path}' has no callable '{attr}'. "
            f"Available: {sorted(ENCODERS)}."
        )
    try:
        encoder = candidate()
    except TypeError as exc:
        raise ValueError(
            f"encode: '{path}' could not be constructed without arguments ({exc}). "
            "Supply an encoder that takes none, or pass the object directly from "
            "Python instead of via YAML."
        ) from exc
    logger.info("Resolved encode '%s' to %s.", path, type(encoder).__name__)
    return encoder
