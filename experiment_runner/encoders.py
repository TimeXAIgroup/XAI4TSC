"""
Label-encoder lookup: the config string to a scikit-learn target encoder.

The table and the dotted-path escape hatch are described in
:mod:`experiment_runner.lookups`, which this shares with the loss and optimizer
families. The runner keeps the string as well as the object, because the
split-cache directory name is built from it
(:func:`~experiment_runner.cache.get_split_cache_path`).
"""

import logging

from sklearn.preprocessing import LabelEncoder

from xai4tsc.data import MultiHotLabelEncoder

from .lookups import Lookup

logger = logging.getLogger("xai4tsc.runner.encoders")

ENCODERS: dict[str, type] = {
    "label": LabelEncoder,
    "multihot": MultiHotLabelEncoder,
}
"""Config name -> an sklearn-style target encoder class, constructed per dataset."""

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

_LOOKUP = Lookup(
    table=ENCODERS,
    key="encode",
    label="an encoder",
    example="sklearn.preprocessing.LabelBinarizer",
    retired=_RETIRED,
)


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
    if name is None or (isinstance(name, str) and not name.strip()):
        return LabelEncoder()
    return _LOOKUP.construct(name)
