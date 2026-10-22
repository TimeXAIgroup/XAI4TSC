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

from xai4tsc.data import MultiHotLabelEncoder, PowersetLabelEncoder

from .lookups import Lookup

logger = logging.getLogger("xai4tsc.runner.encoders")

ENCODERS: dict[str, type] = {
    "label": LabelEncoder,
    "multihot": MultiHotLabelEncoder,
    "powerset": PowersetLabelEncoder,
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

_LOOKUP = Lookup(
    table=ENCODERS,
    key="encode",
    label="an encoder",
    example="sklearn.preprocessing.LabelBinarizer",
    retired=_RETIRED,
)


def build_encoder(name: str | None, params: dict | None = None) -> object:
    """
    Return a fresh encoder for the config value *name*.

    Parameters
    ----------
    name : str or None
        A key of :data:`ENCODERS`, or a dotted import path to an encoder class
        (e.g. ``"sklearn.preprocessing.MultiLabelBinarizer"``). ``None`` or an
        empty string gives the default
        :class:`~sklearn.preprocessing.LabelEncoder`.
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
    params = params or {}
    if name is None or (isinstance(name, str) and not name.strip()):
        if params:
            raise ValueError(
                f"encode_params {params!r} were given without an encode to apply "
                "them to. Name the encoder as well, or drop the parameters."
            )
        return LabelEncoder()
    return _LOOKUP.construct(name, params)
