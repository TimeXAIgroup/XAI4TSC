"""
Loss lookup: the config string to a ``torch.nn`` loss object.

The package takes the loss object itself (see
:meth:`~xai4tsc.models.base.ModelBase.train_model`), so every ``nn.*Loss`` and
every custom :class:`~torch.nn.Module` with a ``(logits, targets) -> scalar``
call works without a table entry. The table exists only because YAML cannot hold
an object; see :mod:`experiment_runner.lookups` for the shared mechanics.
"""

import logging

import torch
from torch import nn

from .lookups import Lookup

logger = logging.getLogger("xai4tsc.runner.losses")

LOSSES: dict[str, type[nn.Module]] = {
    "crossentropy": nn.CrossEntropyLoss,
    "bcewithlogits": nn.BCEWithLogitsLoss,
    "nll": nn.NLLLoss,
}
"""Config name -> loss class. Anything else resolves by dotted path."""

_LOOKUP = Lookup(
    table=LOSSES,
    key="loss_func",
    label="a loss",
    example="torch.nn.BCEWithLogitsLoss",
)


def build_loss(name: str | None, params: dict | None = None) -> nn.Module | None:
    """
    Return the loss object for the config value *name*.

    Parameters
    ----------
    name : str or None
        A key of :data:`LOSSES` (matched case-insensitively, since configs spell
        it ``"CrossEntropy"``), or a dotted import path to a loss class.
        ``None`` returns ``None``, which leaves
        :meth:`~xai4tsc.models.base.ModelBase.train_model` on its own default so
        the default lives in one place.
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
    params = params or {}
    if name is None or (isinstance(name, str) and not name.strip()):
        if params:
            raise ValueError(
                f"loss_params {params!r} were given without a loss_func to apply "
                "them to. Name the loss as well, or drop the parameters."
            )
        return None
    return _LOOKUP.construct(name, as_tensors(params))


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
