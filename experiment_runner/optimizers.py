"""
Optimizer lookup: the config string to a ``torch.optim`` factory.

Unlike a loss, an optimizer cannot be fully constructed here: it needs
``model.parameters()``, which only exists inside
:meth:`~xai4tsc.models.base.ModelBase.train_model`. So this returns the class, or
a :func:`functools.partial` carrying the configured arguments, and the package
calls it with ``(self.parameters(), lr=learn_rate)``. That keeps the package API
at one object per family - the same shape as the loss and the encoder - and uses
the ``partial`` idiom the metric registry already relies on.
"""

import logging
from collections.abc import Callable
from functools import partial

from torch import optim

from .lookups import Lookup

logger = logging.getLogger("xai4tsc.runner.optimizers")

OPTIMIZERS: dict[str, type[optim.Optimizer]] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}
"""Config name -> optimizer class. Anything else resolves by dotted path."""

_LOOKUP = Lookup(
    table=OPTIMIZERS,
    key="optimizer",
    label="an optimizer",
    example="torch.optim.NAdam",
)


def build_optimizer(name: str | None, params: dict | None = None) -> Callable | None:
    """
    Return the optimizer factory for the config value *name*.

    Parameters
    ----------
    name : str or None
        A key of :data:`OPTIMIZERS` (matched case-insensitively), or a dotted
        import path to an optimizer class. ``None`` returns ``None``, which leaves
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
    params = params or {}
    if name is None or (isinstance(name, str) and not name.strip()):
        if params:
            raise ValueError(
                f"optimizer_params {params!r} were given without an optimizer to "
                "apply them to. Name the optimizer as well, or drop the parameters."
            )
        return None
    factory = _LOOKUP.resolve(name)
    # Returning the bare class when there is nothing to bind keeps the run log
    # and any repr readable.
    return partial(factory, **params) if params else factory
