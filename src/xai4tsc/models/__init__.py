"""
TSC model base class, built-in models, and the ``MODELS`` registry.

Submodules:

- :mod:`~xai4tsc.models.models` — built-in architectures (FCN, LeNet,
  ResNet, LSTM, XLSTM, PatchTST), the ``MODELS`` registry, and the
  :func:`load_model` factory.
- :mod:`~xai4tsc.models.base` — the :class:`ModelBase` ABC (training,
  prediction, evaluation, checkpoint save/load).

Subclass :class:`~xai4tsc.models.base.ModelBase` to add a custom model;
use :func:`~xai4tsc.models.models.load_model` to instantiate one by name or
from a checkpoint.
"""

from .base import ModelBase
from .models import (
    FCN,
    LSTM,
    MODELS,
    XLSTM,
    LeNet,
    PatchTST,
    ResNet,
    load_model,
    register_model,
)

__all__ = [
    "FCN",
    "LSTM",
    "MODELS",
    "XLSTM",
    "LeNet",
    "ModelBase",
    "PatchTST",
    "ResNet",
    "load_model",
    "register_model",
]
