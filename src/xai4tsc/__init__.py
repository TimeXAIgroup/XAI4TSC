"""
Top-level namespace for the xai4tsc package.

Exports the public API: registries (``MODELS``, ``EXPLAINERS``, ``METRICS``),
registration functions, and the main entry points
:func:`generate_explanation`, :func:`evaluate`, :func:`load_dataset`, and
:func:`plot_relevance`.
"""

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xai4tsc")
except PackageNotFoundError:  # not installed (e.g. running from a raw checkout)
    __version__ = "0.0.0+unknown"

# Submodules for lower-level access
from . import data, evaluation, models, utils, xai
from .data import (
    SYNTHETIC_DATASETS,
    DatasetBase,
    FreqShapesDataset,
    LocalDataset,
    SyntheticDataset,
    UcrUeaDataset,
    load_dataset,
    register_synthetic_dataset,
)
from .evaluation import (
    METRICS,
    QUANTUS_METRICS,
    EvaluatorBase,
    QuantusEvaluator,
    evaluate,
    register_metric,
)
from .models import (
    FCN,
    LSTM,
    MODELS,
    XLSTM,
    LeNet,
    ModelBase,
    PatchTST,
    ResNet,
    register_model,
)
from .utils import (
    plot_perturbation_curve,
    plot_relevance,
    plot_relevance_f,
    plot_relevance_tf,
)
from .xai import (
    EXPLAINERS,
    DataType,
    Domain,
    ExplainerBase,
    Explanation,
    GradientExplainer,
    PerturbationExplainer,
    SurrogateExplainer,
    generate_explanation,
    register_explainer,
)

# Library convention: add NullHandler so xai4tsc is silent by default.
# The standalone runner (main.py) installs real handlers on this logger.
logging.getLogger("xai4tsc").addHandler(logging.NullHandler())

__all__ = [  # noqa: RUF022 — grouped by role, not alphabetical
    # Version
    "__version__",
    # Base classes
    "ModelBase",
    "FCN",
    "LeNet",
    "ResNet",
    "LSTM",
    "PatchTST",
    "XLSTM",
    "DatasetBase",
    "UcrUeaDataset",
    "LocalDataset",
    "SyntheticDataset",
    "FreqShapesDataset",
    "SYNTHETIC_DATASETS",
    "register_synthetic_dataset",
    "load_dataset",
    "ExplainerBase",
    "GradientExplainer",
    "PerturbationExplainer",
    "SurrogateExplainer",
    "EvaluatorBase",
    "QuantusEvaluator",
    # Core types
    "Explanation",
    "DataType",
    "Domain",
    # Entry points
    "generate_explanation",
    "evaluate",
    # Registries
    "MODELS",
    "EXPLAINERS",
    "METRICS",
    "QUANTUS_METRICS",
    # Registration hooks
    "register_model",
    "register_explainer",
    "register_metric",
    # Visualization
    "plot_relevance",
    "plot_relevance_f",
    "plot_relevance_tf",
    "plot_perturbation_curve",
    # Submodules
    "data",
    "evaluation",
    "models",
    "utils",
    "xai",
]
