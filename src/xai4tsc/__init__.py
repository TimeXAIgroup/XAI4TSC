"""
Top-level namespace for the xai4tsc package.

Exports the public API: the catalogs (``MODELS``, ``EXPLAINERS``, ``METRICS``,
``QUANTUS_METRICS``, ``SYNTHETIC_DATASETS``), the dataset classes, and the main
entry points :func:`load_model`, :func:`generate_explanation`, :func:`evaluate`, and
:func:`plot_relevance`, plus the seeding helpers
(:data:`DEFAULT_SEED`, :func:`derive_seed`, :func:`resolve_rng`) that make a
result reproducible on its own.
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
    LocalDataset,
    MultiHotLabelEncoder,
    PowersetLabelEncoder,
    SyntheticDataset,
    SynthMultilabelFreqDataset,
    UcrUeaDataset,
)
from .evaluation import (
    METRICS,
    QUANTUS_METRICS,
    EvaluatorBase,
    QuantusEvaluator,
    evaluate,
)
from .logging_config import disable_logging, enable_logging
from .models import (
    FCN,
    LSTM,
    MODELS,
    XLSTM,
    LeNet,
    ModelBase,
    PatchTST,
    ResNet,
    load_model,
)
from .utils import (
    DEFAULT_SEED,
    derive_seed,
    plot_perturbation_curve,
    plot_relevance,
    plot_relevance_f,
    plot_relevance_tf,
    resolve_rng,
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
)

# Library convention: add NullHandler so xai4tsc is silent by default.
# An application using the package installs real handlers on this logger.
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
    "MultiHotLabelEncoder",
    "PowersetLabelEncoder",
    "SyntheticDataset",
    "SynthMultilabelFreqDataset",
    "SYNTHETIC_DATASETS",
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
    # Randomness
    "DEFAULT_SEED",
    "derive_seed",
    "resolve_rng",
    # Entry points
    "load_model",
    "generate_explanation",
    "evaluate",
    # Logging
    "enable_logging",
    "disable_logging",
    # Catalogs
    "MODELS",
    "EXPLAINERS",
    "METRICS",
    "QUANTUS_METRICS",
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
