"""
Top-level namespace for the xai4tsc package.

Exports the public API: registries (``MODELS``, ``EXPLAINERS``, ``METRICS``),
registration functions, and the main entry points
:func:`load_dataset`, :func:`load_model`, :func:`generate_explanation`,
:func:`evaluate`, and :func:`plot_relevance`, plus the seeding helpers
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
    register_model,
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
    register_explainer,
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
