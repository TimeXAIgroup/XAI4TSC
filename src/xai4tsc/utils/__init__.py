"""
Shared utilities used across the package.

Submodules:

- :mod:`~xai4tsc.utils.plot` — relevance and perturbation-curve plotting.
- :mod:`~xai4tsc.utils.fourier_transforms` — the ``DomainTransform`` family
  (Fourier / rFFT / STFT) and the :func:`get_transform` factory.
- :mod:`~xai4tsc.utils.utils` — :func:`dict_to_args`, :func:`merge_dicts`,
  :func:`rescale_array`, :func:`load_class_from_path`.
- :mod:`~xai4tsc.utils.animation` — GIF animation of perturbation sweeps.
- :mod:`~xai4tsc.utils.perturbation` — shared perturbation functions.

The seeding helpers (:data:`DEFAULT_SEED`, :func:`derive_seed`,
:func:`resolve_rng`) are re-exported here from the dependency-free
:mod:`xai4tsc._random`, which is where the package's layers import them from
directly to avoid pulling in the plotting stack.
"""

from .._random import DEFAULT_SEED, derive_seed, resolve_rng
from .fourier_transforms import (
    DomainTransform,
    FourierTransform,
    RFFTransform,
    STFTransform,
    get_transform,
)
from .plot import (
    plot_perturbation_curve,
    plot_relevance,
    plot_relevance_f,
    plot_relevance_tf,
)
from .utils import dict_to_args, load_class_from_path, merge_dicts, rescale_array

__all__ = [
    "DEFAULT_SEED",
    "DomainTransform",
    "FourierTransform",
    "RFFTransform",
    "STFTransform",
    "derive_seed",
    "dict_to_args",
    "get_transform",
    "load_class_from_path",
    "merge_dicts",
    "plot_perturbation_curve",
    "plot_relevance",
    "plot_relevance_f",
    "plot_relevance_tf",
    "rescale_array",
    "resolve_rng",
]
