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
- :mod:`~xai4tsc.utils.defaults` — fraction-parameter configuration.
"""

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
    "DomainTransform",
    "FourierTransform",
    "RFFTransform",
    "STFTransform",
    "dict_to_args",
    "get_transform",
    "load_class_from_path",
    "merge_dicts",
    "plot_perturbation_curve",
    "plot_relevance",
    "plot_relevance_f",
    "plot_relevance_tf",
    "rescale_array",
]
