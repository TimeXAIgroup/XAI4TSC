"""
FreqRISE — explaining time series by random frequency masking.

Perturbation-based attribution that masks coefficients in the frequency (real
FFT) or time-frequency (STFT) domain and measures the effect on the model's
predicted class probability. Univariate series (``C == 1``) are the validated
path; multivariate inputs are masked with a shared coefficient mask broadcast
across channels.

Method
------
For an invertible spectral transform ``g`` and a random binary mask ``M`` over
the coefficient grid, the masked signal fed to the classifier is
``x̂(M) = g⁻¹(g(x) ⊙ M)``. Averaging the masked predictions weighted by the masks
gives the per-coefficient relevance for class ``c`` (Brüsch et al., Eq. 3):

    R_c = 1 / (N · E[M]) · Σ_n  ŷ_c(x̂(M_n)) · M_n

where ``N`` is the number of masks, ``ŷ_c`` is the softmax probability of the
target class, and ``E[M]`` is the Bernoulli keep probability. Masks are sampled
on a coarse ``num_cells`` grid and interpolated up to the coefficient grid to
make them smooth.

Reference
---------
T. Brüsch, K. K. Wickstrøm, M. N. Schmidt, T. S. Alstrøm, and R. Jenssen,
"FreqRISE: Explaining time series using frequency masking," in Proc. 6th Northern
Lights Deep Learning Conf. (NLDL), PMLR vol. 265, 2025, pp. 16-31.
https://proceedings.mlr.press/v265/brusch25a.html (arXiv:2406.13584)

This is an independent reimplementation written from the published algorithm
(Eq. 3 and the masking procedure above); it does not derive from any third-party
FreqRISE source code.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from torch.fft import irfft, rfft
from torch.nn.functional import interpolate

from ..utils.fourier_transforms import (
    DomainTransform,
    RFFTransform,
    resolve_transform,
)
from ._types import DataType, Domain
from .base import PerturbationExplainer

if TYPE_CHECKING:
    from ._types import Explanation

logger = logging.getLogger(__name__)


class FreqRISEExplainer(PerturbationExplainer):
    """
    FreqRISE: attribution via random masking in the frequency / TF domain.

    Parameters
    ----------
    batch_size : int
        Number of masks evaluated per forward pass.
    num_batches : int
        Number of mask batches; the total mask count is
        ``num_batches * batch_size``.
    domain : str
        ``"fft"`` (frequency, via real FFT) or ``"stft"`` (time-frequency).
    transform : DomainTransform or dict, optional
        Transform used for ``domain="stft"`` (an
        :class:`~xai4tsc.utils.fourier_transforms.STFTransform`). Required for
        ``"stft"``; ignored for ``"fft"`` (which uses ``torch.fft.rfft``).
    num_cells : int
        Resolution of the coarse mask grid before it is interpolated up to the
        coefficient grid. ``num_cells // 2`` must be smaller than the smallest
        transform dimension.
    probability_of_drop : float
        Bernoulli probability that a coarse cell is kept; equals ``E[M]`` in the
        relevance normalisation (default 0.5, as in the paper).
    seed : int, optional
        Seed for reproducible mask sampling.
    """

    data_applicability: ClassVar[set[DataType]] = {DataType.TIME_SERIES}
    explanation_domains: ClassVar[set[Domain]] = {
        Domain.FREQUENCY,
        Domain.TIME_FREQUENCY,
    }

    def __init__(
        self,
        batch_size: int = 10,
        num_batches: int = 300,
        domain: str = "stft",
        transform: DomainTransform | dict | None = None,
        num_cells: int = 50,
        probability_of_drop: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if domain not in ("fft", "stft"):
            raise ValueError(f"domain must be 'fft' or 'stft', got {domain!r}.")
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.domain = domain
        self.transform = resolve_transform(transform)
        if domain == "stft" and self.transform is None:
            raise ValueError(
                "FreqRISE with domain='stft' requires a time-frequency transform "
                "(e.g. an STFTransform); none was supplied."
            )
        # FFT mode masks the one-sided real FFT directly. Expose a matching
        # RFFTransform so the explanation carries a transform that downstream
        # metrics can forward/invert (its shape matches the (n, C, T//2+1)
        # relevance grid).
        if domain == "fft" and self.transform is None:
            self.transform = RFFTransform()
        self.num_cells = num_cells
        self.probability_of_drop = probability_of_drop
        self.seed = seed

    def explain(
        self,
        model: torch.nn.Module,
        exp: Explanation,
        device: str | torch.device,
        targets: list | None,
        **kwargs: object,
    ) -> np.ndarray:
        """Compute FreqRISE relevance for the samples in *exp*."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        data = torch.as_tensor(exp.data, dtype=torch.float32)
        timesteps = data.shape[-1]
        if isinstance(self.transform, RFFTransform):
            self.transform.n = timesteps
        if targets is None:
            targets = model.predict(exp.data)[0].tolist()

        model = model.eval().to(device)
        spatial_dims = 2 if self.domain == "stft" else 1

        relevances = []
        for idx, (sample, target) in enumerate(zip(data, targets, strict=True)):
            logger.info("FreqRISE: explaining sample %d / %d", idx + 1, len(data))
            with torch.no_grad():
                relevance = self._relevance(
                    model, sample.to(device), int(target), device, spatial_dims
                )
            relevances.append(self._rescale(relevance.cpu()))

        # (n_samples, *grid) → insert the channel axis: (n, 1, *grid).
        result = torch.stack(relevances).unsqueeze(1).numpy()
        exp.explanation_domain = (
            Domain.FREQUENCY if self.domain == "fft" else Domain.TIME_FREQUENCY
        )
        return result

    def _relevance(
        self,
        model: torch.nn.Module,
        sample: torch.Tensor,
        target: int,
        device: str | torch.device,
        spatial_dims: int,
    ) -> torch.Tensor:
        """Accumulate Eq. 3 relevance over all masks for a single ``(C, T)`` sample."""
        timesteps = sample.shape[-1]
        coeffs = self._to_spectral(sample.unsqueeze(0))  # (1, C, *grid)
        grid = tuple(coeffs.shape[-spatial_dims:])
        if self.num_cells // 2 >= min(grid):
            raise ValueError(
                f"num_cells // 2 ({self.num_cells // 2}) must be smaller than the "
                f"smallest transform dimension {min(grid)} (grid {grid}); reduce "
                f"num_cells or use a finer transform."
            )

        # Eq. 3 numerator: Σ_n ŷ_c(x̂(M_n)) · M_n, accumulated over mask batches.
        importance = torch.zeros(grid, device=device)
        for _ in range(self.num_batches):
            masks = self._sample_masks(self.batch_size, grid, device)  # (B, *grid)
            masked = coeffs * masks.unsqueeze(1)  # broadcast over channels
            recon = self._from_spectral(masked, timesteps)  # (B, C, T)
            logits = model(recon.float())
            probs = torch.softmax(logits, dim=1)[:, target]  # ŷ_c per mask
            importance += torch.tensordot(probs, masks, dims=([0], [0]))

        total_masks = self.num_batches * self.batch_size
        return importance / (total_masks * self.probability_of_drop)

    def _to_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a time-domain signal to the masking (coefficient) domain."""
        if self.domain == "fft":
            return rfft(x)
        return self.transform.forward(x)

    def _from_spectral(self, coeffs: torch.Tensor, timesteps: int) -> torch.Tensor:
        """Invert masked coefficients back to a ``(B, C, T)`` time-domain signal."""
        if self.domain == "fft":
            return irfft(coeffs, n=timesteps, dim=-1)
        return self.transform.inverse(coeffs, length=timesteps, return_complex=False)

    def _sample_masks(
        self,
        batch: int,
        grid: tuple[int, ...],
        device: str | torch.device,
    ) -> torch.Tensor:
        """
        Sample a batch of smooth random masks over the coefficient *grid*.

        Bernoulli cells on a coarse ``num_cells`` grid are interpolated up to the
        full coefficient grid. Masks are drawn on CPU so a fixed seed gives the
        same masks on any device (and to avoid backend gaps in ``interpolate``).
        """
        spatial = len(grid)
        coarse = (
            torch.rand(batch, 1, *((self.num_cells,) * spatial))
            < self.probability_of_drop
        ).float()
        if spatial == 1:
            smooth = interpolate(
                coarse, size=grid[-1], mode="linear", align_corners=False
            )
        else:
            smooth = interpolate(
                coarse, size=tuple(grid), mode="bilinear", align_corners=False
            )
        return smooth.squeeze(1).to(device)

    @staticmethod
    def _rescale(relevance: torch.Tensor) -> torch.Tensor:
        """Min-max scale a relevance map to ``[0, 1]`` (identity if constant)."""
        low, high = relevance.min(), relevance.max()
        if high > low:
            return (relevance - low) / (high - low)
        return relevance
