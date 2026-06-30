"""
Domain transforms (Fourier / short-time Fourier) used by frequency-domain XAI.

A :class:`DomainTransform` maps a time-domain signal to a target domain
(frequency or time-frequency) and back. Explainers use the forward direction to
express relevance in the target domain; perturbation metrics use the inverse to
return a perturbed representation to the time domain before re-querying the
model. Transform objects are *stateful* (they may cache expensive setup) and are
meant to be built once and shared via :attr:`xai4tsc.Explanation.transform`.

Mixing Fourier implementations from different libraries (e.g. torch and scipy)
has historically produced inconsistent results, so this module standardises on
``torch.fft`` / ``torch.stft``.
"""

from abc import ABC, abstractmethod

import torch
from scipy.signal.windows import boxcar, tukey
from torch.signal.windows import hann


def _get_window(
    window: str,
    win_length: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Build a window tensor for the short-time Fourier transform.

    Parameters
    ----------
    window : str
        Window name: ``"hann"``, ``"boxcar"``/``"rectangle"``, or ``"tukey"``.
    win_length : int
        Length of the window in samples.
    device : str or torch.device
        Device to place the window tensor on.

    Returns
    -------
    torch.Tensor
        A ``float32`` window tensor of length *win_length*.

    Raises
    ------
    NotImplementedError
        If *window* is not a supported window name.
    """
    if window == "hann":
        win = hann(win_length)
    elif window in ("boxcar", "rectangle"):
        win = torch.tensor(boxcar(win_length))
    elif window == "tukey":
        win = torch.tensor(tukey(win_length))
    else:
        raise NotImplementedError(f"Window {window} not supported.")

    return win.to(device).to(torch.float32)


class DomainTransform(ABC):
    """
    Abstract base for invertible time-domain ↔ target-domain transforms.

    Subclasses implement :meth:`forward` (to the target domain) and
    :meth:`inverse` (back to the time domain). Instances are stateful and may
    cache setup between calls; build once and reuse.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Transform *x* from the time domain to the target domain."""

    @abstractmethod
    def inverse(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Transform *x* from the target domain back to the time domain."""


class FourierTransform(DomainTransform):
    """
    Discrete Fourier transform and its inverse.

    Wraps :func:`torch.fft.fft` / :func:`torch.fft.ifft` (full complex FFT).

    Parameters
    ----------
    n : int, optional
        Signal length passed to the underlying FFT (``None`` uses the input
        length).
    """

    def __init__(self, n: int | None = None) -> None:
        self.n = n

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """
        Apply the Fourier transform.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, ..., T)`` where ``B`` is the batch dimension.
            Higher-dimensional inputs are flattened so every series is
            transformed independently.
        **kwargs : object
            Extra keyword arguments forwarded to :func:`torch.fft.fft`.

        Returns
        -------
        torch.Tensor
            Complex frequencies of shape ``(B, -1, n_freq)``.
        """
        if x.dim() <= 1:
            raise ValueError(
                f"Input signals must have dimension 2 or higher, got {x.dim()}."
            )

        batch_size = x.shape[0]
        timeseries_len = x.shape[-1]
        x = x.reshape(-1, timeseries_len)

        frequencies = torch.fft.fft(x, **kwargs)

        return frequencies.reshape(batch_size, -1, frequencies.shape[-1])

    def inverse(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """
        Apply the inverse Fourier transform.

        Parameters
        ----------
        x : torch.Tensor
            Frequencies of shape ``(B, ..., n_freq)``. Higher-dimensional inputs
            are flattened so every spectrum is transformed independently.
        **kwargs : object
            Extra keyword arguments forwarded to :func:`torch.fft.ifft`.

        Returns
        -------
        torch.Tensor
            Reconstructed signals of shape ``(B, -1, T)``.
        """
        if x.dim() <= 1:
            raise ValueError(
                f"Input frequencies must have dimension 2 or higher, got {x.dim()}."
            )

        batch_size = x.shape[0]
        num_frequencies = x.shape[-1]
        x = x.reshape(-1, num_frequencies)

        signals = torch.fft.ifft(x, **kwargs)

        return signals.reshape(batch_size, -1, signals.shape[-1])


class RFFTransform(DomainTransform):
    """
    One-sided real Fourier transform and its inverse.

    Wraps :func:`torch.fft.rfft` / :func:`torch.fft.irfft`. For a length-``T`` real
    signal the forward yields ``T // 2 + 1`` complex bins. ``irfft`` needs the
    original length to invert exactly (it otherwise assumes an even length), so this
    transform is **stateful**: :meth:`forward` records the signal length and
    :meth:`inverse` reuses it (overridable via the ``length`` keyword). Used by
    FFT-mode FreqRISE so its explanation carries a usable transform.

    Parameters
    ----------
    n : int, optional
        Signal length for the inverse. If ``None`` it is captured on the first
        :meth:`forward` call.
    """

    def __init__(self, n: int | None = None) -> None:
        self.n = n

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """
        Apply the one-sided real Fourier transform.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, ..., T)``. Records ``T`` as the inverse length.
        **kwargs : object
            Extra keyword arguments forwarded to :func:`torch.fft.rfft`.

        Returns
        -------
        torch.Tensor
            Complex coefficients of shape ``(B, -1, T // 2 + 1)``.
        """
        if x.dim() <= 1:
            raise ValueError(
                f"Input signals must have dimension 2 or higher, got {x.dim()}."
            )

        batch_size = x.shape[0]
        timeseries_len = x.shape[-1]
        self.n = timeseries_len
        x = x.reshape(-1, timeseries_len)

        frequencies = torch.fft.rfft(x, **kwargs)

        return frequencies.reshape(batch_size, -1, frequencies.shape[-1])

    def inverse(
        self, x: torch.Tensor, length: int | None = None, **kwargs: object
    ) -> torch.Tensor:
        """
        Apply the inverse one-sided real Fourier transform.

        Parameters
        ----------
        x : torch.Tensor
            One-sided coefficients of shape ``(B, ..., T // 2 + 1)``.
        length : int, optional
            Output signal length. Defaults to the length recorded by
            :meth:`forward` (``self.n``).
        **kwargs : object
            Extra keyword arguments forwarded to :func:`torch.fft.irfft`.

        Returns
        -------
        torch.Tensor
            Real signals of shape ``(B, -1, length)``.
        """
        if x.dim() <= 1:
            raise ValueError(
                f"Input frequencies must have dimension 2 or higher, got {x.dim()}."
            )

        n = length if length is not None else self.n
        batch_size = x.shape[0]
        num_frequencies = x.shape[-1]
        x = x.reshape(-1, num_frequencies)

        signals = torch.fft.irfft(x, n=n, **kwargs)

        return signals.reshape(batch_size, -1, signals.shape[-1])


class STFTransform(DomainTransform):
    """
    Short-time Fourier transform and its inverse.

    Wraps :func:`torch.stft` / :func:`torch.istft`.

    Parameters
    ----------
    n_fft : int
        Size of the Fourier transform.
    win_length : int
        Size of the window frame and STFT filter.
    hop_length : int, optional
        Distance between neighbouring sliding window frames. Defaults to
        ``floor(win_length / 2)``.
    window : str
        Window function name (see :func:`_get_window`). Default ``"hann"``.
    """

    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int | None = None,
        window: str = "hann",
    ) -> None:
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length if hop_length is not None else int(win_length // 2)
        self.window = _get_window(window, win_length)
        # Recorded on forward() so inverse() can reconstruct the exact original
        # length (istft otherwise returns a frame-derived length that may differ).
        self.signal_length: int | None = None

    def forward(
        self,
        x: torch.Tensor,
        return_complex: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        """
        Apply the short-time Fourier transform.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, ..., T)`` where ``B`` is the batch dimension.
            Higher-dimensional inputs are flattened so every series is
            transformed independently.
        return_complex : bool
            Whether to return a complex spectrogram. Default ``True``.
        **kwargs : object
            Extra keyword arguments forwarded to :func:`torch.stft`.

        Returns
        -------
        torch.Tensor
            Spectrograms of shape ``(B, -1, n_freq, n_time)``.
        """
        if x.dim() <= 1:
            raise ValueError(
                f"Input signals must have dimension 2 or higher, got {x.dim()}."
            )

        batch_size = x.shape[0]
        timeseries_len = x.shape[-1]
        self.signal_length = timeseries_len
        x = x.reshape(-1, timeseries_len)

        spectrograms = torch.stft(
            x,
            return_complex=return_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            **kwargs,
        )

        return spectrograms.reshape(
            batch_size, -1, spectrograms.shape[-2], spectrograms.shape[-1]
        )

    def inverse(
        self, x: torch.Tensor, length: int | None = None, **kwargs: object
    ) -> torch.Tensor:
        """
        Apply the inverse short-time Fourier transform.

        Parameters
        ----------
        x : torch.Tensor
            Spectrograms of shape ``(B, ..., n_freq, n_time)``.
            Higher-dimensional inputs are flattened so every spectrogram is
            transformed independently.
        length : int, optional
            Output signal length. Defaults to the length recorded by
            :meth:`forward` (``self.signal_length``); ``istft`` otherwise returns
            a frame-derived length that may not equal the original.
        **kwargs : object
            Extra keyword arguments forwarded to :func:`torch.istft`.

        Returns
        -------
        torch.Tensor
            Reconstructed signals of shape ``(B, -1, T)``.
        """
        if x.dim() <= 2:
            raise ValueError(
                f"Input spectrograms must have dimension 3 or higher, got {x.dim()}."
            )

        batch_size = x.shape[0]
        num_frequencies = x.shape[-2]
        num_timesteps = x.shape[-1]
        x = x.reshape(-1, num_frequencies, num_timesteps)

        signals = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            length=length if length is not None else self.signal_length,
            **kwargs,
        )

        return signals.reshape(batch_size, -1, signals.shape[-1])


def get_transform(transform_config: dict | None) -> DomainTransform | None:
    """
    Build a :class:`DomainTransform` from a config dict.

    Parameters
    ----------
    transform_config : dict or None
        Mapping with a ``"name"`` key (``"fft"`` or ``"stft"``) and a
        ``"params"`` mapping forwarded to the transform constructor. ``None``
        yields ``None`` (time-domain, no transform).

    Returns
    -------
    DomainTransform or None
        The constructed transform, or ``None`` when *transform_config* is
        ``None``.

    Raises
    ------
    ValueError
        If ``transform_config["name"]`` is not a supported transform name.
    """
    if transform_config is None:
        return None

    name = transform_config["name"]
    params = transform_config.get("params", {})
    if name == "fft":
        return FourierTransform(**params)
    if name == "rfft":
        return RFFTransform(**params)
    if name == "stft":
        return STFTransform(**params)
    raise ValueError(f"Unknown transform name: {name!r}.")


def resolve_transform(
    transform: DomainTransform | dict | None,
) -> DomainTransform | None:
    """
    Normalise a transform argument to a :class:`DomainTransform` or ``None``.

    Accepts an already-built transform (returned as-is), a config dict (built via
    :func:`get_transform`), or ``None``. Lets explainer constructors take either a
    Python object (package use) or a YAML config mapping (runner use).

    Parameters
    ----------
    transform : DomainTransform or dict or None
        The transform, a ``get_transform`` config dict, or ``None``.

    Returns
    -------
    DomainTransform or None
        The resolved transform.

    Raises
    ------
    TypeError
        If *transform* is not a ``DomainTransform``, dict, or ``None``.
    """
    if transform is None or isinstance(transform, DomainTransform):
        return transform
    if isinstance(transform, dict):
        return get_transform(transform)
    raise TypeError(
        f"transform must be a DomainTransform, config dict, or None, got "
        f"{type(transform).__name__}."
    )
