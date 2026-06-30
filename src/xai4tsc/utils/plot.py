"""Relevance visualisation: ``plot_relevance()`` and ``add_relevance()``."""

import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from .utils import rescale_array

logger = logging.getLogger(__name__)


def plot_relevance(
    signal: np.ndarray,
    relevance: np.ndarray | None = None,
    rel_type: str = "bubbles",
    rel_type_kwargs: dict | None = None,
    linewidth: float = 1.4,
    threshold: float = 0.0,
    cmap_boost: float = 0.0,
    title: str | None = None,
    colorbar: bool = False,
    graph_only: bool = False,
    save_path: Path | None = None,
    xlabel: str = "Time [ms]",
) -> Path | None:
    """
    Plot a 1-D signal with an overlaid relevance map.

    Works for any 1-D domain: the time domain (signal vs time) and, via
    :func:`plot_relevance_f`, the frequency domain (magnitude spectrum vs
    frequency).

    Parameters
    ----------
    signal : np.ndarray
        Signal data of shape ``(B, C, T)``, ``(C, T)``, or ``(T,)``.
    relevance : np.ndarray, optional
        Relevance values matching the shape of *signal*.  Defaults to zeros.
    rel_type : str
        Visualisation style: ``"bubbles"`` (default), ``"background"``,
        ``"intensity"``, ``"graph"``, ``"bar"``, or ``None``.
    rel_type_kwargs : dict
        Extra keyword arguments forwarded to :func:`add_relevance`.
    linewidth : float
        Line width for the signal plot.
    threshold : float
        Relevance values below this absolute magnitude are suppressed.
    cmap_boost : float
        Value added (with sign) to non-zero relevances before colourmap mapping.
    title : str, optional
        Figure title.
    colorbar : bool
        Whether to add a colourbar to the figure.
    graph_only : bool
        If ``True``, hide all axes decorations.
    save_path : Path, optional
        File path to save the figure.  If ``None``, the figure is shown
        interactively.
    xlabel : str
        Label for the x-axis (``"Time [ms]"`` by default; ``"Frequency"`` for
        frequency-domain plots).

    Returns
    -------
    Path or None
        Path to the last saved file, or ``None`` if not saved.
    """
    if rel_type_kwargs is None:
        rel_type_kwargs = {}
    if relevance is None:
        relevance = np.zeros_like(signal)

    # input dimension checks
    if len(signal.shape) != len(relevance.shape):
        logger.error(
            "Signal %s and relevance %s dimension mismatch",
            signal.shape,
            relevance.shape,
        )
        return None
    if len(signal.shape) == 2:
        logger.warning("Input is missing one dimension, assuming batch size = 1.")
        signal = signal[np.newaxis, ...]
        while len(relevance.shape) < 3:
            relevance = relevance[np.newaxis, ...]
    elif len(signal.shape) == 1:
        logger.warning(
            "Input is missing two dimensions, assuming batch size = channels = 1."
        )
        signal = signal[np.newaxis, np.newaxis, ...]
        while len(relevance.shape) < 3:
            relevance = relevance[np.newaxis, ...]
    # setup colormap
    if rel_type not in ["intensity"]:
        if np.min(relevance) < 0:
            cmap = "bwr"
        else:
            cmap = LinearSegmentedColormap.from_list(
                "WhiteToRed", ["#ffffff", "#ff0000"], N=256
            )
    elif np.min(relevance) < 0:
        cmap = LinearSegmentedColormap.from_list(
            "BlueToBlackToRed", ["#0000ff", "#000000", "#ff0000"], N=256
        )
    else:
        cmap = LinearSegmentedColormap.from_list(
            "BlackToRed", ["#000000", "#ff0000"], N=256
        )
    # check whether relevances are normalized
    if relevance.max() > 1 or relevance.min() < 0:
        relevance = rescale_array(relevance, 0, 1)

    # expects signal_time to be of shape (batch X channel X timepoints) = (BxCxT)
    batch_size = signal.shape[0]
    channels = signal.shape[1]
    timesteps = signal.shape[2]

    if save_path is not None:
        # if multiple plots are generated, create a directory
        if batch_size > 1:
            out_dir = save_path.with_suffix("")
            out_dir.mkdir(parents=True, exist_ok=True)
            save_name = save_path.name
            save_path = out_dir / save_name

        # setup of save path
        save_name = save_path.stem
        save_filetype = save_path.suffix
        out_dir = save_path.parent

    # plot each sample from the batch
    for s_idx, (sample, sample_relevance) in enumerate(
        zip(signal, relevance, strict=True)
    ):
        if save_path is not None:
            save_path = out_dir / (save_name + f"{s_idx}" + save_filetype)
        nrows, ncols = channels, 1
        figsize = (10, 2.5)
        fig, axes = plt.subplots(
            nrows, ncols, sharex=True, figsize=figsize, dpi=300, layout="constrained"
        )
        # fig.text(0.00, 0.5, "Signal Amplitude", va="center", rotation="vertical")
        if not graph_only:
            fig.supylabel("Signal Amplitude")
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        if title is not None:
            fig.suptitle(title)

        # plot one row for each channel
        for channel in range(channels):
            cur_signal = sample[channel]
            cur_relevance = sample_relevance[channel]
            axes[channel].margins(x=0)
            # plot signal
            if rel_type not in ["intensity"]:
                t = np.arange(timesteps)  # linspace(0, timesteps, timesteps)
                axes[channel].plot(
                    t, cur_signal, linewidth=linewidth, color="black", label="signal"
                )
            # plot relevance
            add_relevance(
                axes[channel],
                cur_signal,
                cur_relevance,
                rel_type,
                threshold,
                cmap,
                cmap_boost,
                **rel_type_kwargs,
            )
            if channel == channels - 1:
                axes[channel].set_xlabel(xlabel)
            # ax[channel].set_ylabel("Relative Amplitude")
            if graph_only:
                for ax in axes.flat:
                    ax.set_axis_off()
        if rel_type in ["graph"]:
            plt.legend()
        if colorbar and rel_type not in ["graph"]:
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])  # required for older Matplotlib versions
            fig.colorbar(sm, ax=axes, label="Relevance", use_gridspec=False)
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
            plt.close()
    return save_path


def add_relevance(
    ax: Axes,
    signal: np.ndarray,
    relevance: np.ndarray,
    rel_type: str = "bubble",
    threshold: float = 0.0,
    cmap: str = "bwr",
    cmap_boost: float = 0.0,
    **kwargs: object,
) -> Axes:
    """
    Overlay relevance onto a single Matplotlib axes.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    signal : np.ndarray
        1-D time series for this channel.
    relevance : np.ndarray
        1-D relevance values matching *signal*.
    rel_type : str
        Visualisation style: ``"bubbles"`` (default), ``"background"``,
        ``"intensity"``, ``"graph"``, ``"bar"``, or ``None``.
    threshold : float
        Relevance values below this absolute magnitude are suppressed.
    cmap : str or Colormap
        Matplotlib colourmap used for relevance colouring.
    cmap_boost : float
        Value added (with sign) to non-zero relevances before colour mapping.
    **kwargs
        Style overrides forwarded to the chosen visualisation type
        (e.g. ``bubble_size``, ``bar_height``, ``linewidth``).

    Returns
    -------
    Axes
        The modified axes.
    """
    if rel_type == "background":
        # relevance threshold & boost
        rel = np.where(
            np.abs(relevance) >= threshold,
            relevance + np.sign(relevance) * cmap_boost,
            0,
        )
        y0, y1 = ax.get_ylim()
        rel_plt = ax.imshow(
            np.expand_dims(rel, 0),
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            extent=[
                # *(ax.get_xlim()),
                -0.5,
                signal.shape[-1] - 0.5,
                *(ax.get_ylim()),
            ],
            alpha=1,
            origin="lower",
            zorder=0,
        )
    elif rel_type == "intensity":
        # apply thresholding and boosting
        relevance = np.where(
            np.abs(relevance) >= threshold,
            relevance + np.sign(relevance) * cmap_boost,
            0,
        )
        # # generate points (N, 1, 2)
        # points = np.column_stack([np.arange(signal.shape[-1]), signal])
        # points = points.reshape(-1, 1, 2)
        # # generate segments (N-1, 2, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)

        t_len = signal.shape[-1]
        x = np.arange(t_len, dtype=float)

        # apply thresholding and boosting
        rel = np.where(
            np.abs(relevance) >= threshold,
            relevance + np.sign(relevance) * cmap_boost,
            0,
        )

        # midpoints y at half-steps (x=0.5, 1.5, ..., T-1.5)
        y_mid = 0.5 * (signal[:-1] + signal[1:])

        # y at (t-0.5) and (t+0.5)
        y_left = np.empty(t_len, dtype=float)
        y_right = np.empty(t_len, dtype=float)

        y_left[0] = signal[0]
        y_left[1:] = y_mid

        y_right[:-1] = y_mid
        y_right[-1] = signal[-1]

        # Build 2 segments per timestep: left half + right half
        # left half:  (t-0.5, y_left[t])  -> (t, signal[t])
        # right half: (t, signal[t])      -> (t+0.5, y_right[t])
        segments_left = np.stack(
            [np.column_stack([x - 0.5, y_left]), np.column_stack([x, signal])], axis=1
        )  # (T, 2, 2)

        segments_right = np.stack(
            [np.column_stack([x, signal]), np.column_stack([x + 0.5, y_right])], axis=1
        )  # (T, 2, 2)

        segments = np.empty(
            (
                segments_left.shape[0] + segments_right.shape[0],
                *segments_left.shape[1:],
            ),
        )  # (2T, 2, 2)
        segments[0::2] = segments_left
        segments[1::2] = segments_right
        segments[0, 0, 0] = 0
        segments[-1, 1, 0] = t_len - 1

        # Repeat relevance so both halves get the same color
        rel_for_segments = np.repeat(rel, 2)
        # draw segments + relevance per segment
        rel_plt = LineCollection(segments, cmap=cmap)
        # rel_plt.set_array(relevance[:-1])
        rel_plt.set_array(rel_for_segments)
        rel_plt.set_linewidth(kwargs.get("linewidth", 1.2))
        ax.add_collection(rel_plt)
        ax.autoscale()
    elif rel_type == "graph":
        # relevance threshold & boost
        x = np.where(
            np.abs(relevance) >= threshold,
            relevance + np.sign(relevance) * cmap_boost,
            0,
        )
        rel_plt = ax.plot(relevance, color="red", label="relevance")
    elif rel_type == "bar":
        # apply thresholding and boosting
        rel = np.where(
            np.abs(relevance) >= threshold,
            relevance + np.sign(relevance) * cmap_boost,
            0,
        )

        # get current limits
        y0, y1 = ax.get_ylim()
        bar_h = kwargs.get("bar_height", 0.05) * (y1 - y0)
        gap = kwargs.get("bar_gap", 0.02) * (y1 - y0)
        bar_y = y0 - gap - bar_h

        # extend y-limits to make room for the bar
        ax.set_ylim(bar_y - 0.01 * (y1 - y0), y1)

        # draw the bar under the plot
        rel_plt = ax.imshow(
            np.expand_dims(rel, 0),
            aspect="auto",
            interpolation="nearest",
            extent=[-0.5, signal.shape[-1] - 0.5, bar_y, bar_y + bar_h],
            cmap=cmap,
            alpha=1,
        )
    elif rel_type is None:
        return ax
    else:  # Bubbles as default
        # relevance threshold & boost
        x = np.where(np.abs(relevance) >= threshold)[0]
        y = signal[x]
        z = relevance[x] + np.sign(relevance[x]) * cmap_boost
        rel_plt = ax.scatter(
            x,
            y,
            marker="o",
            c=z,
            cmap=cmap,
            s=kwargs.get("bubble_size", 10),
            zorder=0,
            vmin=np.min(relevance),
            vmax=np.max(relevance),
        )
    return ax


def plot_relevance_f(
    signal: np.ndarray,
    relevance: np.ndarray | None = None,
    rel_type: str = "bubbles",
    save_path: Path | None = None,
    **kwargs: object,
) -> Path | None:
    """
    Plot a frequency-domain explanation by reusing :func:`plot_relevance`.

    The magnitude spectrum is drawn as the 1-D "signal" and the relevance is
    overlaid in the usual styles. Complex inputs (e.g. an FFT spectrum, or a
    complex relevance from an explanation-space wrapper) are reduced to magnitude
    so the shared 1-D machinery applies.

    Parameters
    ----------
    signal : np.ndarray
        Frequency coefficients of shape ``(B, C, F)`` (possibly complex).
    relevance : np.ndarray, optional
        Relevance values matching *signal* (possibly complex).
    rel_type : str
        Visualisation style forwarded to :func:`plot_relevance`.
    save_path : Path, optional
        File path to save the figure; shown interactively if ``None``.
    **kwargs : object
        Extra keyword arguments forwarded to :func:`plot_relevance`.

    Returns
    -------
    Path or None
        Path to the last saved file, or ``None`` if not saved.
    """
    signal_mag = np.abs(signal)
    if relevance is not None and np.iscomplexobj(relevance):
        relevance = np.abs(relevance)
    return plot_relevance(
        signal_mag,
        relevance,
        rel_type=rel_type,
        save_path=save_path,
        xlabel="Frequency",
        **kwargs,
    )


def plot_relevance_tf(
    signal: np.ndarray,
    relevance: np.ndarray,
    save_path: Path | None = None,
    title: str | None = None,
    signal_cmap: str = "viridis",
    relevance_cmap: str = "Reds",
) -> Path | None:
    """
    Plot a time-frequency explanation as side-by-side spectrogram heatmaps.

    For each sample and channel, draws two panels — the magnitude spectrogram and
    the relevance — as 2-D ``imshow`` heatmaps (time on x, frequency on y). This
    is the time-frequency counterpart of :func:`plot_relevance`; a spectrogram is
    2-D and cannot use the 1-D :func:`add_relevance` overlay styles.

    Parameters
    ----------
    signal : np.ndarray
        Spectrograms of shape ``(B, C, n_freq, n_time)`` (possibly complex).
    relevance : np.ndarray
        Relevance of shape ``(B, C, n_freq, n_time)`` (possibly complex).
    save_path : Path, optional
        File path to save the figure. With more than one sample a directory is
        created and per-sample files are written. Shown interactively if ``None``.
    title : str, optional
        Figure title.
    signal_cmap : str
        Colourmap for the spectrogram panel.
    relevance_cmap : str
        Colourmap for the relevance panel.

    Returns
    -------
    Path or None
        Path to the last saved file, or ``None`` if not saved.
    """
    signal = np.abs(np.asarray(signal))
    relevance = np.abs(np.asarray(relevance))
    if signal.ndim != 4 or relevance.ndim != 4:
        logger.error(
            "plot_relevance_tf expects 4-D (B, C, n_freq, n_time) arrays, got "
            "signal %s and relevance %s.",
            signal.shape,
            relevance.shape,
        )
        return None

    batch_size, channels = signal.shape[0], signal.shape[1]

    out_dir = save_name = save_filetype = None
    if save_path is not None:
        if batch_size > 1:
            base = save_path.with_suffix("")
            base.mkdir(parents=True, exist_ok=True)
            save_path = base / save_path.name
        save_name = save_path.stem
        save_filetype = save_path.suffix or ".png"
        out_dir = save_path.parent

    last_path: Path | None = None
    for s_idx in range(batch_size):
        fig, axes = plt.subplots(
            channels,
            2,
            figsize=(12, 3 * channels),
            dpi=300,
            squeeze=False,
            layout="constrained",
        )
        if title is not None:
            fig.suptitle(title)
        for channel in range(channels):
            for col, (data, cmap, panel_title) in enumerate(
                (
                    (signal[s_idx, channel], signal_cmap, "Spectrogram"),
                    (relevance[s_idx, channel], relevance_cmap, "Relevance"),
                )
            ):
                ax = axes[channel, col]
                im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
                ax.set_xlabel("Time")
                ax.set_ylabel(f"Channel {channel + 1}\nFrequency")
                if channel == 0:
                    ax.set_title(panel_title)
                fig.colorbar(im, ax=ax)

        if save_path is not None:
            last_path = out_dir / f"{save_name}{s_idx}{save_filetype}"
            plt.savefig(last_path, dpi=300)
            plt.close()
        else:
            plt.show()
            plt.close()
    return last_path


def plot_perturbation_curve(
    scores: np.ndarray,
    labels: np.ndarray | None = None,
    title: str | None = None,
    save_path: Path | None = None,
) -> Path | None:
    """
    Plot a perturbation (pixel-flipping) curve from per-sample metric scores.

    Draws the mean predicted probability against the fraction of coefficients
    perturbed: one line per class when *labels* are given, plus the overall mean.
    A faithful explanation makes the curve drop quickly. Pairs with the
    frequency / time-frequency perturbation metrics, which return per-sample
    curves of shape ``(n_samples, n_steps)``.

    Parameters
    ----------
    scores : np.ndarray
        Per-sample perturbation curves of shape ``(n_samples, n_steps)``.
    labels : np.ndarray, optional
        Class label per sample; when given, a mean curve is drawn per class.
    title : str, optional
        Figure title.
    save_path : Path, optional
        File path to save the figure; shown interactively if ``None``.

    Returns
    -------
    Path or None
        The saved file path, or ``None`` if not saved.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        logger.error(
            "plot_perturbation_curve expects 2-D (n_samples, n_steps) scores, got %s.",
            scores.shape,
        )
        return None

    fraction = np.linspace(0.0, 1.0, scores.shape[1])
    _, ax = plt.subplots(figsize=(8, 6), dpi=300, layout="constrained")

    if labels is not None:
        labels = np.asarray(labels)
        for cls in np.unique(labels):
            mask = labels == cls
            ax.plot(
                fraction,
                scores[mask].mean(axis=0),
                label=f"class {cls} ({int(mask.sum())} samples)",
                linewidth=1,
            )
    ax.plot(
        fraction,
        scores.mean(axis=0),
        label=f"mean ({scores.shape[0]} samples)",
        color="black",
        linewidth=2,
    )

    ax.set_xlabel("Fraction of coefficients perturbed")
    ax.set_ylabel("Mean prediction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if title is not None:
        ax.set_title(title)
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path
    plt.show()
    plt.close()
    return None
