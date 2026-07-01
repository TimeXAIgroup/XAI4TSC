"""
Perturbation-process animations (GIF) for the frequency / time-frequency metrics.

Each animation shows a single sample's perturbation trajectory over the metric's
flipping steps: the reconstructed time signal, the perturbed frequency /
time-frequency representation (with the static relevance), and the prediction
curve building up. Frames are produced by the perturbation metrics' ``animate``
method (opt-in frame collection); output is always a GIF via Matplotlib's
built-in ``PillowWriter`` (no ffmpeg dependency).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

logger = logging.getLogger(__name__)


class _PerturbationAnimation:
    """
    Base animation: build per-frame panels for one sample and save as a GIF.

    Parameters
    ----------
    frames : dict
        Per-step frame data with keys ``time`` ``(n_frames, C, T)``, ``coeffs``
        ``(n_frames, C, *F)``, ``relevance`` ``(C, *F)`` and ``prediction``
        ``(n_frames,)``.
    step : int
        Stride over frames (plot every *step*-th step).
    max_frames : int
        Cap on the number of frames rendered.
    """

    def __init__(self, frames: dict, step: int = 1, max_frames: int = 300) -> None:
        self.time = np.asarray(frames["time"])
        self.coeffs = np.asarray(frames["coeffs"])
        self.relevance = np.asarray(frames["relevance"])
        self.prediction = np.asarray(frames["prediction"])
        self.n_frames = self.time.shape[0]
        self.n_channels = self.time.shape[1]
        self.step = max(1, step)
        self.stop = min(self.n_frames, max_frames)
        self.fraction = np.linspace(0, 100, self.n_frames)
        self.fig: plt.Figure | None = None

    def _setup(self) -> None:
        """Build the figure and panels; populate updatable artists."""
        raise NotImplementedError

    def _update(self, idx: int) -> None:
        """Update the artists for frame *idx*."""
        raise NotImplementedError

    def _add_prediction_panel(self, ax: plt.Axes) -> None:
        (self.pred_line,) = ax.plot([], [], linewidth=1.5, color="black")
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Prediction")
        ax.set_xlabel("Perturbed coefficients (%)")
        ax.set_ylabel("Prediction")

    def save(self, save_path: Path | str, fps: int = 20) -> Path:
        """
        Render the animation and write it as a GIF.

        Parameters
        ----------
        save_path : Path or str
            Output path; a ``.gif`` suffix is enforced.
        fps : int
            Frames per second.

        Returns
        -------
        Path
            The written GIF path.
        """
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        if save_path.suffix.lower() != ".gif":
            save_path = save_path.with_suffix(".gif")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup()
        animation = FuncAnimation(
            self.fig, self._update, frames=range(0, self.stop, self.step)
        )
        animation.save(str(save_path), writer=PillowWriter(fps=fps))
        plt.close(self.fig)
        logger.info("Saved perturbation animation to:\n%s", save_path)
        return save_path


class FrequencyPerturbationAnimation(_PerturbationAnimation):
    """Animate a frequency-domain perturbation: spectrum + time signal + prediction."""

    def _setup(self) -> None:
        self.fig = plt.figure(figsize=(16, 3 * self.n_channels), layout="constrained")
        gs = self.fig.add_gridspec(self.n_channels, 3)
        n_freq = self.coeffs.shape[-1]
        x = np.arange(n_freq)
        self.freq_lines: list = []
        self.freq_scatters: list = []
        self.time_lines: list = []

        for c in range(self.n_channels):
            ax_f = self.fig.add_subplot(gs[c, 0])
            mag0 = np.abs(self.coeffs[0, c])
            (line_f,) = ax_f.plot(x, mag0, color="black", linewidth=0.8)
            scatter = ax_f.scatter(
                x, mag0, c=np.abs(self.relevance[c]), cmap="bwr", s=10
            )
            ax_f.set_ylim(0, float(np.abs(self.coeffs[:, c]).max()) * 1.05 + 1e-9)
            ax_f.set_ylabel(f"Channel {c + 1}\n|coefficient|")
            if c == 0:
                ax_f.set_title("Frequency")
            if c == self.n_channels - 1:
                ax_f.set_xlabel("Frequency")
            self.freq_lines.append(line_f)
            self.freq_scatters.append(scatter)

            ax_t = self.fig.add_subplot(gs[c, 1])
            (line_t,) = ax_t.plot(self.time[0, c], linewidth=0.8)
            ax_t.set_ylim(float(self.time[:, c].min()), float(self.time[:, c].max()))
            if c == 0:
                ax_t.set_title("Time series")
            if c == self.n_channels - 1:
                ax_t.set_xlabel("Time steps")
            self.time_lines.append(line_t)

        self._add_prediction_panel(self.fig.add_subplot(gs[:, 2]))

    def _update(self, idx: int) -> None:
        x = np.arange(self.coeffs.shape[-1])
        for c in range(self.n_channels):
            mag = np.abs(self.coeffs[idx, c])
            self.freq_lines[c].set_ydata(mag)
            self.freq_scatters[c].set_offsets(np.column_stack([x, mag]))
            self.time_lines[c].set_ydata(self.time[idx, c])
        self.pred_line.set_data(self.fraction[: idx + 1], self.prediction[: idx + 1])


class TimeFrequencyPerturbationAnimation(_PerturbationAnimation):
    """Animate a TF perturbation: relevance + spectrogram + time signal + prediction."""

    def _setup(self) -> None:
        self.fig = plt.figure(figsize=(18, 3 * self.n_channels), layout="constrained")
        gs = self.fig.add_gridspec(self.n_channels, 4)
        self.spec_images: list = []
        self.time_lines = []

        for c in range(self.n_channels):
            ax_r = self.fig.add_subplot(gs[c, 0])
            ax_r.imshow(
                np.abs(self.relevance[c]), origin="lower", aspect="auto", cmap="Reds"
            )
            ax_r.set_ylabel(f"Channel {c + 1}\nFrequency")
            if c == 0:
                ax_r.set_title("Relevance")

            ax_s = self.fig.add_subplot(gs[c, 1])
            vmax = float(np.abs(self.coeffs[:, c]).max()) + 1e-9
            image = ax_s.imshow(
                np.abs(self.coeffs[0, c]),
                origin="lower",
                aspect="auto",
                cmap="plasma",
                vmin=0,
                vmax=vmax,
            )
            self.spec_images.append(image)
            if c == 0:
                ax_s.set_title("Spectrogram")

            ax_t = self.fig.add_subplot(gs[c, 2])
            (line_t,) = ax_t.plot(self.time[0, c], linewidth=0.8)
            ax_t.set_ylim(float(self.time[:, c].min()), float(self.time[:, c].max()))
            if c == 0:
                ax_t.set_title("Time series")
            if c == self.n_channels - 1:
                ax_t.set_xlabel("Time steps")
            self.time_lines.append(line_t)

        self._add_prediction_panel(self.fig.add_subplot(gs[:, 3]))

    def _update(self, idx: int) -> None:
        for c in range(self.n_channels):
            self.spec_images[c].set_array(np.abs(self.coeffs[idx, c]))
            self.time_lines[c].set_ydata(self.time[idx, c])
        self.pred_line.set_data(self.fraction[: idx + 1], self.prediction[: idx + 1])
