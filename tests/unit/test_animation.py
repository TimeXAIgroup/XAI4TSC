"""
Unit tests for perturbation animations (GIF) — headless (Agg), tmp_path only.

Covers opt-in frame collection in _perturbation_curve, the animation classes'
GIF output, and the metric-level animate() methods for both domains.
"""

import matplotlib
import numpy as np
import pytest
import torch
from torch import nn

from xai4tsc.evaluation.evaluate import METRICS
from xai4tsc.evaluation.frequency_evaluate import _perturbation_curve, _zero_replacement
from xai4tsc.utils import RFFTransform, STFTransform
from xai4tsc.utils.animation import (
    FrequencyPerturbationAnimation,
    TimeFrequencyPerturbationAnimation,
)


@pytest.fixture(autouse=True)
def _agg_backend():
    matplotlib.use("Agg", force=True)
    yield
    from matplotlib import pyplot as plt

    plt.close("all")


class _TinyNet(nn.Module):
    def __init__(self, channels, timesteps, n_classes):
        super().__init__()
        self.fc = nn.Linear(channels * timesteps, n_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))

    def predict(self, x):
        with torch.no_grad():
            p = torch.softmax(
                self(torch.as_tensor(np.asarray(x), dtype=torch.float32)), 1
            ).numpy()
        return p.argmax(1), p


# ── frame collection ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_perturbation_curve_collect_frames():
    data = np.random.default_rng(0).standard_normal((3, 1, 32)).astype(np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    model = _TinyNet(1, 32, 2).eval()
    transform = RFFTransform()
    a = np.abs(transform.forward(torch.as_tensor(data)).numpy())

    result, frames = _perturbation_curve(
        model,
        data,
        labels,
        a,
        transform,
        features_in_step=3,
        perturb_func=_zero_replacement(0.0),
        return_aoc=False,
        collect_frames=True,
        frame_sample=1,
    )
    assert isinstance(result, list)  # the curve is still returned
    n_freq = a.shape[-1]
    assert frames["time"].shape[1:] == (1, 32)
    assert frames["coeffs"].shape[1:] == (1, n_freq)
    assert frames["relevance"].shape == (1, n_freq)
    # one prediction per step (n_steps + 1) and finite.
    assert frames["prediction"].ndim == 1
    assert np.isfinite(frames["prediction"]).all()


# ── animation classes write GIFs ──────────────────────────────────────────────


@pytest.mark.unit
def test_frequency_animation_writes_gif(tmp_path):
    n_steps, n_freq = 5, 9
    frames = {
        "time": np.random.default_rng(1).standard_normal((n_steps, 1, 16)),
        "coeffs": np.random.default_rng(2).standard_normal((n_steps, 1, n_freq)),
        "relevance": np.abs(np.random.default_rng(3).standard_normal((1, n_freq))),
        "prediction": np.linspace(1.0, 0.0, n_steps),
    }
    out = FrequencyPerturbationAnimation(frames).save(tmp_path / "a.gif", fps=5)
    assert out.suffix == ".gif" and out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_timefrequency_animation_writes_gif_and_enforces_suffix(tmp_path):
    n_steps = 4
    frames = {
        "time": np.random.default_rng(4).standard_normal((n_steps, 2, 32)),
        "coeffs": np.abs(np.random.default_rng(5).standard_normal((n_steps, 2, 9, 7))),
        "relevance": np.abs(np.random.default_rng(6).standard_normal((2, 9, 7))),
        "prediction": np.linspace(1.0, 0.2, n_steps),
    }
    # ".mp4" requested but only GIF is supported → suffix is forced to .gif.
    out = TimeFrequencyPerturbationAnimation(frames).save(tmp_path / "a.mp4", fps=5)
    assert out.suffix == ".gif" and out.exists()


# ── metric.animate() ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_frequency_metric_animate(tmp_path):
    data = np.random.default_rng(7).standard_normal((2, 1, 32)).astype(np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    model = _TinyNet(1, 32, 2).eval()
    metric = METRICS["Frequency Perturbation"](
        transform=RFFTransform(), features_in_step=3
    )
    a = np.abs(RFFTransform().forward(torch.as_tensor(data)).numpy())
    out = metric.animate(model, data, labels, a, tmp_path / "f.gif", sample=0)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_timefrequency_metric_animate(tmp_path):
    data = np.random.default_rng(8).standard_normal((2, 1, 64)).astype(np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    model = _TinyNet(1, 64, 2).eval()
    transform = STFTransform(n_fft=16, win_length=16, hop_length=4)
    metric = METRICS["Time-Frequency Perturbation Gaussian"](
        transform=transform, features_in_step=8
    )
    a = np.abs(transform.forward(torch.as_tensor(data)).numpy())
    out = metric.animate(model, data, labels, a, tmp_path / "tf.gif")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_animate_without_transform_raises(tmp_path):
    model = _TinyNet(1, 32, 2).eval()
    data = np.zeros((1, 1, 32), dtype=np.float32)
    metric = METRICS["Frequency Perturbation"]()  # no transform
    with pytest.raises(ValueError, match="requires a transform to animate"):
        metric.animate(model, data, np.array([0]), np.abs(data), tmp_path / "x.gif")
