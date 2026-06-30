"""
Unit tests for frequency / time-frequency plotting — headless (Agg), tmp_path only.

Covers plot_relevance_f (frequency, reuses plot_relevance), plot_relevance_tf
(2-D spectrogram heatmaps), and the domain dispatch in plot_exp.
"""

import matplotlib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import LabelEncoder

from xai4tsc.utils import FourierTransform, STFTransform
from xai4tsc.utils.plot import (
    plot_perturbation_curve,
    plot_relevance,
    plot_relevance_f,
    plot_relevance_tf,
)
from xai4tsc.xai import Domain
from xai4tsc.xai._types import Explanation
from xai4tsc.xai.explain import plot_exp


@pytest.fixture(autouse=True)
def _agg_backend():
    matplotlib.use("Agg", force=True)
    yield
    from matplotlib import pyplot as plt

    plt.close("all")


def _exp(domain, exp_values, data, transform):
    labels = np.zeros(data.shape[0], dtype=np.int64)
    return Explanation(
        explainer="x",
        explanation_type="feature_attribution",
        exp_values=exp_values,
        data=data,
        labels=labels,
        indices=np.arange(data.shape[0]),
        encoder=LabelEncoder().fit(labels),
        meta=None,
        explanation_domain=domain,
        transform=transform,
    )


# ── plot_relevance xlabel passthrough ─────────────────────────────────────────


@pytest.mark.unit
def test_plot_relevance_accepts_xlabel(tmp_path):
    sig = np.random.default_rng(0).standard_normal((1, 1, 20)).astype(np.float32)
    out = plot_relevance(sig, sig, save_path=tmp_path / "p.png", xlabel="Frequency")
    assert out is not None and out.exists()


# ── plot_relevance_f ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_plot_relevance_f_complex_inputs(tmp_path):
    # Complex spectrum + complex relevance are reduced to magnitude and rendered.
    rng = np.random.default_rng(1)
    spec = (
        rng.standard_normal((1, 1, 17)) + 1j * rng.standard_normal((1, 1, 17))
    ).astype(np.complex64)
    rel = (
        rng.standard_normal((1, 1, 17)) + 1j * rng.standard_normal((1, 1, 17))
    ).astype(np.complex64)
    out = plot_relevance_f(spec, rel, save_path=tmp_path / "f.png")
    assert out is not None and out.exists()


# ── plot_relevance_tf ─────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("channels", [1, 2])
def test_plot_relevance_tf_writes_file(tmp_path, channels):
    rng = np.random.default_rng(2)
    spec = rng.standard_normal((1, channels, 9, 13)).astype(np.float32)
    rel = np.abs(rng.standard_normal((1, channels, 9, 13))).astype(np.float32)
    out = plot_relevance_tf(spec, rel, save_path=tmp_path / "tf.png")
    assert out is not None and out.exists()


@pytest.mark.unit
def test_plot_relevance_tf_rejects_non_4d(tmp_path):
    bad = np.zeros((1, 1, 10), dtype=np.float32)
    assert plot_relevance_tf(bad, bad, save_path=tmp_path / "x.png") is None


# ── plot_exp domain dispatch ──────────────────────────────────────────────────


@pytest.mark.unit
def test_plot_exp_frequency_dispatch(tmp_path):
    data = np.random.default_rng(3).standard_normal((2, 1, 32)).astype(np.float32)
    exp = _exp(
        Domain.FREQUENCY,
        np.abs(np.random.default_rng(4).standard_normal((2, 1, 32))).astype(np.float32),
        data,
        FourierTransform(),
    )
    plot_exp(exp, save_path=tmp_path, visualization_type=["bubbles"])
    files = list(tmp_path.rglob("*frequency*.png"))
    assert len(files) == 2


@pytest.mark.unit
def test_plot_exp_timefrequency_dispatch(tmp_path):
    data = np.random.default_rng(5).standard_normal((2, 1, 64)).astype(np.float32)
    transform = STFTransform(n_fft=16, win_length=16, hop_length=4)
    tf_shape = transform.forward(torch.as_tensor(data)).shape
    exp = _exp(
        Domain.TIME_FREQUENCY,
        np.abs(np.random.default_rng(6).standard_normal(tf_shape)).astype(np.float32),
        data,
        transform,
    )
    plot_exp(exp, save_path=tmp_path)
    files = list(tmp_path.rglob("*timefrequency*.png"))
    assert len(files) == 2


# ── plot_perturbation_curve ───────────────────────────────────────────────────


@pytest.mark.unit
def test_plot_perturbation_curve_writes_file(tmp_path):
    scores = np.random.default_rng(8).random((6, 11))
    labels = np.array([0, 0, 1, 1, 2, 2])
    out = plot_perturbation_curve(scores, labels=labels, save_path=tmp_path / "c.png")
    assert out is not None and out.exists()


@pytest.mark.unit
def test_plot_perturbation_curve_without_labels(tmp_path):
    scores = np.random.default_rng(9).random((4, 8))
    out = plot_perturbation_curve(scores, save_path=tmp_path / "c.png")
    assert out is not None and out.exists()


@pytest.mark.unit
def test_plot_perturbation_curve_rejects_non_2d(tmp_path):
    assert plot_perturbation_curve(np.zeros((3,)), save_path=tmp_path / "x.png") is None


@pytest.mark.unit
def test_plot_exp_freq_without_transform_raises(tmp_path):
    data = np.random.default_rng(7).standard_normal((1, 1, 32)).astype(np.float32)
    exp = _exp(
        Domain.FREQUENCY,
        np.abs(data),
        data,
        None,  # no transform → cannot derive the spectrum
    )
    with pytest.raises(ValueError, match=r"without exp\.transform"):
        plot_exp(exp, save_path=tmp_path)
