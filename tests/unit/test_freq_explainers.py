"""
Unit tests for the Phase 2 frequency / time-frequency explainers.

Covers FreqRISE, the Frequency/TimeFrequency explanation-space wrappers, and the
random freq/TF baselines: output shapes in the declared domain, the realized
``explanation_domain`` (and its invariant), capability-set declarations, the
wrapper domain-override caveat, and required-transform guards.
"""

import numpy as np
import pytest
import torch
from torch import nn

from xai4tsc.utils import FourierTransform, STFTransform
from xai4tsc.xai import Domain
from xai4tsc.xai._types import Explanation
from xai4tsc.xai.explain import EXPLAINERS, _get_explanation, build_explainer
from xai4tsc.xai.explanation_domains import (
    FrequencyExplainer,
    TimeFrequencyExplainer,
)
from xai4tsc.xai.freqrise import FreqRISEExplainer
from xai4tsc.xai.random_baseline import (
    RandomFrequencyExplainer,
    RandomTimeFreqExplainer,
)


class _TinyNet(nn.Module):
    """Minimal logit model over flattened ``(C, T)`` input, with ``predict``."""

    def __init__(self, channels: int, timesteps: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(channels * timesteps, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))

    def predict(self, x):
        with torch.no_grad():
            logits = self(torch.as_tensor(np.asarray(x), dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs.argmax(1), probs


def _make_exp(data: np.ndarray) -> Explanation:
    n = data.shape[0]
    return Explanation(
        explainer="x",
        explanation_type="feature_attribution",
        exp_values=None,
        data=data,
        labels=np.zeros(n, dtype=np.int64),
        encoder=None,
        indices=np.arange(n),
        meta=None,
    )


def _stft_cfg():
    return {"name": "stft", "params": {"n_fft": 16, "win_length": 16, "hop_length": 4}}


# ── registration ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_freq_explainers_registered():
    for key in (
        "freqrise",
        "frequency",
        "timefrequency",
        "random_frequency",
        "random_timefrequency",
    ):
        assert key in EXPLAINERS


# ── capability sets are declared at class level (for static config checks) ─────


@pytest.mark.unit
def test_class_level_capability_sets():
    # config_sanity_check reads these statically, before instantiation.
    assert FrequencyExplainer.explanation_domains == {Domain.FREQUENCY}
    assert TimeFrequencyExplainer.explanation_domains == {Domain.TIME_FREQUENCY}
    assert RandomFrequencyExplainer.explanation_domains == {Domain.FREQUENCY}
    assert RandomTimeFreqExplainer.explanation_domains == {Domain.TIME_FREQUENCY}
    assert FreqRISEExplainer.explanation_domains == {
        Domain.FREQUENCY,
        Domain.TIME_FREQUENCY,
    }


# ── random baselines ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_random_frequency_shape_and_determinism():
    data = np.random.default_rng(0).standard_normal((3, 1, 32)).astype(np.float32)
    exp = _make_exp(data)
    expected = FourierTransform().forward(torch.as_tensor(data)).shape

    a = RandomFrequencyExplainer(seed=1).explain(None, exp, "cpu", None)
    b = RandomFrequencyExplainer(seed=1).explain(None, exp, "cpu", None)
    assert a.shape == tuple(expected)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)  # same seed → identical


@pytest.mark.unit
def test_random_timefrequency_requires_transform():
    with pytest.raises(ValueError, match="requires a time-frequency transform"):
        RandomTimeFreqExplainer()


@pytest.mark.unit
def test_random_timefrequency_shape():
    data = np.random.default_rng(0).standard_normal((2, 1, 64)).astype(np.float32)
    exp = _make_exp(data)
    transform = STFTransform(n_fft=16, win_length=16, hop_length=4)
    out = RandomTimeFreqExplainer(transform=transform).explain(None, exp, "cpu", None)
    assert out.shape == tuple(transform.forward(torch.as_tensor(data)).shape)
    assert out.ndim == 4


# ── explanation-space wrappers ────────────────────────────────────────────────


@pytest.mark.unit
def test_frequency_wrapper_overrides_domain_and_applicability():
    # Wrapper must NOT inherit the base's {TIME}/{AGNOSTIC}; it changes the domain.
    from xai4tsc.xai._types import DataType

    w = build_explainer("frequency", {"base": "integrated_gradients"})
    assert isinstance(w, FrequencyExplainer)
    assert w.explanation_domains == {Domain.FREQUENCY}
    assert w.data_applicability == {DataType.TIME_SERIES}


@pytest.mark.unit
def test_frequency_wrapper_shape_is_fft_of_relevance():
    data = np.random.default_rng(2).standard_normal((4, 1, 32)).astype(np.float32)
    model = _TinyNet(1, 32, 3).eval()
    exp = _get_explanation(
        method="frequency",
        model=model,
        data=data,
        labels=np.array([0, 1, 2, 0], dtype=np.int64),
        encoder=None,
        params={
            "base": {"method": "integrated_gradients", "multiply_by_inputs": False}
        },
        targets="label",
        device="cpu",
        orig_indices=np.arange(4),
    )
    assert exp.explanation_domain is Domain.FREQUENCY
    assert exp.exp_values.shape == data.shape  # full FFT keeps T
    assert np.iscomplexobj(exp.exp_values)


@pytest.mark.unit
def test_timefrequency_wrapper_requires_transform():
    with pytest.raises(ValueError, match="requires a time-frequency transform"):
        TimeFrequencyExplainer(base="guided_backpropagation")


@pytest.mark.unit
def test_timefrequency_wrapper_domain_and_shape():
    data = np.random.default_rng(3).standard_normal((2, 1, 64)).astype(np.float32)
    model = _TinyNet(1, 64, 2).eval()
    exp = _get_explanation(
        method="timefrequency",
        model=model,
        data=data,
        labels=np.array([0, 1], dtype=np.int64),
        encoder=None,
        params={"base": "guided_backpropagation", "transform": _stft_cfg()},
        targets="label",
        device="cpu",
        orig_indices=np.arange(2),
    )
    assert exp.explanation_domain is Domain.TIME_FREQUENCY
    assert exp.exp_values.ndim == 4


# ── FreqRISE ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_freqrise_invalid_domain_raises():
    with pytest.raises(ValueError, match="domain must be"):
        FreqRISEExplainer(domain="wavelet")


@pytest.mark.unit
def test_freqrise_stft_requires_transform():
    with pytest.raises(ValueError, match="requires a time-frequency transform"):
        FreqRISEExplainer(domain="stft")


@pytest.mark.unit
def test_freqrise_num_cells_too_large_raises():
    # num_cells // 2 must be smaller than the smallest STFT grid dimension.
    data = np.random.default_rng(6).standard_normal((1, 1, 64)).astype(np.float32)
    exp = _make_exp(data)
    model = _TinyNet(1, 64, 2).eval()
    explainer = FreqRISEExplainer(
        domain="stft",
        transform=STFTransform(n_fft=16, win_length=16, hop_length=4),
        num_cells=50,
        num_batches=1,
        batch_size=1,
    )
    with pytest.raises(ValueError, match="smallest transform dimension"):
        explainer.explain(model, exp, "cpu", [0])


@pytest.mark.unit
def test_freqrise_fft_shape_and_domain():
    data = np.random.default_rng(4).standard_normal((3, 1, 32)).astype(np.float32)
    model = _TinyNet(1, 32, 3).eval()
    exp = _get_explanation(
        method="freqrise",
        model=model,
        data=data,
        labels=np.array([0, 1, 2], dtype=np.int64),
        encoder=None,
        params={
            "domain": "fft",
            "num_batches": 3,
            "batch_size": 4,
            "num_cells": 8,
            "seed": 0,
        },
        targets="label",
        device="cpu",
        orig_indices=np.arange(3),
    )
    assert exp.explanation_domain is Domain.FREQUENCY
    # one-sided rfft → T // 2 + 1 frequency bins, single channel.
    assert exp.exp_values.shape == (3, 1, 32 // 2 + 1)
    assert np.isfinite(exp.exp_values).all()


@pytest.mark.unit
def test_freqrise_stft_shape_domain_and_seed():
    data = np.random.default_rng(5).standard_normal((2, 1, 64)).astype(np.float32)
    model = _TinyNet(1, 64, 2).eval()
    params = {
        "domain": "stft",
        "num_batches": 3,
        "batch_size": 4,
        "num_cells": 8,
        "seed": 0,
        "transform": _stft_cfg(),
    }
    common = dict(
        method="freqrise",
        model=model,
        data=data,
        labels=np.array([0, 1], dtype=np.int64),
        encoder=None,
        targets="label",
        device="cpu",
        orig_indices=np.arange(2),
    )
    exp_a = _get_explanation(params=dict(params), **common)
    exp_b = _get_explanation(params=dict(params), **common)
    assert exp_a.explanation_domain is Domain.TIME_FREQUENCY
    assert exp_a.exp_values.ndim == 4
    assert np.isfinite(exp_a.exp_values).all()
    np.testing.assert_allclose(
        exp_a.exp_values, exp_b.exp_values
    )  # seed → reproducible


class _FFTBinModel(nn.Module):
    """
    Class-1 logit = signal power at a single rfft bin ``k0``; class-0 = 0.

    A model that provably depends on exactly one frequency, so a correct
    FreqRISE must place its relevance peak at bin ``k0``.
    """

    def __init__(self, k0: int, scale: float = 8.0) -> None:
        super().__init__()
        self.k0, self.scale = k0, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        power = torch.fft.rfft(x, dim=-1)[..., self.k0].abs().pow(2).sum(1)
        return torch.stack([torch.zeros_like(power), self.scale * power], dim=1)

    def predict(self, x):
        with torch.no_grad():
            logits = self(torch.as_tensor(np.asarray(x), dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs.argmax(1), probs


@pytest.mark.unit
@pytest.mark.parametrize("k0", [5, 12, 20])
def test_freqrise_fft_localizes_to_discriminative_frequency(k0):
    # Behavioral correctness: a model that only reads rfft bin k0 must make
    # FreqRISE's relevance peak at bin k0.
    timesteps = 64
    t = np.arange(timesteps)
    data = np.sin(2 * np.pi * k0 * t / timesteps).astype(np.float32)[None, None, :]
    explainer = FreqRISEExplainer(
        domain="fft", num_batches=200, batch_size=16, num_cells=16, seed=0
    )
    relevance = explainer.explain(_FFTBinModel(k0), _make_exp(data), "cpu", [1])
    rel = relevance[0, 0]  # (T // 2 + 1,), min-max scaled to [0, 1]
    bins = np.arange(len(rel))
    assert abs(int(rel.argmax()) - k0) <= 1  # peak at the right frequency
    assert rel[k0] >= 0.8  # k0 is at (or next to) the dominant bin
    assert rel[np.abs(bins - k0) >= 5].max() < 0.5  # bin
