"""Unit tests for src/xai4tsc/utils/fourier_transforms.py and the Domain schema."""

import numpy as np
import pytest
import torch

from xai4tsc import Domain, Explanation
from xai4tsc.utils import (
    DomainTransform,
    FourierTransform,
    STFTransform,
    get_transform,
)
from xai4tsc.xai import ExplainerBase

# ── FourierTransform ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_fourier_forward_shape():
    x = torch.randn(4, 2, 32)
    out = FourierTransform().forward(x)
    assert out.shape == (4, 2, 32)
    assert torch.is_complex(out)


@pytest.mark.unit
def test_fourier_round_trip():
    x = torch.randn(4, 2, 32)
    ft = FourierTransform()
    recon = ft.inverse(ft.forward(x))
    assert recon.shape == x.shape
    assert torch.allclose(recon.real, x, atol=1e-5)
    assert torch.allclose(recon.imag, torch.zeros_like(recon.imag), atol=1e-5)


@pytest.mark.unit
def test_fourier_rejects_1d():
    with pytest.raises(ValueError, match="dimension 2 or higher"):
        FourierTransform().forward(torch.randn(16))
    with pytest.raises(ValueError, match="dimension 2 or higher"):
        FourierTransform().inverse(torch.randn(16))


# ── STFTransform ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_stft_forward_shape():
    x = torch.randn(3, 1, 64)
    out = STFTransform(n_fft=16, win_length=16, hop_length=4).forward(x)
    # (B, C, n_freq, n_time); n_freq = n_fft // 2 + 1 for the default one-sided STFT
    assert out.shape[0] == 3
    assert out.shape[1] == 1
    assert out.shape[2] == 16 // 2 + 1
    assert torch.is_complex(out)


@pytest.mark.unit
def test_stft_round_trip():
    x = torch.randn(3, 1, 64)
    st = STFTransform(n_fft=16, win_length=16, hop_length=4)
    recon = st.inverse(st.forward(x))
    assert recon.shape[0] == 3
    assert recon.shape[1] == 1
    # hann + hop = win/4 satisfies NOLA; interior reconstructs to precision.
    n = min(recon.shape[-1], x.shape[-1])
    assert torch.allclose(recon[..., :n], x[..., :n], atol=1e-4)


@pytest.mark.unit
def test_stft_default_hop_length():
    st = STFTransform(n_fft=16, win_length=16)
    assert st.hop_length == 8


@pytest.mark.unit
def test_stft_rejects_2d():
    with pytest.raises(ValueError, match="dimension 3 or higher"):
        STFTransform(n_fft=16, win_length=16).inverse(torch.randn(4, 16))


# ── get_transform factory ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_get_transform_none():
    assert get_transform(None) is None


@pytest.mark.unit
def test_get_transform_fft():
    t = get_transform({"name": "fft", "params": {}})
    assert isinstance(t, FourierTransform)


@pytest.mark.unit
def test_get_transform_stft():
    t = get_transform({"name": "stft", "params": {"n_fft": 16, "win_length": 16}})
    assert isinstance(t, STFTransform)
    assert isinstance(t, DomainTransform)


@pytest.mark.unit
def test_get_transform_unknown_raises():
    with pytest.raises(ValueError, match="Unknown transform name"):
        get_transform({"name": "wavelet"})


# ── Domain schema (Explanation defaults + explainer capability) ───────────────


@pytest.mark.unit
def test_explanation_domain_defaults_are_time():
    exp = Explanation(
        explainer="x",
        explanation_type="feature_attribution",
        exp_values=None,
        data=np.zeros((2, 1, 8)),
        labels=np.zeros(2),
        indices=np.arange(2),
        encoder=None,
        meta=None,
    )
    assert exp.data_domain is Domain.TIME
    assert exp.explanation_domain is Domain.TIME
    assert exp.transform is None


@pytest.mark.unit
def test_explainer_base_default_capability_is_time():
    assert ExplainerBase.explanation_domains == {Domain.TIME}
