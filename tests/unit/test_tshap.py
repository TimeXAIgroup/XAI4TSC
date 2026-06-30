"""
Unit tests for TSHAP: perturbation helpers, the explainer, and runner wiring.

The Shapley efficiency identity holds for *any* model, so these tests use a
fast NumPy stub model exposing ``predict`` instead of training a network — no
torch, no I/O.
"""

import numpy as np
import pytest
from experiment_runner import explain as runner_explain
from sklearn.preprocessing import LabelEncoder

from xai4tsc.utils.perturbation import baseline_replacement, resolve_perturb_func
from xai4tsc.xai import DataType
from xai4tsc.xai._types import Explanation
from xai4tsc.xai.explain import generate_explanation
from xai4tsc.xai.feature_attribution import TSHAPExplainer

# ── stub models ───────────────────────────────────────────────────────────────


class _StubModel:
    """Base stub: the orchestrator calls ``eval()`` before explaining."""

    def eval(self):
        return self


class _SumModel(_StubModel):
    """Two-class model whose class-1 logit is the sum of all inputs (softmax)."""

    def predict(self, data, labels=None):
        data = np.asarray(data, dtype=np.float64)
        s = data.reshape(data.shape[0], -1).sum(axis=1)
        logits = np.stack([np.zeros_like(s), s], axis=1)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return np.argmax(probs, axis=1), probs


class _ChannelSumModel(_StubModel):
    """Class-1 logit is the sum of channel 0 only (channels are not symmetric)."""

    def predict(self, data, labels=None):
        data = np.asarray(data, dtype=np.float64)
        s = data[:, 0, :].sum(axis=1)
        logits = np.stack([np.zeros_like(s), s], axis=1)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return np.argmax(probs, axis=1), probs


class _ConstModel(_StubModel):
    """Returns a fixed probability vector regardless of the input."""

    def predict(self, data, labels=None):
        n = len(data)
        probs = np.tile([0.3, 0.7], (n, 1))
        return np.argmax(probs, axis=1), probs


def _make_exp(data, encoder=None):
    return Explanation(
        explainer="tshap",
        explanation_type="feature_attribution",
        exp_values=None,
        data=data.astype(np.float32),
        labels=np.zeros(len(data), dtype=int),
        indices=np.arange(len(data)),
        encoder=encoder,
        meta=None,
    )


# ── baseline_replacement / resolve_perturb_func ───────────────────────────────


@pytest.mark.unit
def test_baseline_vocabulary():
    x = np.arange(8, dtype=np.float32).reshape(2, 4)  # (C, T)
    idx = np.array([1, 2])

    assert np.all(baseline_replacement(x, idx, baseline="black")[:, idx] == 0)
    assert np.all(baseline_replacement(x, idx, baseline="white")[:, idx] == 1)

    mean_out = baseline_replacement(x, idx, baseline="mean")
    assert np.allclose(mean_out[:, idx], x.mean(axis=1, keepdims=True))

    ref = np.full_like(x, 9.0)
    cent = baseline_replacement(x, idx, baseline="centroid", reference=ref)
    assert np.all(cent[:, idx] == 9.0)
    assert np.array_equal(cent[:, 0], x[:, 0])  # untouched columns preserved


@pytest.mark.unit
def test_baseline_empty_indices_returns_unmodified_copy():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = baseline_replacement(x, np.array([], dtype=int), baseline="black")
    assert np.array_equal(out, x)
    assert out is not x


@pytest.mark.unit
def test_baseline_centroid_without_reference_warns_and_uses_mean(caplog):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    idx = np.array([0])
    with caplog.at_level("WARNING", logger="xai4tsc.utils.perturbation"):
        out = baseline_replacement(x, idx, baseline="centroid", reference=None)
    assert np.allclose(out[:, idx], x.mean(axis=1, keepdims=True))
    assert "centroid" in caplog.text.lower()


@pytest.mark.unit
def test_baseline_unknown_raises():
    with pytest.raises(ValueError, match="Unknown baseline"):
        baseline_replacement(np.zeros((1, 3)), np.array([0]), baseline="nope")


@pytest.mark.unit
def test_tshap_is_time_series_specific():
    assert TSHAPExplainer.data_applicability == {DataType.TIME_SERIES}


@pytest.mark.unit
def test_resolve_perturb_func():
    def custom(arr, indices, **kwargs):
        return arr

    assert resolve_perturb_func(custom) is custom
    assert resolve_perturb_func("mean") is baseline_replacement
    assert resolve_perturb_func("definitely_not_resolvable") is None


# ── Shapley correctness ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_window_shapley_efficiency():
    """phi(w) + phi(comp) == f(x) - f(xbar) for a window and its complement."""
    rng = np.random.default_rng(0)
    model = _SumModel()
    expl = TSHAPExplainer(perturb_baseline="black", window_length=3, stride=1)
    x = rng.standard_normal((1, 10)).astype(np.float32)
    target = 1

    x_bar = expl._full_background(x)
    f_x = expl._target_probs(model, x[None], target)[0]
    f_x_bar = expl._target_probs(model, x_bar[None], target)[0]

    mask = np.zeros_like(x, dtype=bool)
    mask[:, 2:5] = True
    phi = expl._phi_for_masks(model, x, x_bar, target, f_x, f_x_bar, [mask, ~mask])
    assert np.isclose(phi[0] + phi[1], f_x - f_x_bar, atol=1e-5)


@pytest.mark.unit
def test_constant_model_gives_zero_attribution():
    expl = TSHAPExplainer(perturb_baseline="black", window_length=3, stride=1)
    data = np.random.default_rng(1).standard_normal((2, 1, 12)).astype(np.float32)
    out = expl.explain(_ConstModel(), _make_exp(data), "cpu", targets=[1, 1])
    assert np.allclose(out, 0.0, atol=1e-6)


# ── shapes & dtype ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_output_shape_and_dtype():
    data = np.random.default_rng(2).standard_normal((3, 2, 16)).astype(np.float32)
    expl = TSHAPExplainer(perturb_baseline="black", window_length=4, stride=2)
    out = expl.explain(_SumModel(), _make_exp(data), "cpu", targets=[1, 1, 1])
    assert out.shape == (3, 2, 16)
    assert out.dtype == np.float32


@pytest.mark.unit
def test_targets_all_path_shape():
    data = np.random.default_rng(3).standard_normal((2, 1, 12)).astype(np.float32)
    encoder = LabelEncoder().fit(np.array([0, 1]))
    exp = generate_explanation(
        method="tshap",
        model=_SumModel(),
        data=data,
        labels=np.array([0, 1]),
        targets="all",
        indices=[0, 1],
        encoder=encoder,
        device="cpu",
        params={"window_length": 3, "stride": 1, "perturb_baseline": "black"},
    )
    # (n_classes, n_samples, C, T)
    assert exp.exp_values.shape == (2, 2, 1, 12)


# ── channel modes ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_shared_mode_broadcasts_across_channels():
    data = np.random.default_rng(4).standard_normal((1, 3, 12)).astype(np.float32)
    expl = TSHAPExplainer(
        perturb_baseline="black", window_length=3, stride=1, channel_mode="shared"
    )
    out = expl.explain(_SumModel(), _make_exp(data), "cpu", targets=[1])
    assert np.allclose(out[0, 0], out[0, 1])
    assert np.allclose(out[0, 0], out[0, 2])


@pytest.mark.unit
def test_per_channel_mode_resolves_channels():
    data = np.random.default_rng(5).standard_normal((1, 2, 12)).astype(np.float32)
    expl = TSHAPExplainer(
        perturb_baseline="black",
        window_length=3,
        stride=1,
        channel_mode="per_channel",
    )
    # Only channel 0 drives the prediction → channel 1 gets ~zero attribution.
    out = expl.explain(_ChannelSumModel(), _make_exp(data), "cpu", targets=[1])
    assert out.shape == (1, 2, 12)
    assert np.abs(out[0, 1]).max() < 1e-6
    assert np.abs(out[0, 0]).max() > 1e-6


@pytest.mark.unit
def test_invalid_channel_mode_raises():
    with pytest.raises(ValueError, match="channel_mode"):
        TSHAPExplainer(channel_mode="nonsense")


# ── ROI ───────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_roi_sharpens_to_uniform_regions():
    # Signal localised to timesteps 5:8 → ROI keeps that region, zeros elsewhere.
    x = np.zeros((1, 1, 15), dtype=np.float32)
    x[0, 0, 5:8] = 3.0
    expl = TSHAPExplainer(perturb_baseline="black", window_length=3, stride=1, roi=True)
    out = expl.explain(_SumModel(), _make_exp(x[0][None]), "cpu", targets=[1])[0, 0]
    assert (np.abs(out) < 1e-9).any()  # zeros outside the region
    nonzero = out[np.abs(out) > 1e-9]
    assert nonzero.size > 0
    # Values are assigned uniformly within each region.
    assert np.unique(np.round(nonzero, 5)).size <= 2


@pytest.mark.unit
def test_roi_no_relevant_windows_returns_zeros():
    # A constant model yields φ ≡ 0, so no window crosses the ROI threshold.
    data = np.random.default_rng(9).standard_normal((1, 1, 14)).astype(np.float32)
    expl = TSHAPExplainer(perturb_baseline="black", window_length=3, stride=1, roi=True)
    out = expl.explain(_ConstModel(), _make_exp(data), "cpu", targets=[1])
    assert out.shape == (1, 1, 14)
    assert np.all(out == 0.0)


@pytest.mark.unit
def test_roi_per_channel_runs():
    data = np.random.default_rng(10).standard_normal((1, 2, 14)).astype(np.float32)
    expl = TSHAPExplainer(
        perturb_baseline="black",
        window_length=3,
        stride=2,
        roi=True,
        channel_mode="per_channel",
    )
    out = expl.explain(_ChannelSumModel(), _make_exp(data), "cpu", targets=[1])
    assert out.shape == (1, 2, 14)
    assert np.all(np.isfinite(out))
    # The inert channel stays exactly zero after ROI sharpening.
    assert np.all(out[0, 1] == 0.0)


# ── window-coverage edges ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_single_window_when_length_ge_series():
    # window_length=1.0 (fraction) → one window spanning the whole series.
    data = np.random.default_rng(11).standard_normal((1, 1, 8)).astype(np.float32)
    expl = TSHAPExplainer(perturb_baseline="black", window_length=1.0, stride=1)
    out = expl.explain(_SumModel(), _make_exp(data), "cpu", targets=[1])
    assert out.shape == (1, 1, 8)
    assert np.all(np.isfinite(out))


@pytest.mark.unit
def test_large_stride_still_covers_endpoints():
    # stride larger than the window count must still attribute the final timestep.
    data = np.random.default_rng(12).standard_normal((1, 1, 12)).astype(np.float32)
    expl = TSHAPExplainer(perturb_baseline="black", window_length=3, stride=100)
    out = expl.explain(_SumModel(), _make_exp(data), "cpu", targets=[1])[0, 0]
    assert out.shape == (12,)
    assert np.all(np.isfinite(out))
    assert np.abs(out[-1]) > 0  # last timestep is covered, not left at zero


# ── determinism & window length ───────────────────────────────────────────────


@pytest.mark.unit
def test_random_baseline_deterministic_with_seed():
    data = np.random.default_rng(6).standard_normal((1, 1, 12)).astype(np.float32)
    kw = dict(
        perturb_baseline="random",
        window_length=3,
        stride=1,
        n_perturb_samples=3,
        seed=42,
    )
    a = TSHAPExplainer(**kw).explain(_SumModel(), _make_exp(data), "cpu", [1])
    b = TSHAPExplainer(**kw).explain(_SumModel(), _make_exp(data), "cpu", [1])
    assert np.array_equal(a, b)


@pytest.mark.unit
def test_window_length_fraction_and_int():
    expl = TSHAPExplainer(window_length=0.1, perturb_baseline="black")
    assert expl._window_len(100) == 10
    expl_int = TSHAPExplainer(window_length=5, perturb_baseline="black")
    assert expl_int._window_len(100) == 5
    # Clamped to series length.
    assert expl_int._window_len(3) == 3


# ── centroid / background guard ───────────────────────────────────────────────


@pytest.mark.unit
def test_centroid_constructed_from_background_data():
    bg = np.stack([np.full((1, 6), 2.0), np.full((1, 6), 4.0)]).astype(
        np.float32
    )  # (2, 1, 6)
    expl = TSHAPExplainer(perturb_baseline="centroid", background_data=bg)
    assert expl.centroid.shape == (1, 6)
    assert np.allclose(expl.centroid, 3.0)


@pytest.mark.unit
def test_centroid_without_background_falls_back_to_mean(caplog):
    with caplog.at_level("WARNING", logger="xai4tsc.xai.feature_attribution"):
        expl = TSHAPExplainer(perturb_baseline="centroid", background_data=None)
    assert expl.perturb_baseline == "mean"
    assert expl.centroid is None
    assert "centroid" in caplog.text.lower()


@pytest.mark.unit
def test_non_array_background_data_raises():
    with pytest.raises(TypeError, match="numpy array"):
        TSHAPExplainer(background_data="train_set")


# ── runner: background-data selector resolution ───────────────────────────────


@pytest.mark.unit
def test_resolve_background_data_known_selector():
    train = np.zeros((4, 1, 6), dtype=np.float32)
    splits = {"train_set": train, "test_set": None, "val_set": None}
    assert runner_explain._resolve_background_data("train_set", splits) is train


@pytest.mark.unit
def test_resolve_background_data_unknown_selector_warns(caplog):
    with caplog.at_level("WARNING", logger="xai4tsc.runner.explain"):
        out = runner_explain._resolve_background_data("bogus", {"train_set": None})
    assert out is None
    assert "unknown background_data selector" in caplog.text.lower()


@pytest.mark.unit
def test_resolve_background_data_none_and_passthrough():
    assert runner_explain._resolve_background_data(None, {}) is None
    arr = np.zeros((1, 1, 3))
    assert runner_explain._resolve_background_data(arr, {}) is arr
