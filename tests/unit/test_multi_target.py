"""
Unit tests for domain-aware multi-target detection and the ``targets="all"`` path.

Covers :func:`xai4tsc.xai.is_multi_target` (the domain-aware replacement for the
``ndim == 4`` heuristic), the ``targets="all"`` expansion producing correctly
shaped and typed stacks for the time / frequency / time-frequency domains
(including complex-dtype preservation for the wrapper explainers), and a
``plot_exp`` smoke test that writes flat, index-tagged files without per-sample
sub-directories.
"""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn

from xai4tsc.data.base import MultiHotLabelEncoder, PowersetLabelEncoder
from xai4tsc.utils import STFTransform
from xai4tsc.xai import Domain, Explanation, is_multi_target
from xai4tsc.xai.explain import _get_explanation, plot_exp


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


def _exp(exp_values: np.ndarray, domain: Domain) -> Explanation:
    """Build a minimal Explanation carrying only what the detectors read."""
    return Explanation(
        explainer="x",
        explainer_params={},
        explanation_type="feature_attribution",
        exp_values=exp_values,
        data=np.zeros(1),
        labels=np.zeros(1),
        encoder=None,
        indices=np.arange(1),
        meta=None,
        explanation_domain=domain,
    )


# ── is_multi_target: domain-aware rank comparison ─────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("domain", "single_ndim"),
    [
        (Domain.TIME, 3),
        (Domain.FREQUENCY, 3),
        (Domain.TIME_FREQUENCY, 4),
    ],
)
def test_is_multi_target_by_domain(domain, single_ndim):
    single = _exp(np.zeros((1,) * single_ndim), domain)
    multi = _exp(np.zeros((1,) * (single_ndim + 1)), domain)
    assert is_multi_target(single) is False
    assert is_multi_target(multi) is True


@pytest.mark.unit
def test_single_target_tf_not_mistaken_for_multi_target():
    # A single-target time-frequency explanation is 4-D, exactly the rank a
    # bare ndim==4 heuristic would flag as multi-target.
    tf = _exp(np.zeros((2, 1, 9, 17)), Domain.TIME_FREQUENCY)
    assert is_multi_target(tf) is False


# ── targets="all" expansion: shape + dtype per domain ─────────────────────────


@pytest.mark.unit
def test_all_path_freqrise_fft_shape():
    data = np.random.default_rng(0).standard_normal((3, 1, 32)).astype(np.float32)
    encoder = LabelEncoder().fit(np.array([0, 1]))
    exp = _get_explanation(
        method="freqrise",
        model=_TinyNet(1, 32, 2).eval(),
        data=data,
        labels=np.array([0, 1, 0], dtype=np.int64),
        encoder=encoder,
        params={"domain": "fft", "num_cells": 16, "num_batches": 2, "batch_size": 2},
        targets="all",
        device="cpu",
        orig_indices=np.arange(3),
    )
    # (n_classes, n_samples, C=1, T//2 + 1)
    assert exp.exp_values.shape == (2, 3, 1, 17)
    assert exp.explanation_domain is Domain.FREQUENCY
    assert not np.iscomplexobj(exp.exp_values)  # FreqRISE relevance is real
    assert is_multi_target(exp) is True


@pytest.mark.unit
def test_all_path_freqrise_stft_is_5d():
    data = np.random.default_rng(1).standard_normal((2, 1, 64)).astype(np.float32)
    encoder = LabelEncoder().fit(np.array([0, 1]))
    transform = {
        "name": "stft",
        "params": {"n_fft": 16, "win_length": 16, "hop_length": 4},
    }
    exp = _get_explanation(
        method="freqrise",
        model=_TinyNet(1, 64, 2).eval(),
        data=data,
        labels=np.array([0, 1], dtype=np.int64),
        encoder=encoder,
        params={
            "domain": "stft",
            "transform": transform,
            "num_cells": 16,
            "num_batches": 2,
            "batch_size": 2,
        },
        targets="all",
        device="cpu",
        orig_indices=np.arange(2),
    )
    # (n_classes, n_samples, C=1, n_freq, n_time)
    assert exp.exp_values.ndim == 5
    assert exp.exp_values.shape[:3] == (2, 2, 1)
    assert exp.explanation_domain is Domain.TIME_FREQUENCY
    assert is_multi_target(exp) is True


@pytest.mark.unit
def test_all_path_frequency_wrapper_preserves_complex():
    # The wrapper transforms the *relevance* itself, so exp_values is complex —
    # the expansion must not silently drop the imaginary part.
    data = np.random.default_rng(2).standard_normal((2, 1, 32)).astype(np.float32)
    encoder = LabelEncoder().fit(np.array([0, 1]))
    exp = _get_explanation(
        method="frequency",
        model=_TinyNet(1, 32, 2).eval(),
        data=data,
        labels=np.array([0, 1], dtype=np.int64),
        encoder=encoder,
        params={
            "base": {"method": "integrated_gradients", "multiply_by_inputs": False}
        },
        targets="all",
        device="cpu",
        orig_indices=np.arange(2),
    )
    assert exp.exp_values.shape == (2, 2, 1, 32)  # full FFT keeps T
    assert np.iscomplexobj(exp.exp_values)
    assert is_multi_target(exp) is True


# ── plot_exp: flat, index-tagged output, no per-sample sub-directories ─────────


@pytest.mark.unit
@pytest.mark.parametrize("multi", [False, True])
def test_plot_exp_time_writes_flat_files(tmp_path, multi):
    encoder = LabelEncoder().fit(np.array([0, 1]))
    data = np.zeros((2, 1, 8), dtype=np.float32)
    if multi:
        exp_values = np.zeros((2, 2, 1, 8), dtype=np.float32)  # (K, n, C, T)
    else:
        exp_values = np.zeros((2, 1, 8), dtype=np.float32)  # (n, C, T)
    exp = Explanation(
        explainer="x",
        explainer_params={},
        explanation_type="feature_attribution",
        exp_values=exp_values,
        data=data,
        labels=np.array([0, 1], dtype=np.int64),
        encoder=encoder,
        indices=np.array([7, 9]),
        meta=None,
        explanation_domain=Domain.TIME,
    )
    plot_exp(exp, save_path=tmp_path, visualization_type=["points"])

    # Flat files tagged by index; no per-index sub-directories. (The plotters
    # append a per-sample counter before the extension, so match with a glob.)
    assert not any(p.is_dir() for p in tmp_path.iterdir())
    assert list(tmp_path.glob("relevance_index_7_class_0_points*.png"))
    assert list(tmp_path.glob("relevance_index_9_class_1_points*.png"))
    if multi:
        assert list(tmp_path.glob("relevance_index_7_class_1_points*.png"))


@pytest.mark.unit
@pytest.mark.parametrize("multi", [False, True])
def test_plot_exp_timefrequency_writes_flat_files(tmp_path, multi):
    encoder = LabelEncoder().fit(np.array([0, 1]))
    data = np.random.default_rng(3).standard_normal((2, 1, 64)).astype(np.float32)
    transform = STFTransform(n_fft=16, win_length=16, hop_length=4)
    coeffs = transform.forward(torch.as_tensor(data)).cpu().numpy()  # (n, C, nf, nt)
    grid = coeffs.shape[1:]
    # (K, n, C, nf, nt) when multi-target, else (n, C, nf, nt).
    exp_values = np.abs(np.stack([coeffs, coeffs], axis=0)) if multi else np.abs(coeffs)
    exp = Explanation(
        explainer="x",
        explainer_params={},
        explanation_type="feature_attribution",
        exp_values=exp_values,
        data=data,
        labels=np.array([0, 1], dtype=np.int64),
        encoder=encoder,
        indices=np.array([3, 5]),
        meta=None,
        explanation_domain=Domain.TIME_FREQUENCY,
        transform=transform,
    )
    assert grid  # sanity: grid is non-empty
    plot_exp(exp, save_path=tmp_path)

    assert not any(p.is_dir() for p in tmp_path.iterdir())
    assert list(tmp_path.glob("relevance_index_3_class_0_timefrequency*.png"))
    assert list(tmp_path.glob("relevance_index_5_class_1_timefrequency*.png"))
    if multi:
        assert list(tmp_path.glob("relevance_index_3_class_1_timefrequency*.png"))


# ── plot_exp: filenames name the explained class, spelled for a filesystem ────


def _time_exp(labels, encoder, explained=None, multi=False):
    """Build a minimal 2-sample time-domain explanation for the filename tests."""
    n_classes = len(getattr(encoder, "classes_", [0, 1]))
    exp_values = (
        np.zeros((n_classes, 2, 1, 8), dtype=np.float32)
        if multi
        else np.zeros((2, 1, 8), dtype=np.float32)
    )
    return Explanation(
        explainer="x",
        explainer_params={},
        explanation_type="feature_attribution",
        exp_values=exp_values,
        data=np.zeros((2, 1, 8), dtype=np.float32),
        encoder=encoder,
        labels=np.asarray(labels, dtype=np.int64),
        indices=np.array([7, 9]),
        meta=None,
        explanation_domain=Domain.TIME,
        explained_targets=None if explained is None else np.asarray(explained, int),
    )


@pytest.mark.unit
def test_plot_exp_multihot_filenames_need_no_quoting(tmp_path):
    # The raw decoded label is a numpy row, so interpolating it gave names like
    # "relevance_index_7_class_[0 0 1]_points.png": invalid on Windows, and a
    # single `find | xargs` splits one such name across three words.
    encoder = PowersetLabelEncoder().fit(np.array([[0, 0, 1], [1, 0, 0]]))
    exp = _time_exp([0, 1], encoder)

    plot_exp(exp, save_path=tmp_path, visualization_type=["points"])

    written = [p.name for p in tmp_path.iterdir()]
    assert written
    assert not [p for p in written if set(" []") & set(p)], written
    assert list(tmp_path.glob("relevance_index_7_class_0-0-1_points*.png"))


@pytest.mark.unit
def test_plot_exp_spells_a_class_the_same_in_both_target_modes(tmp_path):
    # The two paths used to read different helpers: class_names for multi-target
    # (already hyphen-joined) and decode_labels for single-target (a raw row).
    encoder = PowersetLabelEncoder().fit(np.array([[0, 0, 1], [1, 0, 0]]))
    single, multi = tmp_path / "single", tmp_path / "multi"

    plot_exp(
        _time_exp([0, 1], encoder), save_path=single, visualization_type=["points"]
    )
    plot_exp(
        _time_exp([0, 1], encoder, multi=True),
        save_path=multi,
        visualization_type=["points"],
    )

    def classes_of(d):
        return {p.name.split("_class_")[1].split("_points")[0] for p in d.iterdir()}

    assert classes_of(single) <= classes_of(multi)
    assert classes_of(single) == {"0-0-1", "1-0-0"}


@pytest.mark.unit
def test_plot_exp_names_the_explained_class_not_the_label(tmp_path):
    # Every shipped config explains target: "predicted", so for a misclassified
    # sample the label is not the class the attribution is about.
    encoder = LabelEncoder().fit(np.array([0, 1, 2]))
    exp = _time_exp([0, 1], encoder, explained=[2, 2])

    plot_exp(exp, save_path=tmp_path, visualization_type=["points"])

    assert list(tmp_path.glob("relevance_index_7_class_2_points*.png"))
    assert not list(tmp_path.glob("relevance_index_7_class_0_points*.png"))


@pytest.mark.unit
def test_plot_exp_falls_back_to_the_class_index_when_unnamed(tmp_path):
    # An encoder exposing no classes names nothing; the index is then the only
    # identifier, and a plot is still written rather than an IndexError raised.
    exp = _time_exp([0, 1], LabelEncoder())

    plot_exp(exp, save_path=tmp_path, visualization_type=["points"])

    assert list(tmp_path.glob("relevance_index_7_class_0_points*.png"))
    assert list(tmp_path.glob("relevance_index_9_class_1_points*.png"))


@pytest.mark.unit
def test_plot_exp_makes_a_supplied_class_name_safe_for_the_filesystem(tmp_path):
    # Class names used to be generated - an index, or a dash-joined multi-hot
    # row - and so were always usable as-is. An encoder's `classes` argument
    # puts arbitrary text in `classes_`, and a '/' would write into a
    # subdirectory that does not exist rather than naming the file.
    encoder = MultiHotLabelEncoder(classes=["a/b", "7 Hz"]).fit(
        np.array([[1, 0], [0, 1]])
    )
    exp = _time_exp([0, 1], encoder)

    plot_exp(exp, save_path=tmp_path, visualization_type=["points"])

    written = [p.name for p in tmp_path.iterdir()]
    assert written
    assert not [p for p in written if set(" /[]") & set(p)], written
    assert any("a_b" in name for name in written)
