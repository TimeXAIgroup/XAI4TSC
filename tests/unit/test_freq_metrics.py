"""
Unit tests for the Phase 3 frequency / time-frequency evaluation metrics.

Covers the RFFTransform (Step 0), the EvaluatorBase.real_attribution invariant,
the FrequencyEvaluator / TimeFrequencyEvaluator(+Gaussian) metrics on both
real (FreqRISE) and complex (explanation-space wrapper) relevance, their
required_domains declarations, and the config-level domain-compatibility check.
"""

import numpy as np
import pytest
import torch
from torch import nn

from xai4tsc import EvaluatorBase
from xai4tsc.evaluation.evaluate import METRICS, evaluate
from xai4tsc.evaluation.frequency_evaluate import FrequencyEvaluator
from xai4tsc.evaluation.timefrequency_auc import (
    TimeFrequencyAUCEvaluator,
    _ground_truth_mask,
    _sample_regions,
)
from xai4tsc.evaluation.timefrequency_perturbation import (
    TimeFrequencyEvaluator,
    TimeFrequencyEvaluatorGaussian,
)
from xai4tsc.utils import RFFTransform, get_transform
from xai4tsc.utils.fourier_transforms import STFTransform
from xai4tsc.xai import Domain
from xai4tsc.xai._types import Explanation
from xai4tsc.xai.explain import _get_explanation

_STFT = {"name": "stft", "params": {"n_fft": 16, "win_length": 16, "hop_length": 4}}


class _TinyNet(nn.Module):
    def __init__(self, channels: int, timesteps: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(channels * timesteps, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))

    def predict(self, x):
        with torch.no_grad():
            p = torch.softmax(
                self(torch.as_tensor(np.asarray(x), dtype=torch.float32)), 1
            ).numpy()
        return p.argmax(1), p


def _explain(method, params, data, labels, model):
    return _get_explanation(
        method=method,
        model=model,
        data=data,
        labels=labels,
        encoder=None,
        params=params,
        targets="label",
        device="cpu",
        orig_indices=np.arange(len(data)),
    )


# ── RFFTransform (Step 0) ─────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("t_len", [32, 31])
def test_rfftransform_round_trip(t_len):
    x = torch.randn(3, 1, t_len)
    r = RFFTransform()
    fwd = r.forward(x)
    assert fwd.shape == (3, 1, t_len // 2 + 1)
    assert torch.is_complex(fwd)
    recon = r.inverse(fwd)
    assert recon.shape == x.shape
    assert torch.allclose(recon, x, atol=1e-5)


@pytest.mark.unit
def test_get_transform_rfft():
    assert isinstance(get_transform({"name": "rfft", "params": {}}), RFFTransform)


@pytest.mark.unit
def test_freqrise_fft_exposes_rfftransform():
    data = np.random.default_rng(0).standard_normal((2, 1, 32)).astype(np.float32)
    model = _TinyNet(1, 32, 2).eval()
    exp = _explain(
        "freqrise",
        {"domain": "fft", "num_batches": 2, "batch_size": 2, "num_cells": 8, "seed": 0},
        data,
        np.array([0, 1], dtype=np.int64),
        model,
    )
    assert isinstance(exp.transform, RFFTransform)
    assert exp.transform.forward(torch.as_tensor(data)).shape == exp.exp_values.shape


# ── real_attribution invariant ────────────────────────────────────────────────


@pytest.mark.unit
def test_real_attribution_reduces_complex():
    a = np.array([3 + 4j, 0 + 1j], dtype=np.complex64)
    out = EvaluatorBase.real_attribution(a)
    assert not np.iscomplexobj(out)
    np.testing.assert_allclose(out, [5.0, 1.0], atol=1e-5)


@pytest.mark.unit
def test_real_attribution_passes_real_through_and_none():
    a = np.array([-1.0, 2.0], dtype=np.float32)
    assert EvaluatorBase.real_attribution(a) is a  # untouched (signed preserved)
    assert EvaluatorBase.real_attribution(None) is None


# ── metric registration + required_domains ────────────────────────────────────


@pytest.mark.unit
def test_freq_metrics_registered_and_required_domains():
    assert METRICS["Frequency Perturbation"] is FrequencyEvaluator
    assert METRICS["Time-Frequency Perturbation"] is TimeFrequencyEvaluator
    assert (
        METRICS["Time-Frequency Perturbation Gaussian"]
        is TimeFrequencyEvaluatorGaussian
    )
    assert FrequencyEvaluator.required_domains == {Domain.FREQUENCY}
    assert TimeFrequencyEvaluator.required_domains == {Domain.TIME_FREQUENCY}
    assert TimeFrequencyEvaluatorGaussian.required_domains == {Domain.TIME_FREQUENCY}


# ── FrequencyPerturbation (real + complex relevance via evaluate) ─────────────


@pytest.mark.unit
def test_frequency_perturbation_on_real_relevance():
    data = np.random.default_rng(1).standard_normal((4, 1, 32)).astype(np.float32)
    labels = np.array([0, 1, 2, 0], dtype=np.int64)
    model = _TinyNet(1, 32, 3).eval()
    exp = _explain(
        "freqrise",
        {"domain": "fft", "num_batches": 3, "batch_size": 4, "num_cells": 8, "seed": 0},
        data,
        labels,
        model,
    )
    res = np.asarray(
        evaluate(
            model=model,
            metric="Frequency Perturbation",
            explanation=exp,
            data=data,
            labels=labels,
            metric_class_params={"features_in_step": 3},
        )
    )
    assert res.shape[0] == 4
    assert np.isfinite(res).all()


@pytest.mark.unit
def test_frequency_perturbation_on_complex_relevance():
    # FrequencyExplainer relevance is complex; the boundary reduces it to magnitude.
    data = np.random.default_rng(2).standard_normal((3, 1, 32)).astype(np.float32)
    labels = np.array([0, 1, 2], dtype=np.int64)
    model = _TinyNet(1, 32, 3).eval()
    exp = _explain("frequency", {"base": "integrated_gradients"}, data, labels, model)
    assert np.iscomplexobj(exp.exp_values)
    res = np.asarray(
        evaluate(
            model=model,
            metric="Frequency Perturbation",
            explanation=exp,
            data=data,
            labels=labels,
            metric_class_params={"features_in_step": 4},
        )
    )
    assert res.shape[0] == 3
    assert np.isfinite(res).all()


@pytest.mark.unit
def test_evaluate_filters_quantus_only_params_for_native_metric():
    # Shared Quantus-only params (e.g. normalise/abs) must be filtered out before a
    # native evaluator is constructed, not raise a TypeError.
    data = np.random.default_rng(7).standard_normal((3, 1, 32)).astype(np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    model = _TinyNet(1, 32, 2).eval()
    exp = _explain(
        "freqrise",
        {"domain": "fft", "num_batches": 2, "batch_size": 4, "num_cells": 8, "seed": 0},
        data,
        labels,
        model,
    )
    res = np.asarray(
        evaluate(
            model=model,
            metric="Frequency Perturbation",
            explanation=exp,
            data=data,
            labels=labels,
            metric_class_params={
                "features_in_step": 3,
                "normalise": True,  # Quantus-only — must be dropped
                "abs": True,  # Quantus-only — must be dropped
            },
        )
    )
    assert res.shape[0] == 3
    assert np.isfinite(res).all()


@pytest.mark.unit
def test_frequency_perturbation_aoc_mode():
    data = np.random.default_rng(3).standard_normal((3, 1, 32)).astype(np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    model = _TinyNet(1, 32, 2).eval()
    exp = _explain(
        "freqrise",
        {"domain": "fft", "num_batches": 2, "batch_size": 4, "num_cells": 8, "seed": 0},
        data,
        labels,
        model,
    )
    res = evaluate(
        model=model,
        metric="Frequency Perturbation",
        explanation=exp,
        data=data,
        labels=labels,
        metric_class_params={"features_in_step": 3, "return_aoc_per_sample": True},
    )
    assert isinstance(res, list)
    assert len(res) == 3
    assert all(np.isfinite(r) for r in res)


# ── TimeFrequency metrics ─────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "metric", ["Time-Frequency Perturbation", "Time-Frequency Perturbation Gaussian"]
)
def test_timefrequency_perturbation(metric):
    data = np.random.default_rng(4).standard_normal((2, 1, 64)).astype(np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    model = _TinyNet(1, 64, 2).eval()
    exp = _explain(
        "timefrequency",
        {"base": "guided_backpropagation", "transform": _STFT},
        data,
        labels,
        model,
    )
    res = np.asarray(
        evaluate(
            model=model,
            metric=metric,
            explanation=exp,
            data=data,
            labels=labels,
            metric_class_params={"features_in_step": 5},
        )
    )
    assert res.shape[0] == 2
    assert np.isfinite(res).all()


# ── TimeFrequencyAUC (ground-truth localization) ──────────────────────────────


def _gt_meta(n, freq=7, pos=0, length=None, channel=0):
    """One discriminative region per sample at *freq* (class key 0)."""
    region = {"channel": channel, "pos": pos, "len": length, "freq": freq, "phase": 0.0}
    return [
        {"ground_truth": {0: [dict(region)]}, "non_discriminative": []}
        for _ in range(n)
    ]


def _freq_explanation(a, data, metadata, transform):
    return Explanation(
        explainer="x",
        explanation_type="feature_attribution",
        exp_values=a,
        data=data,
        labels=np.zeros(len(data), dtype=np.int64),
        indices=np.arange(len(data)),
        encoder=None,
        meta={},
        explanation_domain=Domain.FREQUENCY,
        transform=transform,
        metadata=metadata,
    )


@pytest.mark.unit
def test_tfauc_registered_and_required_domains():
    assert METRICS["Time-Frequency AUC"] is TimeFrequencyAUCEvaluator
    assert TimeFrequencyAUCEvaluator.required_domains == {
        Domain.FREQUENCY,
        Domain.TIME_FREQUENCY,
    }


@pytest.mark.unit
def test_sample_regions_accepts_dict_and_flat_list():
    # Per-class dict {class_key: [region, ...]} is unioned across classes.
    per_class = {
        "ground_truth": {0: [{"freq": 7}], 1: [{"freq": 19}, {"freq": 37}]},
    }
    assert _sample_regions(per_class) == [{"freq": 7}, {"freq": 19}, {"freq": 37}]
    # The shipped freq_shapes dataset stores a flat (already-unioned) region list.
    flat = {"ground_truth": [{"freq": 7}, {"freq": 19}, {"freq": 37}]}
    assert _sample_regions(flat) == [{"freq": 7}, {"freq": 19}, {"freq": 37}]
    # Non-dict regions are dropped; non-mapping metadata yields nothing.
    assert _sample_regions({"ground_truth": [{"freq": 7}, 42]}) == [{"freq": 7}]
    assert _sample_regions(None) == []
    assert _sample_regions({"ground_truth": 5}) == []


@pytest.mark.unit
def test_tfauc_mask_frequency_grid():
    # rFFT grid (C, n_freq): a region at freq=7 marks exactly bin 7.
    regions = _sample_regions({"ground_truth": {0: [{"channel": 0, "freq": 7}]}})
    mask = _ground_truth_mask(regions, (1, 17), signal_length=32, transform=None)
    assert mask.sum() == 1 and mask[0, 7]


@pytest.mark.unit
def test_tfauc_mask_time_frequency_grid():
    stft = STFTransform(n_fft=16, win_length=16, hop_length=8)
    stft.forward(torch.randn(1, 1, 64))
    regions = [{"channel": 0, "pos": 16, "len": 16, "freq": 4}]
    mask = _ground_truth_mask(regions, (1, 9, 9), signal_length=64, transform=stft)
    # freq bin round(4*16/64)=1; frames overlapping [16,32) with hop 8 → 2..4.
    assert mask.any() and not mask.all()
    assert mask[0, 1, 2] and mask[0, 1, 3]


@pytest.mark.unit
def test_tfauc_perfect_localization_scores_one():
    t_len, n = 64, 4
    data = np.random.default_rng(0).standard_normal((n, 1, t_len)).astype(np.float32)
    rfft = RFFTransform()
    rfft.forward(torch.as_tensor(data))
    n_freq = t_len // 2 + 1
    a = np.full((n, 1, n_freq), 0.01, dtype=np.float32)
    a[:, :, 7] = 9.0  # all relevance on the discriminative bin
    exp = _freq_explanation(a, data, _gt_meta(n, freq=7), rfft)
    res = evaluate(
        model=_TinyNet(1, t_len, 2).eval(),
        metric="Time-Frequency AUC",
        explanation=exp,
        data=data,
        labels=np.zeros(n, dtype=np.int64),
        metric_class_params={"return_aggregate": True},
    )
    assert res == pytest.approx(1.0)


@pytest.mark.unit
def test_tfauc_per_sample_and_random_baseline():
    t_len, n = 64, 4
    data = np.random.default_rng(1).standard_normal((n, 1, t_len)).astype(np.float32)
    rfft = RFFTransform()
    rfft.forward(torch.as_tensor(data))
    n_freq = t_len // 2 + 1
    a = np.random.default_rng(2).random((n, 1, n_freq)).astype(np.float32)
    exp = _freq_explanation(a, data, _gt_meta(n, freq=11), rfft)
    res = evaluate(
        model=_TinyNet(1, t_len, 2).eval(),
        metric="Time-Frequency AUC",
        explanation=exp,
        data=data,
        labels=np.zeros(n, dtype=np.int64),
        metric_class_params={"return_aggregate": False},
    )
    assert isinstance(res, list) and len(res) == n
    assert all(0.0 <= r <= 1.0 for r in res)


@pytest.mark.unit
def test_tfauc_no_ground_truth_regions_is_nan():
    # A sample with no discriminative regions (e.g. the all-zero label) is undefined.
    t_len, n = 64, 2
    data = np.random.default_rng(3).standard_normal((n, 1, t_len)).astype(np.float32)
    rfft = RFFTransform()
    rfft.forward(torch.as_tensor(data))
    n_freq = t_len // 2 + 1
    a = np.random.default_rng(4).random((n, 1, n_freq)).astype(np.float32)
    empty_meta = [{"ground_truth": {}, "non_discriminative": []} for _ in range(n)]
    exp = _freq_explanation(a, data, empty_meta, rfft)
    res = evaluate(
        model=_TinyNet(1, t_len, 2).eval(),
        metric="Time-Frequency AUC",
        explanation=exp,
        data=data,
        labels=np.zeros(n, dtype=np.int64),
        metric_class_params={"return_aggregate": True},
    )
    assert np.isnan(res)


@pytest.mark.unit
def test_tfauc_missing_metadata_returns_nan_and_warns(caplog):
    data = np.zeros((2, 1, 8), dtype=np.float32)
    exp = _freq_explanation(
        np.zeros((2, 1, 5), dtype=np.float32), data, None, RFFTransform()
    )
    metric = TimeFrequencyAUCEvaluator()
    with caplog.at_level("WARNING", logger="xai4tsc.evaluation.timefrequency_auc"):
        res = metric.evaluate(None, exp, data, np.zeros(2, dtype=np.int64))
    assert np.isnan(res)
    assert "ground-truth metadata" in caplog.text


# ── domain-compatibility validation (config_sanity_check) ─────────────────────


def _cfg(explainer_method, metric_name):
    return {
        "general": {},
        "train_config": {"models": [{"model": "FCN"}]},
        "explanation_config": {"explainers": [{"method": explainer_method}]},
        "evaluation_config": {"metrics": [{"metric": metric_name}]},
    }


@pytest.mark.unit
def test_config_rejects_incompatible_metric():
    from experiment_runner.config import config_sanity_check

    with pytest.raises(ValueError, match="requires an explanation in domain"):
        config_sanity_check(_cfg("Integrated_Gradients", "Frequency Perturbation"))


@pytest.mark.unit
def test_config_accepts_compatible_metric():
    from experiment_runner.config import config_sanity_check

    assert config_sanity_check(_cfg("FreqRISE", "Frequency Perturbation"))


@pytest.mark.unit
def test_config_rejects_tfauc_on_time_domain_explainer():
    from experiment_runner.config import config_sanity_check

    with pytest.raises(ValueError, match="requires an explanation in domain"):
        config_sanity_check(_cfg("Integrated_Gradients", "Time-Frequency AUC"))


@pytest.mark.unit
def test_config_accepts_tfauc_on_frequency_explainer():
    from experiment_runner.config import config_sanity_check

    assert config_sanity_check(_cfg("FreqRISE", "Time-Frequency AUC"))
