"""
Unit tests for the Quantus metric classification and the domain rules it drives.

Covers the QUANTUS_GROUPS table, the required_domains it derives, how those reach
the METRICS registry through the partial factory, and the behaviour that follows:
attribution-only metrics score frequency / time-frequency explanations, while the
rest are refused rather than crashing inside Quantus.
"""

from typing import ClassVar

import numpy as np
import pytest
import torch
from torch import nn

from xai4tsc.evaluation import (
    METRICS,
    QUANTUS_GROUPS,
    QUANTUS_METRICS,
    QuantusEvaluator,
    QuantusGroup,
    evaluate,
    required_domains_for,
)
from xai4tsc.xai import Domain
from xai4tsc.xai.explain import _get_explanation

_ATTRIBUTION_ONLY = ("Complexity", "Effective Complexity", "Sparseness")
_STFT = {"name": "stft", "params": {"n_fft": 16, "win_length": 16, "hop_length": 8}}


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


def _explanation(method, params, data, labels, model):
    return _get_explanation(
        method=method,
        model=model,
        data=data,
        labels=labels,
        encoder=None,
        params=params,
        targets="label",
        device="cpu",
    )


# ── the table ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_every_quantus_metric_is_classified():
    # The table and the lookup must stay in step: an unclassified metric would
    # silently fall back to the conservative time-only default.
    assert set(QUANTUS_GROUPS) == set(QUANTUS_METRICS)


@pytest.mark.unit
def test_group_sizes_match_the_documented_partition():
    sizes = {g: 0 for g in QuantusGroup}
    for group in QUANTUS_GROUPS.values():
        sizes[group] += 1
    assert sizes == {
        QuantusGroup.ATTRIBUTION_ONLY: 3,
        QuantusGroup.MODEL_PERTURBATION: 16,
        QuantusGroup.LOCALISATION: 6,
        QuantusGroup.RE_EXPLAIN: 12,
    }


# ── the declarations the runner reads ─────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("metric", _ATTRIBUTION_ONLY)
def test_attribution_only_metrics_apply_to_any_domain(metric):
    # An empty set is the "any domain" convention shared with EvaluatorBase.
    assert required_domains_for(metric) == set()
    assert required_domains_for(metric, Domain.FREQUENCY) == set()
    assert QuantusEvaluator(metric).requires_native_domain is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "metric", ["Pixel-Flipping", "AUC", "Max-Sensitivity", "Faithfulness Estimate"]
)
def test_input_dependent_metrics_follow_the_model_native_domain(metric):
    # Not "time" but "whatever the model consumes": a model trained on spectra has
    # a frequency data domain, and these metrics then apply there instead.
    assert QuantusEvaluator(metric).requires_native_domain is True
    assert required_domains_for(metric) == {Domain.TIME}
    assert required_domains_for(metric, Domain.FREQUENCY) == {Domain.FREQUENCY}
    assert required_domains_for(metric, Domain.TIME_FREQUENCY) == {
        Domain.TIME_FREQUENCY
    }


@pytest.mark.unit
def test_native_metrics_keep_their_fixed_domain_whatever_the_data_is():
    # A frequency perturbation curve is defined *for* the frequency domain, so it
    # does not follow the model the way the input-dependent Quantus metrics do.
    assert required_domains_for("Frequency Perturbation") == {Domain.FREQUENCY}
    assert required_domains_for("Frequency Perturbation", Domain.FREQUENCY) == {
        Domain.FREQUENCY
    }


@pytest.mark.unit
def test_registry_entries_expose_their_declarations():
    # required_domains_for reads the declaration off the registry *value*, which for
    # a Quantus metric is a partial — it has to carry the declaration itself.
    assert METRICS["Complexity"].requires_native_domain is False
    assert METRICS["Pixel-Flipping"].requires_native_domain is True
    assert METRICS["Complexity"].data_applicability
    # ...but not the native evaluators' animate(), which is how the runner decides
    # a metric cannot be animated.
    assert not hasattr(METRICS["Complexity"], "animate")


@pytest.mark.unit
def test_required_domains_for_unknown_metric_is_empty():
    assert required_domains_for("Not A Metric") == set()


# ── behaviour ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("frequency", {"base": "integrated_gradients"}),
        ("timefrequency", {"base": "guided_backpropagation", "transform": _STFT}),
    ],
)
@pytest.mark.parametrize("metric", _ATTRIBUTION_ONLY)
def test_attribution_only_metrics_score_a_non_time_explanation(method, params, metric):
    # These read only the attribution, so they are meaningful on a coefficient
    # grid; Quantus only needs inputs on the same grid to pass its shape assert.
    data = np.random.default_rng(0).standard_normal((3, 1, 64)).astype(np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    model = _TinyNet(1, 64, 2).eval()
    exp = _explanation(method, params, data, labels, model)

    res = np.asarray(
        evaluate(
            model=model,
            metric=metric,
            explanation=exp,
            data=data,
            labels=labels,
            metric_class_params={"disable_warnings": True},
        ),
        dtype=float,
    )
    assert res.shape == (3,)
    assert np.isfinite(res).all()


@pytest.mark.unit
def test_input_dependent_metric_on_a_frequency_explanation_is_refused(caplog):
    # Previously an AssertionError from four frames inside Quantus, which by the
    # runner's error handling discarded the whole dataset.
    data = np.random.default_rng(1).standard_normal((3, 1, 64)).astype(np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    model = _TinyNet(1, 64, 2).eval()
    exp = _explanation(
        "frequency", {"base": "integrated_gradients"}, data, labels, model
    )

    with caplog.at_level("WARNING", logger="xai4tsc.evaluation.evaluate"):
        res = evaluate(
            model=model,
            metric="Pixel-Flipping",
            explanation=exp,
            data=data,
            labels=labels,
            metric_class_params={"disable_warnings": True},
        )
    assert res is None
    assert "Pixel-Flipping" in caplog.text
    assert "Frequency" in caplog.text


@pytest.mark.unit
def test_domain_inputs_are_the_coefficients_of_the_explanation_domain():
    data = np.random.default_rng(2).standard_normal((2, 1, 64)).astype(np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    model = _TinyNet(1, 64, 2).eval()

    time_exp = _explanation("integrated_gradients", {}, data, labels, model)
    assert QuantusEvaluator._domain_inputs(time_exp, data) is data

    freq_exp = _explanation(
        "frequency", {"base": "integrated_gradients"}, data, labels, model
    )
    x_batch = QuantusEvaluator._domain_inputs(freq_exp, data)
    assert x_batch.shape == freq_exp.exp_values.shape
    assert not np.iscomplexobj(x_batch)


@pytest.mark.unit
def test_domain_inputs_are_untouched_in_the_model_native_domain():
    # A model trained on coefficients has frequency `data`, already on the
    # attribution's grid: transforming it again would be the bug this guards.
    import dataclasses

    data = np.random.default_rng(4).standard_normal((2, 1, 33)).astype(np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    model = _TinyNet(1, 33, 2).eval()
    exp = _explanation("integrated_gradients", {}, data, labels, model)
    native_freq = dataclasses.replace(
        exp, data_domain=Domain.FREQUENCY, explanation_domain=Domain.FREQUENCY
    )
    assert QuantusEvaluator._domain_inputs(native_freq, data) is data


@pytest.mark.unit
def test_domain_inputs_reject_an_unreachable_domain_pair():
    import dataclasses

    data = np.random.default_rng(5).standard_normal((2, 1, 64)).astype(np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    model = _TinyNet(1, 64, 2).eval()
    exp = _explanation(
        "frequency", {"base": "integrated_gradients"}, data, labels, model
    )
    # Frequency data with a time-domain explanation would need an inverse transform.
    backwards = dataclasses.replace(
        exp, data_domain=Domain.FREQUENCY, explanation_domain=Domain.TIME
    )
    with pytest.raises(ValueError, match="only be lifted from the time domain"):
        QuantusEvaluator._domain_inputs(backwards, data)


@pytest.mark.unit
def test_domain_inputs_without_a_transform_raise():
    import dataclasses

    data = np.random.default_rng(3).standard_normal((2, 1, 64)).astype(np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    model = _TinyNet(1, 64, 2).eval()
    exp = _explanation(
        "frequency", {"base": "integrated_gradients"}, data, labels, model
    )

    with pytest.raises(ValueError, match="domain transform"):
        QuantusEvaluator._domain_inputs(dataclasses.replace(exp, transform=None), data)


# ── channel_first (the canonical layout, stated rather than guessed) ──────────


class _RecordingMetric:
    """Stands in for a Quantus class, recording what the adapter calls it with."""

    calls: ClassVar[list[dict]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        type(self).calls.append(kwargs)
        return [0.0]


@pytest.fixture
def _recording_metric():
    QUANTUS_METRICS["Recording"] = _RecordingMetric
    _RecordingMetric.calls = []
    yield _RecordingMetric
    del QUANTUS_METRICS["Recording"]


@pytest.mark.unit
def test_channel_first_is_stated_for_every_call(_recording_metric):
    # Quantus would otherwise infer the channel axis from the shape and transpose
    # a genuine (B, C, T) input whose channels outnumber its timesteps.
    data = np.zeros((2, 40, 24), dtype=np.float32)
    exp = _explanation(
        "integrated_gradients",
        {},
        data,
        np.array([0, 1], dtype=np.int64),
        _TinyNet(40, 24, 2).eval(),
    )
    QuantusEvaluator("Recording").evaluate(
        _TinyNet(40, 24, 2).eval(), exp, data, np.array([0, 1], dtype=np.int64)
    )
    assert _recording_metric.calls[-1]["channel_first"] is True


@pytest.mark.unit
def test_a_user_supplied_channel_first_wins(_recording_metric):
    data = np.zeros((2, 1, 8), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    exp = _explanation(
        "integrated_gradients", {}, data, labels, _TinyNet(1, 8, 2).eval()
    )
    QuantusEvaluator("Recording").evaluate(
        _TinyNet(1, 8, 2).eval(), exp, data, labels, channel_first=False
    )
    assert _recording_metric.calls[-1]["channel_first"] is False
