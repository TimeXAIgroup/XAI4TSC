"""Unit tests for the re-explain callable the RE_EXPLAIN metrics need."""

import dataclasses
import inspect
from typing import ClassVar

import numpy as np
import pytest
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn

from xai4tsc.evaluation.base import QuantusEvaluator
from xai4tsc.evaluation.evaluate import QUANTUS_METRICS, evaluate
from xai4tsc.evaluation.quantus_groups import (
    BLOCKED_BY_BACKEND,
    QUANTUS_GROUPS,
    QuantusGroup,
    quantus_requires_explain_func,
)
from xai4tsc.xai._types import Domain
from xai4tsc.xai.explain import _get_explanation, build_explainer

_RE_EXPLAIN = sorted(
    n for n, g in QUANTUS_GROUPS.items() if g is QuantusGroup.RE_EXPLAIN
)
_WORKING = [n for n in _RE_EXPLAIN if n not in BLOCKED_BY_BACKEND]


def _cheap_params(metric: str) -> dict:
    """Keep a re-explain metric fast, skipping params it does not accept."""
    params = {
        "disable_warnings": True,
        "display_progressbar": False,
        "return_aggregate": True,
    }
    accepted = inspect.signature(QUANTUS_METRICS[metric].__init__).parameters
    if "nr_samples" in accepted:
        params["nr_samples"] = 2
    return params


class _Tiny(nn.Module):
    """Two-class linear model over a flattened series."""

    def __init__(self, t: int = 24):
        super().__init__()
        self.fc = nn.Linear(t, 2)

    def forward(self, x):
        return self.fc(x.flatten(1))


@pytest.fixture
def setup():
    """Build a tiny model and a real Integrated Gradients explanation."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1, 24)).astype(np.float32)
    labels = np.array([0, 1, 0, 1])
    model = _Tiny().eval()
    exp = _get_explanation(
        method="Integrated_Gradients",
        model=model,
        data=data,
        labels=labels,
        encoder=LabelEncoder().fit(labels),
        params={"n_steps": 8, "not_a_real_param": 1},
        targets="label",
        orig_indices=np.arange(4),
        device="cpu",
    )
    return model, exp


# ── the Explanation records what it was built with ────────────────────────────


@pytest.mark.unit
def test_explainer_params_are_recorded(setup):
    _, exp = setup
    assert exp.explainer == "integrated_gradients"
    assert exp.explainer_params == {"n_steps": 8, "not_a_real_param": 1}


@pytest.mark.unit
def test_explainer_params_rebuild_an_equivalent_explainer(setup):
    """The point of the field: the same attribution can be produced again."""
    model, exp = setup
    rebuilt = build_explainer(exp.explainer, exp.explainer_params)

    again = rebuilt.explain(model, exp, "cpu", [0, 1, 0, 1])

    assert np.allclose(again, exp.exp_values)


# ── the closure ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_closure_reproduces_a_direct_explain(setup):
    model, exp = setup
    f = QuantusEvaluator._make_explain_func(exp, "cpu")

    assert np.allclose(f(model, exp.data, np.array([0, 1, 0, 1])), exp.exp_values)


@pytest.mark.unit
def test_closure_accepts_the_device_kwarg_quantus_injects(setup):
    """Quantus writes `device` into explain_func_kwargs and splats it in."""
    model, exp = setup
    f = QuantusEvaluator._make_explain_func(exp, "cpu")

    assert np.allclose(
        f(model, exp.data, np.array([0, 1, 0, 1]), device="cpu"), exp.exp_values
    )


@pytest.mark.unit
def test_closure_does_not_mutate_the_explanation(setup):
    """An explainer may write to the explanation it is handed; the probe absorbs it."""
    model, exp = setup
    before_values = exp.exp_values.copy()
    before_data = exp.data.copy()
    before_domain = exp.explanation_domain

    other = np.random.default_rng(1).standard_normal((4, 1, 24)).astype(np.float32)
    QuantusEvaluator._make_explain_func(exp, "cpu")(model, other, np.zeros(4, int))

    assert np.array_equal(exp.exp_values, before_values)
    assert np.array_equal(exp.data, before_data)
    assert exp.explanation_domain is before_domain


@pytest.mark.unit
def test_closure_rejects_an_unregistered_explainer(setup):
    _, exp = setup
    orphan = dataclasses.replace(exp, explainer="no_such_explainer")

    with pytest.raises(ValueError, match="not a registered explainer"):
        QuantusEvaluator._make_explain_func(orphan, "cpu")


# ── the group predicate ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_requires_explain_func_matches_the_group_table():
    needing = {n for n in QUANTUS_METRICS if quantus_requires_explain_func(n)}
    assert needing == set(_RE_EXPLAIN)


@pytest.mark.unit
def test_an_unclassified_metric_does_not_require_one():
    assert not quantus_requires_explain_func("Not A Metric")


# ── end to end through evaluate() ─────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("metric", _WORKING)
def test_re_explain_metrics_score(setup, metric):
    """Each unblocked RE_EXPLAIN metric returns a number instead of raising."""
    model, exp = setup
    score = evaluate(
        model,
        metric,
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
        metric_class_params=_cheap_params(metric),
        metric_call_params={"softmax": True},
    )
    assert score is not None
    assert np.all(np.isfinite(np.asarray(score, dtype=float)))


@pytest.mark.unit
@pytest.mark.parametrize("metric", sorted(BLOCKED_BY_BACKEND))
def test_backend_blocked_metrics_skip_with_a_reason(setup, metric, caplog):
    """Blocked metrics are refused with an explanation, not left to raise."""
    model, exp = setup
    with caplog.at_level("WARNING"):
        score = evaluate(
            model, metric, explanation=exp, data=exp.data, labels=exp.labels
        )

    assert score is None
    assert metric in caplog.text


# ── explain_func_kwargs is not mutated ────────────────────────────────────────


@pytest.mark.unit
def test_user_explain_func_kwargs_is_not_written_into(setup):
    """Quantus stamps `device` into a populated dict it is handed; copy first."""
    model, exp = setup
    mine = {"spare": 1}

    evaluate(
        model,
        "Max-Sensitivity",
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
        metric_class_params={
            "disable_warnings": True,
            "display_progressbar": False,
            "nr_samples": 2,
        },
        metric_call_params={"softmax": True, "explain_func_kwargs": mine},
    )

    assert mine == {"spare": 1}, "the caller's dict must come back unchanged"


# ── aggregation ───────────────────────────────────────────────────────────────


class _Capture:
    """Stand-in Quantus metric that records the kwargs it was constructed with."""

    seen: ClassVar[dict] = {}

    def __init__(self, **kwargs):
        _Capture.seen = kwargs

    def __call__(self, **kwargs):
        return 1.0


@pytest.mark.unit
def test_aggregate_func_defaults_to_nanmean(setup, monkeypatch):
    """Quantus aggregates with np.mean, so one NaN sample would poison the score."""
    model, exp = setup
    monkeypatch.setitem(QUANTUS_METRICS, "Max-Sensitivity", _Capture)

    QuantusEvaluator("Max-Sensitivity").evaluate(
        model, exp, exp.data, exp.labels, device="cpu"
    )

    assert _Capture.seen["aggregate_func"] is np.nanmean


@pytest.mark.unit
def test_a_user_aggregate_func_wins(setup, monkeypatch):
    model, exp = setup
    monkeypatch.setitem(QUANTUS_METRICS, "Max-Sensitivity", _Capture)

    QuantusEvaluator("Max-Sensitivity", aggregate_func=np.median).evaluate(
        model, exp, exp.data, exp.labels, device="cpu"
    )

    assert _Capture.seen["aggregate_func"] is np.median


# ── domain gating still applies ───────────────────────────────────────────────


@pytest.mark.unit
def test_re_explain_metric_still_refuses_a_frequency_explanation(setup, caplog):
    """Supplying explain_func must not widen the domain rules."""
    model, exp = setup
    lifted = dataclasses.replace(exp, explanation_domain=Domain.FREQUENCY)

    with caplog.at_level("WARNING"):
        score = evaluate(
            model,
            "Max-Sensitivity",
            explanation=lifted,
            data=exp.data,
            labels=exp.labels,
        )

    assert score is None


@pytest.mark.unit
def test_every_unblocked_metric_actually_calls_our_explainer(setup, monkeypatch):
    """
    A metric that ignores the supplied callable must be blocked, not scored.

    Smooth MPRT returns a number while calling the library's own explain function
    internally, so it would report a score for an explainer it never ran. Counting
    invocations is the only way to tell that apart from a working metric.
    """
    model, exp = setup
    original = QuantusEvaluator._make_explain_func

    for metric in _WORKING:
        calls = []

        def counting(explanation, device, _orig=original, _calls=calls):
            inner = _orig(explanation, device)

            def wrapped(*args, **kwargs):
                _calls.append(1)
                return inner(*args, **kwargs)

            return wrapped

        monkeypatch.setattr(
            QuantusEvaluator, "_make_explain_func", staticmethod(counting)
        )
        evaluate(
            model,
            metric,
            explanation=exp,
            data=exp.data,
            labels=exp.labels,
            metric_class_params=_cheap_params(metric),
            metric_call_params={"softmax": True},
        )
        assert calls, f"{metric} never called the supplied explain_func"


# ── the closure stays pure for a stochastic explainer ─────────────────────────


def _stochastic_exp(method: str, params: dict):
    """Build a real explanation from a stochastic explainer."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1, 24)).astype(np.float32)
    labels = np.array([0, 1, 0, 1])
    model = _Tiny().eval()
    exp = _get_explanation(
        method=method,
        model=model,
        data=data,
        labels=labels,
        encoder=LabelEncoder().fit(labels),
        params=params,
        targets="label",
        orig_indices=np.arange(4),
        device="cpu",
    )
    return model, exp


_STOCHASTIC = [
    ("FreqRISE", {"domain": "fft", "num_batches": 2, "batch_size": 2, "num_cells": 4}),
    ("Random_Frequency", {}),
    ("TSHAP", {"perturb_baseline": "random", "stride": 8, "window_length": 8}),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "params"), _STOCHASTIC, ids=[m for m, _ in _STOCHASTIC]
)
def test_closure_is_pure_for_a_stochastic_explainer(method, params):
    """
    A re-explanation must match the attribution it is compared against.

    The closure builds one explainer and calls it repeatedly, so an explainer that
    advanced its generator between calls would make every RE_EXPLAIN metric a
    function of how many times that metric happens to re-explain.
    """
    model, exp = _stochastic_exp(method, params)
    f = QuantusEvaluator._make_explain_func(exp, "cpu")
    targets = np.array([0, 1, 0, 1])

    first = f(model, exp.data, targets)
    assert np.allclose(first, f(model, exp.data, targets))
    assert np.allclose(first, QuantusEvaluator.real_attribution(exp.exp_values))


@pytest.mark.unit
def test_max_sensitivity_is_nonzero_for_a_stochastic_explainer():
    """
    TSHAP with the "random" baseline is stochastic *and* input-dependent.

    It has to be a time-domain explainer: Max-Sensitivity requires the
    model-native domain, so a frequency attribution is skipped rather than scored.
    """
    model, exp = _stochastic_exp(
        "TSHAP",
        {"perturb_baseline": "random", "stride": 8, "window_length": 8},
    )
    score = evaluate(
        model,
        "Max-Sensitivity",
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
        metric_class_params=_cheap_params("Max-Sensitivity"),
        device="cpu",
    )
    assert score is not None and float(np.nanmean(score)) > 0


# ── the derived seed reaches only the metrics that declare one ────────────────


@pytest.mark.unit
def test_seed_is_offered_only_to_metrics_that_declare_it(setup):
    """
    Quantus metrics declare **kwargs and then reject what they do not name.

    So a seed cannot be passed blindly: `Complexity` would raise on it, while the
    randomisation metrics need it.
    """
    model, exp = setup

    score = evaluate(
        model,
        "Complexity",
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
        metric_class_params={"disable_warnings": True},
        device="cpu",
        seed=12345,
    )
    assert score is not None

    accepting = evaluate(
        model,
        "Random Logit",
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
        metric_class_params=_cheap_params("Random Logit"),
        device="cpu",
        seed=12345,
    )
    assert accepting is not None
