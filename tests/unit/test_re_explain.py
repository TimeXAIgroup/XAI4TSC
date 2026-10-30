"""Unit tests for the re-explain callable the RE_EXPLAIN metrics need."""

import dataclasses
import inspect
from typing import ClassVar

import numpy as np
import pytest
import quantus as q
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
from xai4tsc.xai.base import ExplainerBase
from xai4tsc.xai.explain import _get_explanation
from xai4tsc.xai.feature_attribution import (
    IntegratedGradientsExplainer,
    TSHAPExplainer,
)
from xai4tsc.xai.freqrise import FreqRISEExplainer
from xai4tsc.xai.random_baseline import RandomFrequencyExplainer

_RE_EXPLAIN = sorted(
    (c for c, g in QUANTUS_GROUPS.items() if g is QuantusGroup.RE_EXPLAIN),
    key=lambda c: c.__name__,
)
_WORKING = [c for c in _RE_EXPLAIN if c not in BLOCKED_BY_BACKEND]
_BLOCKED = sorted(BLOCKED_BY_BACKEND, key=lambda c: c.__name__)


def _name(metric_class: type) -> str:
    """Readable test id for a Quantus class."""
    return metric_class.__name__


def _cheap_params(metric: type) -> dict:
    """Keep a re-explain metric fast, skipping params it does not accept."""
    params = {
        "disable_warnings": True,
        "display_progressbar": False,
        "return_aggregate": True,
    }
    accepted = inspect.signature(metric.__init__).parameters
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
        explainer=IntegratedGradientsExplainer(n_steps=8),
        model=model,
        data=data,
        labels=labels,
        encoder=LabelEncoder().fit(labels),
        targets="label",
        orig_indices=np.arange(4),
        device="cpu",
    )
    return model, exp


# ── the Explanation records what it was built with ────────────────────────────


@pytest.mark.unit
def test_explainer_is_recorded(setup):
    _, exp = setup
    assert isinstance(exp.explainer, IntegratedGradientsExplainer)
    assert exp.explainer._attribute_kwargs["n_steps"] == 8


class _Counting(ExplainerBase):
    """Counts its own calls, so a re-explanation on the stored instance shows."""

    explanation_type = "feature_attribution"

    def __init__(self):
        self.calls = 0

    def explain(self, model, exp, device, targets, **kwargs):
        self.calls += 1
        return np.ones(exp.data.shape, np.float32)


@pytest.mark.unit
def test_closure_leaves_the_stored_explainer_unchanged():
    """Re-explaining runs a deep copy, so the stored instance keeps its state."""
    labels = np.array([0, 1, 0, 1])
    model = _Tiny().eval()
    exp = _get_explanation(
        explainer=_Counting(),
        model=model,
        data=np.zeros((4, 1, 24), dtype=np.float32),
        labels=labels,
        encoder=LabelEncoder().fit(labels),
        targets="label",
        orig_indices=np.arange(4),
        device="cpu",
    )
    f = QuantusEvaluator._make_explain_func(exp, "cpu")
    f(model, exp.data, labels)
    f(model, exp.data, labels)

    assert exp.explainer.calls == 1


# ── the closure ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_closure_reproduces_a_direct_explain(setup):
    model, exp = setup
    f = QuantusEvaluator._make_explain_func(exp, "cpu")

    assert np.allclose(f(model, exp.data, np.array([0, 1, 0, 1])), exp.exp_values)


@pytest.mark.unit
def test_closure_accepts_the_device_kwarg_quantus_injects(setup):
    """Quantus writes `device` into explain_func_kwargs and forwards it."""
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


# ── the group predicate ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_requires_explain_func_matches_the_group_table():
    needing = {c for c in QUANTUS_METRICS.values() if quantus_requires_explain_func(c)}
    assert needing == set(_RE_EXPLAIN)


@pytest.mark.unit
def test_an_unclassified_metric_does_not_require_one():
    assert not quantus_requires_explain_func(_Capture)


# ── end to end through evaluate() ─────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("metric", _WORKING, ids=_name)
def test_re_explain_metrics_score(setup, metric):
    """Each unblocked RE_EXPLAIN metric returns a number instead of raising."""
    model, exp = setup
    score = evaluate(
        QuantusEvaluator(metric, **_cheap_params(metric)),
        model,
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
        call_params={"softmax": True},
    )
    assert score is not None
    assert np.all(np.isfinite(np.asarray(score, dtype=float)))


@pytest.mark.unit
@pytest.mark.parametrize("metric", _BLOCKED, ids=_name)
def test_backend_blocked_metrics_skip_with_a_reason(setup, metric, caplog):
    """Blocked metrics are refused with an explanation, not left to raise."""
    model, exp = setup
    with caplog.at_level("WARNING"):
        score = evaluate(
            QuantusEvaluator(metric),
            model,
            explanation=exp,
            data=exp.data,
            labels=exp.labels,
        )

    assert score is None
    assert metric.__name__ in caplog.text


# ── explain_func_kwargs is not mutated ────────────────────────────────────────


@pytest.mark.unit
def test_user_explain_func_kwargs_is_not_written_into(setup):
    """Quantus stamps `device` into a populated dict it is handed; copy first."""
    model, exp = setup
    mine = {"spare": 1}

    evaluate(
        QuantusEvaluator(
            q.MaxSensitivity,
            disable_warnings=True,
            display_progressbar=False,
            nr_samples=2,
        ),
        model,
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
        call_params={"softmax": True, "explain_func_kwargs": mine},
    )

    assert mine == {"spare": 1}, "the caller's dict must come back unchanged"


# ── aggregation ───────────────────────────────────────────────────────────────


class _Capture(q.Metric):
    """Stand-in Quantus metric that records the kwargs it was constructed with."""

    seen: ClassVar[dict] = {}

    def __init__(self, **kwargs):
        _Capture.seen = kwargs

    def __call__(self, **kwargs):
        return 1.0


@pytest.mark.unit
def test_aggregate_func_defaults_to_nanmean(setup):
    """Quantus aggregates with np.mean, so one NaN sample would poison the score."""
    model, exp = setup

    QuantusEvaluator(_Capture).evaluate(model, exp, exp.data, exp.labels, device="cpu")

    assert _Capture.seen["aggregate_func"] is np.nanmean


@pytest.mark.unit
def test_a_user_aggregate_func_wins(setup):
    model, exp = setup

    QuantusEvaluator(_Capture, aggregate_func=np.median).evaluate(
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
            QuantusEvaluator(q.MaxSensitivity),
            model,
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
            QuantusEvaluator(metric, **_cheap_params(metric)),
            model,
            explanation=exp,
            data=exp.data,
            labels=exp.labels,
            call_params={"softmax": True},
        )
        assert calls, f"{metric.__name__} never called the supplied explain_func"


# ── the closure stays pure for a stochastic explainer ─────────────────────────


def _stochastic_exp(explainer: ExplainerBase):
    """Build a real explanation from a stochastic explainer."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1, 24)).astype(np.float32)
    labels = np.array([0, 1, 0, 1])
    model = _Tiny().eval()
    exp = _get_explanation(
        explainer=explainer,
        model=model,
        data=data,
        labels=labels,
        encoder=LabelEncoder().fit(labels),
        targets="label",
        orig_indices=np.arange(4),
        device="cpu",
    )
    return model, exp


_STOCHASTIC = {
    "FreqRISE": lambda: FreqRISEExplainer(
        domain="fft", num_batches=2, batch_size=2, num_cells=4
    ),
    "Random_Frequency": RandomFrequencyExplainer,
    "TSHAP": lambda: TSHAPExplainer(
        perturb_baseline="random", stride=8, window_length=8
    ),
}


@pytest.mark.unit
@pytest.mark.parametrize("method", sorted(_STOCHASTIC))
def test_closure_is_pure_for_a_stochastic_explainer(method):
    """
    A re-explanation must match the attribution it is compared against.

    The closure builds one explainer and calls it repeatedly, so an explainer that
    advanced its generator between calls would make every RE_EXPLAIN metric a
    function of how many times that metric happens to re-explain.
    """
    model, exp = _stochastic_exp(_STOCHASTIC[method]())
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
    model, exp = _stochastic_exp(_STOCHASTIC["TSHAP"]())
    score = evaluate(
        QuantusEvaluator(q.MaxSensitivity, **_cheap_params(q.MaxSensitivity)),
        model,
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
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
        QuantusEvaluator(q.Complexity, disable_warnings=True),
        model,
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
        device="cpu",
        seed=12345,
    )
    assert score is not None

    accepting = evaluate(
        QuantusEvaluator(q.RandomLogit, **_cheap_params(q.RandomLogit)),
        model,
        explanation=exp,
        data=exp.data,
        labels=exp.labels,
        device="cpu",
        seed=12345,
    )
    assert accepting is not None


# ── per-call options are recorded and replayed ────────────────────────────────


class _Scaling(ExplainerBase):
    """Reads a per-call option, so a dropped one is visible in the output."""

    explanation_type = "feature_attribution"

    def explain(self, model, exp, device, targets, **kwargs):
        return np.full(exp.data.shape, float(kwargs.get("scale", 1.0)), np.float32)


def _scaled_explanation(call_params):
    """Explain four tiny samples with *call_params*, returning the Explanation."""
    labels = np.array([0, 1, 0, 1])
    return _get_explanation(
        explainer=_Scaling(),
        model=_Tiny().eval(),
        data=np.zeros((4, 1, 24), dtype=np.float32),
        labels=labels,
        encoder=LabelEncoder().fit(labels),
        call_params=call_params,
        targets="label",
        orig_indices=np.arange(4),
        device="cpu",
    )


@pytest.mark.unit
def test_call_params_are_recorded():
    exp = _scaled_explanation({"scale": 3})

    assert exp.explainer_call_params == {"scale": 3}


@pytest.mark.unit
def test_call_params_default_to_empty():
    assert _scaled_explanation(None).explainer_call_params == {}


@pytest.mark.unit
def test_call_params_reach_the_explainer():
    """Without the splat the option would be ignored and every value be 1.0."""
    exp = _scaled_explanation({"scale": 3})

    assert np.allclose(exp.exp_values, 3.0)
    assert not np.allclose(exp.exp_values, 1.0), (
        "the explainer default must differ, or the replay test cannot fail"
    )


@pytest.mark.unit
def test_closure_replays_the_recorded_call_params():
    """
    The invariant R43 protects: re-explaining reproduces what was scored.

    A closure that dropped the call params would return the explainer's default
    (1.0) while the attribution being scored holds 3.0, and the metric would
    report that configuration difference as a property of the explainer.
    """
    exp = _scaled_explanation({"scale": 3})
    model = _Tiny().eval()

    again = QuantusEvaluator._make_explain_func(exp, "cpu")(
        model, exp.data, np.array([0, 1, 0, 1])
    )

    assert np.allclose(again, exp.exp_values)


@pytest.mark.unit
def test_closure_replays_call_params_alongside_quantus_device():
    """`device` is already a positional of explain(); forwarding it would raise."""
    exp = _scaled_explanation({"scale": 3})

    again = QuantusEvaluator._make_explain_func(exp, "cpu")(
        _Tiny().eval(), exp.data, np.array([0, 1, 0, 1]), device="cpu"
    )

    assert np.allclose(again, exp.exp_values)


@pytest.mark.unit
def test_closure_warns_on_other_explain_func_kwargs(caplog):
    """A user's own explain_func_kwargs must not quietly override the record."""
    exp = _scaled_explanation({"scale": 3})

    with caplog.at_level("WARNING", logger="xai4tsc.evaluation.base"):
        again = QuantusEvaluator._make_explain_func(exp, "cpu")(
            _Tiny().eval(), exp.data, np.array([0, 1, 0, 1]), scale=99
        )

    assert "scale" in caplog.text
    assert np.allclose(again, 3.0), "the record wins over explain_func_kwargs"


@pytest.mark.unit
def test_multi_target_forwards_call_params_on_every_class():
    """The targets="all" loop calls explain() once per class; each must get them."""
    labels = np.array([0, 1, 0, 1])
    exp = _get_explanation(
        explainer=_Scaling(),
        model=_Tiny().eval(),
        data=np.zeros((4, 1, 24), dtype=np.float32),
        labels=labels,
        encoder=LabelEncoder().fit(labels),
        call_params={"scale": 3},
        targets="all",
        orig_indices=np.arange(4),
        device="cpu",
    )

    assert exp.exp_values.shape == (2, 4, 1, 24)
    assert np.allclose(exp.exp_values, 3.0)
