"""Unit tests for src/xai4tsc/evaluation/base.py helpers — pure, no Quantus runs."""

import math
from functools import partial

import numpy as np
import pytest

from xai4tsc.evaluation import METRICS, EvaluatorBase, register_metric
from xai4tsc.evaluation.base import (
    QuantusEvaluator,
    _apply_fraction_params,
    _resolve_callable_perturb_func,
    _resolve_string_callables,
)
from xai4tsc.utils.defaults import _fraction_params_T

# A real (class_name, param) pair from the live registry so these tests track
# the actual fraction-param configuration rather than a hard-coded guess.
_FRACTION_CLASS = "FaithfulnessCorrelation"
_FRACTION_KEY = "subset_size"


def _data(t=100):
    # Shape (B, C, T); only the last axis (T) matters for fraction conversion.
    return np.zeros((2, 1, t), dtype=np.float32)


# ── _apply_fraction_params ────────────────────────────────────────────────────


@pytest.mark.unit
def test_apply_fraction_params_registry_pair_is_known():
    # Guard: if the registry changes, the rest of this file needs updating.
    assert _FRACTION_KEY in _fraction_params_T[_FRACTION_CLASS]


@pytest.mark.unit
def test_apply_fraction_params_float_converted_to_int():
    out = _apply_fraction_params(_FRACTION_CLASS, {_FRACTION_KEY: 0.05}, _data(100))
    assert out[_FRACTION_KEY] == 5  # max(1, int(0.05 * 100))


@pytest.mark.unit
def test_apply_fraction_params_small_fraction_floors_to_one():
    out = _apply_fraction_params(_FRACTION_CLASS, {_FRACTION_KEY: 0.0001}, _data(100))
    assert out[_FRACTION_KEY] == 1  # never 0


@pytest.mark.unit
def test_apply_fraction_params_no_fraction_keys_returns_unchanged():
    params = {"nr_runs": 10}
    out = _apply_fraction_params("ClassWithNoFractionParams", params, _data(100))
    assert out == params


@pytest.mark.unit
def test_apply_fraction_params_non_fraction_key_passed_through():
    out = _apply_fraction_params(
        _FRACTION_CLASS, {_FRACTION_KEY: 0.1, "nr_runs": 10}, _data(100)
    )
    assert out[_FRACTION_KEY] == 10
    assert out["nr_runs"] == 10  # not a fraction param → untouched


@pytest.mark.unit
def test_apply_fraction_params_float_outside_unit_interval_kept():
    out = _apply_fraction_params(_FRACTION_CLASS, {_FRACTION_KEY: 5.0}, _data(100))
    assert out[_FRACTION_KEY] == 5.0  # > 1.0 → treated as an explicit int-like value


@pytest.mark.unit
def test_apply_fraction_params_int_value_kept():
    out = _apply_fraction_params(_FRACTION_CLASS, {_FRACTION_KEY: 7}, _data(100))
    assert out[_FRACTION_KEY] == 7  # already an int → no conversion


@pytest.mark.unit
def test_apply_fraction_params_does_not_mutate_input():
    params = {_FRACTION_KEY: 0.05}
    _apply_fraction_params(_FRACTION_CLASS, params, _data(100))
    assert params == {_FRACTION_KEY: 0.05}  # original untouched


# ── _resolve_callable_perturb_func ────────────────────────────────────────────


@pytest.mark.unit
def test_resolve_perturb_func_from_quantus_namespace():
    quantus = pytest.importorskip("quantus")
    # Pick any public callable exported by quantus to exercise the namespace path
    # without coupling to a specific function name.
    name = next(
        n
        for n in dir(quantus)
        if not n.startswith("_") and callable(getattr(quantus, n))
    )
    assert callable(_resolve_callable_perturb_func(name))


@pytest.mark.unit
def test_resolve_perturb_func_from_dotted_path():
    assert _resolve_callable_perturb_func("math.sqrt") is math.sqrt


@pytest.mark.unit
def test_resolve_perturb_func_unresolvable_returns_none():
    assert _resolve_callable_perturb_func("no_such_module.no_such_func") is None
    assert _resolve_callable_perturb_func("definitely_not_a_function") is None


# ── _resolve_string_callables ─────────────────────────────────────────────────


@pytest.mark.unit
def test_resolve_string_callables_replaces_resolvable_string():
    out = _resolve_string_callables({"perturb_func": "math.sqrt"}, "AnyMetric")
    assert out["perturb_func"] is math.sqrt


@pytest.mark.unit
def test_resolve_string_callables_drops_unresolvable_and_warns(caplog):
    with caplog.at_level("WARNING"):
        out = _resolve_string_callables({"perturb_func": "garbage"}, "AnyMetric")
    assert "perturb_func" not in out  # dropped so metric uses its own default
    assert "garbage" in caplog.text


@pytest.mark.unit
def test_resolve_string_callables_passes_through_non_callable_params():
    out = _resolve_string_callables({"abs": True, "nr_runs": 5}, "AnyMetric")
    assert out == {"abs": True, "nr_runs": 5}


@pytest.mark.unit
def test_resolve_string_callables_does_not_mutate_input():
    params = {"perturb_func": "math.sqrt"}
    _resolve_string_callables(params, "AnyMetric")
    assert params == {"perturb_func": "math.sqrt"}  # original string preserved


# ── unified registry: QuantusEvaluator(name) + homogeneous METRICS ────────────


@pytest.mark.unit
def test_quantus_evaluator_resolves_name():
    ev = QuantusEvaluator("AUC")
    assert ev.metric_name == "AUC"
    assert ev._metric_class.__name__ == "AUC"


@pytest.mark.unit
def test_quantus_evaluator_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown Quantus metric"):
        QuantusEvaluator("NotAMetric")


@pytest.mark.unit
def test_metrics_entries_yield_evaluatorbase():
    # Quantus entries are partials producing a QuantusEvaluator; native entries are
    # EvaluatorBase subclasses. Either way, calling the entry yields an EvaluatorBase.
    auc = METRICS["AUC"]
    assert isinstance(auc, partial)
    assert isinstance(auc(disable_warnings=True), QuantusEvaluator)
    native = METRICS["Frequency Perturbation"]
    assert isinstance(native, type) and issubclass(native, EvaluatorBase)


# ── register_metric contract ──────────────────────────────────────────────────


@pytest.fixture
def _clean_metrics():
    """Snapshot METRICS so tests that register can't leak into the global registry."""
    before = dict(METRICS)
    yield
    METRICS.clear()
    METRICS.update(before)


class _OkEvaluator(EvaluatorBase):
    def evaluate(self, model, explanation, data, labels, device="cpu", **kwargs):
        return 0.0


class _NotAnEvaluator:
    pass


@pytest.mark.unit
def test_register_metric_accepts_evaluatorbase_subclass(_clean_metrics):
    register_metric("MyEvaluator", _OkEvaluator)
    assert METRICS["MyEvaluator"] is _OkEvaluator


@pytest.mark.unit
def test_register_metric_rejects_non_evaluatorbase_class(_clean_metrics):
    with pytest.raises(TypeError, match="EvaluatorBase subclass"):
        register_metric("Bad", _NotAnEvaluator)


@pytest.mark.unit
def test_register_metric_rejects_non_class(_clean_metrics):
    with pytest.raises(TypeError, match="EvaluatorBase subclass"):
        register_metric("Bad", 123)
