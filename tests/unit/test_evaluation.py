"""Unit tests for src/xai4tsc/evaluation/base.py helpers — pure, no Quantus runs."""

from functools import partial

import numpy as np
import pytest
import quantus as q
from sklearn.preprocessing import LabelEncoder

from xai4tsc.evaluation import METRICS, EvaluatorBase
from xai4tsc.evaluation.base import (
    QuantusEvaluator,
    _apply_fraction_params,
    _fraction_params_T,
)

# A real (class_name, param) pair from the live registry so these tests track
# the actual fraction-param configuration rather than a hard-coded guess.
_FRACTION_CLASS = q.FaithfulnessCorrelation
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
    out = _apply_fraction_params(q.Complexity, params, _data(100))
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


# ── unified registry: QuantusEvaluator(class) + homogeneous METRICS ───────────


@pytest.mark.unit
def test_quantus_evaluator_takes_the_class():
    ev = QuantusEvaluator(q.AUC)
    assert ev.metric_name == "AUC"
    assert ev.quantus_class is q.AUC


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["AUC", LabelEncoder])
def test_quantus_evaluator_rejects_anything_but_a_metric_class(bad):
    with pytest.raises(TypeError, match=r"quantus\.Metric subclass"):
        QuantusEvaluator(bad)


@pytest.mark.unit
def test_evaluate_rejects_a_metric_name():
    from xai4tsc.evaluation import evaluate

    with pytest.raises(TypeError, match="EvaluatorBase instance"):
        evaluate("Complexity", None, None)


@pytest.mark.unit
def test_every_quantus_metric_name_maps_to_its_class():
    from xai4tsc.evaluation import QUANTUS_METRICS

    for name, cls in QUANTUS_METRICS.items():
        assert METRICS[name]().quantus_class is cls


@pytest.mark.unit
def test_metrics_entries_yield_evaluatorbase():
    # Quantus entries are partials producing a QuantusEvaluator; native entries are
    # EvaluatorBase subclasses. Either way, calling the entry yields an EvaluatorBase.
    auc = METRICS["AUC"]
    assert isinstance(auc, partial)
    assert isinstance(auc(disable_warnings=True), QuantusEvaluator)
    native = METRICS["Frequency Perturbation"]
    assert isinstance(native, type) and issubclass(native, EvaluatorBase)
