"""
Unit tests for xai4tsc/xai/explain.py sample selection and registry — no torch.

The heavy ``_get_explanation`` call is monkeypatched with a recorder that
captures the indices it receives, so the index-selection logic of
``generate_explanation`` is tested without a real model or attribution run.
"""

import numpy as np
import pytest

from xai4tsc.xai import DataType
from xai4tsc.xai import explain as xai_explain
from xai4tsc.xai.explain import (
    EXPLAINERS,
    _get_explanation,
    build_explainer,
    generate_explanation,
    register_explainer,
)


class _DummyModel:
    """Stand-in model: only ``eval()`` is called before index selection."""

    def eval(self):
        return self


def _patch_get_explanation(monkeypatch):
    """Patch _get_explanation to record orig_indices and return a sentinel."""
    recorded = {}

    def _recorder(**kwargs):
        recorded["orig_indices"] = kwargs["orig_indices"]
        return "SENTINEL"

    monkeypatch.setattr(xai_explain, "_get_explanation", _recorder)
    return recorded


# ── index selection ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_indices_none_samples_positive_shuffles_and_truncates(monkeypatch):
    recorded = _patch_get_explanation(monkeypatch)
    data = np.zeros((10, 1, 4), dtype=np.float32)
    labels = np.arange(10)

    result = generate_explanation(
        method="integrated_gradients",
        model=_DummyModel(),
        data=data,
        labels=labels,
        indices=None,
        samples=3,
        prng=np.random.default_rng(0),
    )

    assert result == "SENTINEL"
    chosen = recorded["orig_indices"]
    assert len(chosen) == 3
    assert set(chosen).issubset(set(range(10)))


@pytest.mark.unit
def test_empty_indices_samples_positive_selects_random(monkeypatch):
    recorded = _patch_get_explanation(monkeypatch)
    data = np.zeros((8, 1, 4), dtype=np.float32)
    labels = np.arange(8)

    generate_explanation(
        method="integrated_gradients",
        model=_DummyModel(),
        data=data,
        labels=labels,
        indices=[],
        samples=2,
        prng=np.random.default_rng(1),
    )

    assert len(recorded["orig_indices"]) == 2


@pytest.mark.unit
def test_empty_indices_nonpositive_samples_returns_none(monkeypatch, caplog):
    called = {"hit": False}

    def _should_not_run(**_kwargs):
        called["hit"] = True
        return "SENTINEL"

    monkeypatch.setattr(xai_explain, "_get_explanation", _should_not_run)

    with caplog.at_level("WARNING", logger="xai4tsc.xai.explain"):
        result = generate_explanation(
            method="integrated_gradients",
            model=_DummyModel(),
            data=np.zeros((5, 1, 4), dtype=np.float32),
            labels=np.arange(5),
            indices=[],
            samples=0,
        )

    assert result is None
    assert called["hit"] is False
    assert "aborting" in caplog.text.lower()


# ── explainer validation & registry ──────────────────────────────────────────────


@pytest.mark.unit
def test_get_explanation_unknown_method_raises_notimplemented():
    with pytest.raises(NotImplementedError, match="not supported"):
        _get_explanation(
            method="definitely_not_a_method",
            model=_DummyModel(),
            data=np.zeros((1, 1, 4), dtype=np.float32),
            labels=np.array([0]),
            encoder=None,
            params={},
        )


@pytest.mark.unit
def test_register_explainer_adds_to_registry():
    from xai4tsc.xai.base import ExplainerBase

    class _Custom(ExplainerBase):
        explanation_type = "feature_attribution"

        def explain(self, model, exp, device, targets, **kwargs):
            return None

    register_explainer("DummyXAI", _Custom)
    try:
        # Stored case-insensitively (lowercased key).
        assert EXPLAINERS["dummyxai"] is _Custom
    finally:
        EXPLAINERS.pop("dummyxai", None)


@pytest.mark.unit
def test_register_explainer_rejects_non_subclass():
    class _NotAnExplainer:
        pass

    with pytest.raises(TypeError, match="ExplainerBase subclass"):
        register_explainer("BadXAI", _NotAnExplainer)
    assert "badxai" not in EXPLAINERS


# ── data_applicability attribute ──────────────────────────────────────────────────

# Some explainers need a config to build: wrappers (e.g. "sign") need a base, and
# the time-frequency methods need an STFT transform (no sensible default window).
# These are excluded from the param-free instance test; their class-level
# data_applicability is still covered by test_data_applicability_class_level below.
_CONFIG_REQUIRING = {"sign", "freqrise", "timefrequency", "random_timefrequency"}
_DIRECT_EXPLAINERS = [k for k in EXPLAINERS if k not in _CONFIG_REQUIRING]


@pytest.mark.unit
@pytest.mark.parametrize("method", _DIRECT_EXPLAINERS)
def test_data_applicability_is_nonempty_set_of_datatype(method):
    inst = build_explainer(method, None)
    assert isinstance(inst.data_applicability, set)
    assert inst.data_applicability
    assert all(isinstance(d, DataType) for d in inst.data_applicability)


@pytest.mark.unit
@pytest.mark.parametrize("method", sorted(EXPLAINERS))
def test_data_applicability_class_level(method):
    # data_applicability is a ClassVar, so it is valid without instantiation —
    # this covers config-requiring explainers (wrappers, time-frequency) too.
    cls = EXPLAINERS[method]
    assert isinstance(cls.data_applicability, set)
    assert cls.data_applicability
    assert all(isinstance(d, DataType) for d in cls.data_applicability)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("integrated_gradients", {DataType.AGNOSTIC}),
        ("guided_backpropagation", {DataType.AGNOSTIC}),
        ("deconvolution", {DataType.AGNOSTIC}),
        ("deeplift", {DataType.AGNOSTIC}),
        ("occlusion", {DataType.AGNOSTIC}),
        ("tshap", {DataType.TIME_SERIES}),
    ],
)
def test_data_applicability_values(method, expected):
    assert build_explainer(method, None).data_applicability == expected


@pytest.mark.unit
def test_sign_wrapper_inherits_base_data_applicability():
    sign = build_explainer("sign", {"base": {"method": "deconvolution"}})
    assert (
        sign.data_applicability == sign._base.data_applicability == {DataType.AGNOSTIC}
    )
