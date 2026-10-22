"""
Unit tests for xai4tsc/xai/explain.py sample selection and registry — no torch.

The heavy ``_get_explanation`` call is monkeypatched with a recorder that
captures the indices it receives, so the index-selection logic of
``generate_explanation`` is tested without a real model or attribution run.
"""

import numpy as np
import pytest
import torch

from xai4tsc.xai import DataType, Explanation
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
        n_samples=3,
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
        n_samples=2,
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
            n_samples=0,
        )

    assert result is None
    assert called["hit"] is False
    assert "aborting" in caplog.text.lower()


# ── explainer validation & registry ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("bad", [[0, 999999], [-1], [10]])
def test_out_of_range_indices_raise_naming_the_range(monkeypatch, bad):
    """
    An index past the end of the data must say so, not reach numpy.

    It used to surface as ``IndexError: index 999999 is out of bounds for axis 0``
    from ``data[indices]``, which names an axis rather than the setting that
    produced it - and in a run, only after the model had already trained.
    """
    _patch_get_explanation(monkeypatch)
    data = np.zeros((10, 1, 4), dtype=np.float32)
    labels = np.arange(10)

    with pytest.raises(ValueError, match=r"out of range for 10 samples"):
        generate_explanation(
            method="integrated_gradients",
            model=_DummyModel(),
            data=data,
            labels=labels,
            indices=bad,
            prng=np.random.default_rng(0),
        )


@pytest.mark.unit
def test_in_range_indices_are_passed_through(monkeypatch):
    # The boundary index must not be rejected: valid is 0..len-1.
    recorded = _patch_get_explanation(monkeypatch)
    data = np.zeros((10, 1, 4), dtype=np.float32)

    generate_explanation(
        method="integrated_gradients",
        model=_DummyModel(),
        data=data,
        labels=np.arange(10),
        indices=[0, 9],
        prng=np.random.default_rng(0),
    )

    assert list(recorded["orig_indices"]) == [0, 9]


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


# ── targets under a per-output (2-D) label encoding ───────────────────────────

# A 2-D target is one column per model output, where a prediction is a SET of
# labels. So all three modes stack one attribution per output and differ only in
# the mask saying which apply. Before this, "label" argmaxed the multi-hot row
# and explained only its first active output, and "predicted" argmaxed a softmax
# over outputs that do not compete - both silent.


class _StubEncoder:
    """Minimal fitted-encoder stand-in: ``classes_`` is all the dispatch reads."""

    def __init__(self, n_outputs=3):
        self.classes_ = np.arange(n_outputs)


class _FixedProbs(torch.nn.Module):
    """Per-output logits chosen so the thresholded prediction is known."""

    def __init__(self, logits):
        super().__init__()
        self.output_activation = "sigmoid"
        self._logits = torch.tensor(logits, dtype=torch.float32)
        self.unused = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self._logits[: x.shape[0]]


def _register_recording_explainer(monkeypatch):
    """Register a stub explainer that records the targets it is handed."""
    from xai4tsc.xai.base import ExplainerBase

    recorded = {"targets": []}

    class _Recorder(ExplainerBase):
        explanation_type = "feature_attribution"

        def explain(self, model, exp, device, targets, **kwargs):
            recorded["targets"].append(list(targets))
            return np.zeros((len(exp.data), 1, 4), dtype=np.float32)

    register_explainer("_RecorderXAI", _Recorder)
    monkeypatch.setattr(
        xai_explain, "plot_exp", lambda *args, **kwargs: None, raising=False
    )
    return recorded


def _explain(monkeypatch, labels, targets, model=None, **kwargs):
    recorded = _register_recording_explainer(monkeypatch)
    try:
        exp = _get_explanation(
            method="_RecorderXAI",
            model=model if model is not None else _DummyModel(),
            data=np.zeros((len(labels), 1, 4), dtype=np.float32),
            labels=np.asarray(labels),
            encoder=_StubEncoder(np.shape(labels)[1] if np.ndim(labels) > 1 else 3),
            params={},
            targets=targets,
            **kwargs,
        )
    finally:
        EXPLAINERS.pop("_recorderxai", None)
    return exp, recorded


@pytest.mark.unit
def test_label_targets_stack_every_output_and_mask_the_ones_present():
    # The stack is rectangular - one attribution per (output, sample) - and the
    # mask says which of them mean anything. A ragged array could not hold it,
    # and flattening would break the rule that a row lines up with `indices`.
    labels = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0]])
    exp, _ = _explain(pytest.MonkeyPatch(), labels, "label")

    assert exp.exp_values.shape == (3, 4, 1, 4)
    np.testing.assert_array_equal(exp.target_mask, labels.astype(bool).T)
    # Sample 3 carries nothing, so no output is scored on it.
    assert not exp.target_mask[:, 3].any()
    # Output 0 applies to samples 0 and 2.
    np.testing.assert_array_equal(
        exp.rows_for_target(0), np.array([True, False, True, False])
    )


@pytest.mark.unit
def test_predicted_targets_mask_by_the_per_output_threshold():
    # Independent sigmoids, not a competitive argmax: a sample can be explained
    # for several outputs or for none.
    labels = np.array([[1, 0, 0], [0, 1, 1]])
    model = _FixedProbs([[5.0, -5.0, 5.0], [-5.0, -5.0, -5.0]])

    exp, _ = _explain(pytest.MonkeyPatch(), labels, "predicted", model=model)

    # Row 0: outputs 0 and 2 over 0.5. Row 1: none of them.
    np.testing.assert_array_equal(
        exp.target_mask, np.array([[True, False], [False, False], [True, False]])
    )


@pytest.mark.unit
def test_the_threshold_decides_which_outputs_are_explained():
    labels = np.array([[1, 1, 1]])
    # sigmoid(1.0) ~ 0.73, so a 0.9 threshold excludes it and a 0.5 keeps it.
    model = _FixedProbs([[1.0, 1.0, 1.0]])

    lenient, _ = _explain(
        pytest.MonkeyPatch(), labels, "predicted", model=model, threshold=0.5
    )
    strict, _ = _explain(
        pytest.MonkeyPatch(), labels, "predicted", model=model, threshold=0.9
    )

    assert lenient.target_mask.all()
    assert not strict.target_mask.any()


@pytest.mark.unit
def test_all_targets_leave_every_row_applicable():
    # Under "all" there is nothing to mask: every output is explained for every
    # sample, which is what rows_for_target returns without a mask.
    labels = np.array([[1, 0, 0], [0, 1, 0]])
    exp, _ = _explain(pytest.MonkeyPatch(), labels, "all")

    assert exp.target_mask is None
    assert exp.rows_for_target(1).all()


@pytest.mark.unit
def test_every_output_is_explained_as_its_own_captum_target():
    # Captum selects output unit j with target=j, so the stack is built by
    # asking for each one in turn rather than by reinterpreting one attribution.
    labels = np.array([[1, 1, 0], [0, 0, 1]])
    _, recorded = _explain(pytest.MonkeyPatch(), labels, "label")

    assert recorded["targets"] == [[0, 0], [1, 1], [2, 2]]


@pytest.mark.unit
def test_a_mask_that_does_not_match_the_stack_raises():
    # The mask is only useful while it lines up with the stack; drifted, it
    # would score one output's attributions against another's samples.
    exp = Explanation(
        explainer="x",
        explainer_params={},
        explanation_type="feature_attribution",
        exp_values=np.zeros((3, 4, 1, 8), dtype=np.float32),
        data=np.zeros((4, 1, 8), dtype=np.float32),
        labels=np.zeros((4, 3), dtype=np.int64),
        indices=np.arange(4),
        encoder=_StubEncoder(),
        meta=None,
    )
    exp.target_mask = np.ones((3, 5), dtype=bool)
    with pytest.raises(ValueError, match="target_mask has shape"):
        exp.validate_target_mask()


@pytest.mark.unit
def test_label_targets_pass_1d_class_indices_through_unchanged():
    # The regression guard: the single-label path is what every shipped config
    # uses and it must be untouched by any of the above - one attribution per
    # sample, no stack, no mask.
    exp, recorded = _explain(pytest.MonkeyPatch(), np.array([2, 0, 1]), "label")

    assert recorded["targets"] == [[2, 0, 1]]
    assert exp.exp_values.shape == (3, 1, 4)
    assert exp.target_mask is None
    assert exp.explained_targets.tolist() == [2, 0, 1]
