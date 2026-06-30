"""
Unit tests for experiment_runner/explain.py adapter — no model, no torch.

The heavy ``generate_explanation`` call is monkeypatched with a recorder that
returns a prebuilt :class:`Explanation`, so these tests exercise only the
runner adapter's config unpacking, sample-selection branches, and artifact
saving.
"""

import json

import numpy as np
import pytest
from experiment_runner import explain as runner_explain

from xai4tsc.xai._types import Explanation

# ── wrapper display name ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_display_name_plain_method():
    assert (
        runner_explain._explainer_display_name({"method": "integrated_gradients"})
        == "integrated_gradients"
    )


@pytest.mark.unit
def test_display_name_wrapper_appends_base():
    cfg = {"method": "SIGN", "base": {"method": "Integrated_Gradients"}}
    assert runner_explain._explainer_display_name(cfg) == "SIGN-Integrated_Gradients"


class _OkEncoder:
    """Encoder whose inverse_transform succeeds."""

    def inverse_transform(self, labels):
        return np.asarray([f"class_{int(v)}" for v in np.asarray(labels).ravel()])


class _BadEncoder:
    """Encoder whose inverse_transform raises (exercises the fallback path)."""

    def inverse_transform(self, labels):
        raise ValueError("cannot decode")


def _make_explanation(encoder):
    return Explanation(
        explainer="integrated_gradients",
        explanation_type="feature_attribution",
        exp_values=np.zeros((2, 1, 4), dtype=np.float32),
        data=np.zeros((2, 1, 4), dtype=np.float32),
        labels=np.array([0, 1]),
        indices=np.array([0, 2]),
        encoder=encoder,
        meta=None,
    )


def _patch_recorder(monkeypatch, return_value):
    """Patch generate_explanation to record kwargs and return *return_value*."""
    calls = {}

    def _recorder(**kwargs):
        calls.update(kwargs)
        return return_value

    monkeypatch.setattr(runner_explain, "generate_explanation", _recorder)
    return calls


def _explainer_config(sample_type, **extra):
    cfg = {
        "method": "integrated_gradients",
        "target": "predicted",
        "samples": {"type": sample_type},
    }
    cfg.update(extra)
    return cfg


# ── sample-selection branches ───────────────────────────────────────────────────


@pytest.mark.unit
def test_by_index_branch_passes_indices(tmp_path, monkeypatch):
    calls = _patch_recorder(monkeypatch, _make_explanation(_OkEncoder()))
    cfg = _explainer_config("by_index")
    cfg["samples"]["indices"] = [0, 2]

    runner_explain._generate_explanation(
        model=object(),
        data=np.zeros((4, 1, 4), dtype=np.float32),
        labels=np.array([0, 1, 0, 1]),
        encoder=_OkEncoder(),
        config={"results_rel_path": tmp_path},
        explainer_config=cfg,
    )

    assert calls["indices"] == [0, 2]
    assert "samples" not in calls  # by_index does not forward a random count
    assert (
        calls["save_path"]
        == tmp_path / "explanations" / "explanations_integrated_gradients"
    )


@pytest.mark.unit
def test_wrapper_save_folder_uses_qualified_name(tmp_path, monkeypatch):
    calls = _patch_recorder(monkeypatch, _make_explanation(_OkEncoder()))
    cfg = {
        "method": "SIGN",
        "target": "predicted",
        "samples": {"type": "by_index", "indices": [0, 2]},
        "base": {"method": "Integrated_Gradients"},
    }

    runner_explain._generate_explanation(
        model=object(),
        data=np.zeros((4, 1, 4), dtype=np.float32),
        labels=np.array([0, 1, 0, 1]),
        encoder=_OkEncoder(),
        config={"results_rel_path": tmp_path},
        explainer_config=cfg,
    )

    assert (
        calls["save_path"]
        == tmp_path / "explanations" / "explanations_SIGN-Integrated_Gradients"
    )


@pytest.mark.unit
def test_random_branch_passes_count(tmp_path, monkeypatch):
    calls = _patch_recorder(monkeypatch, _make_explanation(_OkEncoder()))
    cfg = _explainer_config("random")
    cfg["samples"]["count"] = 3

    runner_explain._generate_explanation(
        model=object(),
        data=np.zeros((4, 1, 4), dtype=np.float32),
        labels=np.array([0, 1, 0, 1]),
        encoder=_OkEncoder(),
        config={"results_rel_path": tmp_path},
        explainer_config=cfg,
    )

    assert calls["indices"] is None
    assert calls["samples"] == 3


# ── artifact saving ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_save_exp_values_writes_artifacts(tmp_path, monkeypatch):
    _patch_recorder(monkeypatch, _make_explanation(_OkEncoder()))
    cfg = _explainer_config("by_index", save_exp_values=True)
    cfg["samples"]["indices"] = [0, 2]

    runner_explain._generate_explanation(
        model=object(),
        data=np.zeros((4, 1, 4), dtype=np.float32),
        labels=np.array([0, 1, 0, 1]),
        encoder=_OkEncoder(),
        config={"results_rel_path": tmp_path},
        explainer_config=cfg,
    )

    save_to = tmp_path / "explanations" / "explanations_integrated_gradients"
    assert (save_to / "exp_values.npy").exists()
    assert (save_to / "indices.npy").exists()
    labels = json.loads((save_to / "labels.json").read_text())
    assert labels == ["class_0", "class_1"]  # decoded via the encoder


@pytest.mark.unit
def test_inverse_transform_fallback_on_exception(tmp_path, monkeypatch):
    _patch_recorder(monkeypatch, _make_explanation(_BadEncoder()))
    cfg = _explainer_config("by_index", save_exp_values=True)
    cfg["samples"]["indices"] = [0, 2]

    runner_explain._generate_explanation(
        model=object(),
        data=np.zeros((4, 1, 4), dtype=np.float32),
        labels=np.array([0, 1, 0, 1]),
        encoder=_BadEncoder(),
        config={"results_rel_path": tmp_path},
        explainer_config=cfg,
    )

    save_to = tmp_path / "explanations" / "explanations_integrated_gradients"
    labels = json.loads((save_to / "labels.json").read_text())
    assert labels == [0, 1]  # fell back to raw labels when decoding failed


@pytest.mark.unit
def test_save_path_overrides_config_base(tmp_path, monkeypatch):
    calls = _patch_recorder(monkeypatch, _make_explanation(_OkEncoder()))
    cfg = _explainer_config("by_index")
    cfg["samples"]["indices"] = [0, 2]
    alt = tmp_path / "alt"

    runner_explain._generate_explanation(
        model=object(),
        data=np.zeros((4, 1, 4), dtype=np.float32),
        labels=np.array([0, 1, 0, 1]),
        encoder=_OkEncoder(),
        config={"results_rel_path": tmp_path},
        explainer_config=cfg,
        save_path=alt,
    )

    assert (
        calls["save_path"] == alt / "explanations" / "explanations_integrated_gradients"
    )
