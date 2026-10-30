"""
Unit tests for wrapper explainers.

Covers SIGN's correctness identity (``raw_gradient * sign(input - mu)``), its
incompatible-base warnings, and dispatch through ``_get_explanation``.
"""

import logging

import numpy as np
import pytest
import torch
from torch import nn

from xai4tsc.xai import SignExplainer, WrapperExplainer
from xai4tsc.xai._types import Explanation
from xai4tsc.xai.explain import _get_explanation
from xai4tsc.xai.feature_attribution import (
    GuidedBackpropagationExplainer,
    IntegratedGradientsExplainer,
    OcclusionExplainer,
)

_WRAPPERS_LOGGER = "xai4tsc.xai.wrappers"


class _TinyNet(nn.Module):
    """Minimal logit model over flattened ``(C, T)`` input."""

    def __init__(self, channels: int, timesteps: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(channels * timesteps, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


def _make_exp(data: np.ndarray) -> Explanation:
    """Build a minimal Explanation holding *data* for the samples to explain."""
    n = data.shape[0]
    return Explanation(
        explainer="sign",
        explanation_type="feature_attribution",
        exp_values=None,
        data=data,
        labels=np.zeros(n, dtype=np.int64),
        encoder=None,
        indices=np.arange(n),
        meta=None,
    )


# ── SignExplainer construction ────────────────────────────────────────────────


@pytest.mark.unit
def test_sign_wraps_base_and_inherits_explanation_type():
    base = GuidedBackpropagationExplainer()
    sign = SignExplainer(base, mu=0.0)
    assert isinstance(sign, WrapperExplainer)
    assert sign._base is base
    assert sign.explanation_type == sign._base.explanation_type
    assert sign.data_applicability == sign._base.data_applicability
    assert sign.mu == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("base", ["guided_backpropagation", {"method": "deeplift"}])
def test_wrapper_rejects_a_base_that_is_not_an_explainer(base):
    with pytest.raises(TypeError, match="ExplainerBase instance"):
        SignExplainer(base)


# ── SignExplainer correctness identity ────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("mu", [0.0, 0.5, -0.3])
def test_sign_equals_raw_gradient_times_sign(mu):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1, 6)).astype(np.float32)
    exp = _make_exp(data)
    model = _TinyNet(1, 6, 3).eval()
    targets = [0, 1, 2, 0]

    # Raw-gradient base (multiply_by_inputs=False) so SIGN supplies the weighting.
    base = IntegratedGradientsExplainer(multiply_by_inputs=False)
    sign = SignExplainer(base, mu=mu)

    base_rel = base.explain(model, exp, "cpu", targets)
    sign_rel = sign.explain(model, exp, "cpu", targets)

    expected = base_rel * np.where(data < mu, -1.0, 1.0).astype(base_rel.dtype)
    assert sign_rel.shape == base_rel.shape == data.shape
    np.testing.assert_allclose(sign_rel, expected, rtol=1e-5, atol=1e-6)


# ── SignExplainer guards ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_sign_warns_on_non_gradient_base(caplog):
    with caplog.at_level(logging.WARNING, logger=_WRAPPERS_LOGGER):
        SignExplainer(OcclusionExplainer())
    assert "not a GradientExplainer" in caplog.text


@pytest.mark.unit
def test_sign_enforces_multiply_by_inputs_false(caplog):
    base = IntegratedGradientsExplainer(multiply_by_inputs=True)
    with caplog.at_level(logging.WARNING, logger=_WRAPPERS_LOGGER):
        sign = SignExplainer(base)
    # User-supplied True is overridden (with a warning) so the base emits the
    # raw gradient that SIGN re-weights.
    assert "overriding multiply_by_inputs" in caplog.text
    assert sign._base._multiply_by_inputs is False


@pytest.mark.unit
def test_sign_does_not_mutate_caller_base():
    base = IntegratedGradientsExplainer(multiply_by_inputs=True)
    SignExplainer(base)
    # The override happens on a copy, not the caller's explainer.
    assert base._multiply_by_inputs is True


@pytest.mark.unit
def test_sign_no_warning_for_raw_gradient_base(caplog):
    with caplog.at_level(logging.WARNING, logger=_WRAPPERS_LOGGER):
        SignExplainer(GuidedBackpropagationExplainer())
    assert caplog.text == ""


# ── dispatch through _get_explanation ─────────────────────────────────────────


@pytest.mark.unit
def test_sign_dispatch_via_get_explanation_label_targets():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((4, 1, 6)).astype(np.float32)
    model = _TinyNet(1, 6, 3).eval()

    sign = SignExplainer(IntegratedGradientsExplainer(multiply_by_inputs=False))
    exp = _get_explanation(
        explainer=sign,
        model=model,
        data=data,
        labels=np.array([0, 1, 2, 0], dtype=np.int64),
        encoder=None,
        targets="label",
        device="cpu",
        orig_indices=np.arange(4),
    )

    assert exp.explainer is sign
    assert exp.explanation_type == "feature_attribution"
    assert exp.exp_values.shape == data.shape


@pytest.mark.unit
def test_integrated_gradients_rejects_an_unknown_integration_method():
    with pytest.raises(ValueError, match=r"integration_method must be one of \[.*"):
        IntegratedGradientsExplainer(integration_method="simpson")
