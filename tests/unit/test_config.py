"""Unit tests for experiment_runner/config.py."""

from pathlib import Path

import pytest
from experiment_runner.config import (
    _merge_warn_drop,
    config_resolve,
    config_sanity_check,
)

_RUNNER = Path(__file__).parent.parent.parent / "experiment_runner"
MASTER_CONFIG = _RUNNER / "configs" / "master.yaml"


# ── _merge_warn_drop ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_merge_warn_drop_user_overrides_master():
    master = {"lr": 0.001, "epochs": 10}
    user = {"lr": 0.01}
    result = _merge_warn_drop(user, master, "test")
    assert result["lr"] == 0.01
    assert result["epochs"] == 10


@pytest.mark.unit
def test_merge_warn_drop_unknown_key_dropped(caplog):
    import logging

    master = {"lr": 0.001}
    user = {"lr": 0.01, "unknown_key": 99}
    # caplog captures via the root handler; "xai4tsc" has propagate=False after
    # _setup_console_logging runs in integration tests, so temporarily re-enable
    # propagation so caplog can see messages from "xai4tsc.runner.config".
    xai4tsc_logger = logging.getLogger("xai4tsc")
    prev_propagate = xai4tsc_logger.propagate
    xai4tsc_logger.propagate = True
    try:
        with caplog.at_level("WARNING"):
            result = _merge_warn_drop(user, master, "test")
    finally:
        xai4tsc_logger.propagate = prev_propagate
    assert "unknown_key" not in result
    assert any("unknown_key" in msg for msg in caplog.messages)


@pytest.mark.unit
def test_merge_warn_drop_nested_recursive():
    master = {"hp": {"lr": 0.001, "epochs": 10}}
    user = {"hp": {"lr": 0.01}}
    result = _merge_warn_drop(user, master, "test")
    assert result["hp"]["lr"] == 0.01
    assert result["hp"]["epochs"] == 10


@pytest.mark.unit
def test_merge_warn_drop_master_only_key_preserved():
    master = {"lr": 0.001, "momentum": 0.9}
    user = {"lr": 0.01}
    result = _merge_warn_drop(user, master, "test")
    assert result["momentum"] == 0.9


# ── config_resolve ────────────────────────────────────────────────────────────


def _minimal_user_config():
    return {
        "experiment_name": "unit_test",
        "general": {
            "reproducible": True,
            "seed": 42,
            "device": "cpu",
        },
        "data_config": {"datasets": []},
        "train_config": {"models": []},
        "explanation_config": {"explainers": []},
        "evaluation_config": {"metrics": []},
        "results_rel_path": "./results",
    }


@pytest.mark.unit
def test_config_resolve_has_all_top_level_keys():
    config = config_resolve(_minimal_user_config(), MASTER_CONFIG)
    for key in (
        "experiment_name",
        "general",
        "data_config",
        "train_config",
        "explanation_config",
        "evaluation_config",
        "results_rel_path",
    ):
        assert key in config


@pytest.mark.unit
def test_config_resolve_drops_unknown_top_level_key(caplog):
    import logging

    user = _minimal_user_config()
    user["totally_unknown"] = "value"
    xai4tsc_logger = logging.getLogger("xai4tsc")
    prev_propagate = xai4tsc_logger.propagate
    xai4tsc_logger.propagate = True
    try:
        with caplog.at_level("WARNING"):
            config = config_resolve(user, MASTER_CONFIG)
    finally:
        xai4tsc_logger.propagate = prev_propagate
    assert "totally_unknown" not in config


@pytest.mark.unit
def test_config_resolve_occlusion_defaults_applied():
    user = _minimal_user_config()
    user["explanation_config"]["explainers"] = [{"method": "Occlusion"}]
    config = config_resolve(user, MASTER_CONFIG)
    explainer = config["explanation_config"]["explainers"][0]
    assert "strides" in explainer
    assert "window_shape" in explainer


@pytest.mark.unit
def test_config_resolve_occlusion_user_value_preserved():
    user = _minimal_user_config()
    user["explanation_config"]["explainers"] = [{"method": "Occlusion", "strides": 5}]
    config = config_resolve(user, MASTER_CONFIG)
    explainer = config["explanation_config"]["explainers"][0]
    assert explainer["strides"] == 5


@pytest.mark.unit
def test_config_resolve_no_internal_keys_in_output():
    """Resolved config must not expose any internal master-only keys."""
    config = config_resolve(_minimal_user_config(), MASTER_CONFIG)
    assert "_method_defaults" not in config.get("explanation_config", {})
    assert "_method_defaults" not in config.get("evaluation_config", {})


# ── config_sanity_check ───────────────────────────────────────────────────────


def _valid_config():
    cfg = _minimal_user_config()
    cfg["train_config"]["models"] = [
        {"model": "FCN", "init_params": {"in_channels": 1}}
    ]
    cfg["explanation_config"]["explainers"] = [{"method": "Integrated_Gradients"}]
    return cfg


@pytest.mark.unit
def test_config_sanity_check_valid_returns_true():
    cfg = config_resolve(_valid_config(), MASTER_CONFIG)
    assert config_sanity_check(cfg) is True


@pytest.mark.unit
def test_config_sanity_check_bad_model_raises():
    cfg = _valid_config()
    cfg["train_config"]["models"] = [{"model": "NonExistentModel9999"}]
    cfg = config_resolve(cfg, MASTER_CONFIG)
    with pytest.raises(ValueError, match="NonExistentModel9999"):
        config_sanity_check(cfg)


@pytest.mark.unit
def test_config_sanity_check_bad_explainer_raises():
    cfg = _valid_config()
    cfg["explanation_config"]["explainers"] = [{"method": "NonExistentExplainer9999"}]
    cfg = config_resolve(cfg, MASTER_CONFIG)
    with pytest.raises(ValueError, match=r"(?i)nonexistentexplainer9999"):
        config_sanity_check(cfg)


# ── external-component fallback ───────────────────────────────────────────────

_EXT_SOURCE = """
from xai4tsc.models.base import ModelBase
from xai4tsc.xai.base import ExplainerBase
from xai4tsc.evaluation.base import EvaluatorBase


class ExtModel(ModelBase):
    def forward(self, x):
        return x


class ExtExplainer(ExplainerBase):
    explanation_type = "feature_attribution"

    def explain(self, model, exp, device, targets, **kwargs):
        return None


class ExtMetric(EvaluatorBase):
    def evaluate(self, model, explanation, data, labels, device="cpu", **kwargs):
        return 0.0
"""


@pytest.fixture
def ext_source(tmp_path):
    """Write a source file defining external model/explainer/metric classes."""
    path = tmp_path / "ext_components.py"
    path.write_text(_EXT_SOURCE)
    return path


@pytest.fixture
def clean_registries():
    """Snapshot and restore the runtime registries around a test."""
    from xai4tsc.evaluation.evaluate import METRICS
    from xai4tsc.models.models import MODELS
    from xai4tsc.xai.explain import EXPLAINERS

    snaps = [(reg, dict(reg)) for reg in (MODELS, EXPLAINERS, METRICS)]
    yield
    for reg, snap in snaps:
        reg.clear()
        reg.update(snap)


@pytest.mark.unit
def test_sanity_check_loads_external_model(ext_source, clean_registries):
    from xai4tsc.models.models import MODELS

    cfg = _minimal_user_config()
    cfg["general"]["allow_external_code"] = True
    cfg["train_config"]["models"] = [
        {
            "model": "ExtModelName",
            "class_name": "ExtModel",
            "class_path": str(ext_source),
        }
    ]
    cfg = config_resolve(cfg, MASTER_CONFIG)
    assert config_sanity_check(cfg) is True
    # Registered under the lower-cased name (models are looked up case-insensitively)
    assert "extmodelname" in MODELS


@pytest.mark.unit
def test_sanity_check_loads_external_explainer_and_metric(ext_source, clean_registries):
    from xai4tsc.evaluation.evaluate import METRICS
    from xai4tsc.xai.explain import EXPLAINERS

    cfg = _minimal_user_config()
    cfg["general"]["allow_external_code"] = True
    cfg["explanation_config"]["explainers"] = [
        {
            "method": "ExtExpl",
            "class_name": "ExtExplainer",
            "class_path": str(ext_source),
        }
    ]
    cfg["evaluation_config"]["metrics"] = [
        {"metric": "ExtMet", "class_name": "ExtMetric", "class_path": str(ext_source)}
    ]
    cfg = config_resolve(cfg, MASTER_CONFIG)
    assert config_sanity_check(cfg) is True
    assert "extexpl" in EXPLAINERS  # explainers are registered lower-cased
    assert "ExtMet" in METRICS  # metric keys are exact display names


@pytest.mark.unit
def test_sanity_check_external_blocked_when_flag_false(ext_source, clean_registries):
    from xai4tsc.models.models import MODELS

    cfg = _minimal_user_config()  # allow_external_code defaults to False
    cfg["train_config"]["models"] = [
        {
            "model": "ExtModelName",
            "class_name": "ExtModel",
            "class_path": str(ext_source),
        }
    ]
    cfg = config_resolve(cfg, MASTER_CONFIG)
    with pytest.raises(ValueError, match="ExtModelName"):
        config_sanity_check(cfg)
    assert "extmodelname" not in MODELS


@pytest.mark.unit
def test_sanity_check_external_missing_class_raises(ext_source, clean_registries):
    cfg = _minimal_user_config()
    cfg["general"]["allow_external_code"] = True
    cfg["train_config"]["models"] = [
        {"model": "Ext", "class_name": "NotThere", "class_path": str(ext_source)}
    ]
    cfg = config_resolve(cfg, MASTER_CONFIG)
    with pytest.raises(ImportError, match="NotThere"):
        config_sanity_check(cfg)


@pytest.mark.unit
def test_sanity_check_external_wrong_base_raises(ext_source, clean_registries):
    # ExtMetric is not a ModelBase subclass, so registering it as a model fails.
    cfg = _minimal_user_config()
    cfg["general"]["allow_external_code"] = True
    cfg["train_config"]["models"] = [
        {"model": "Ext", "class_name": "ExtMetric", "class_path": str(ext_source)}
    ]
    cfg = config_resolve(cfg, MASTER_CONFIG)
    with pytest.raises(TypeError, match="ModelBase subclass"):
        config_sanity_check(cfg)
