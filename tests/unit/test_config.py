"""Unit tests for experiment_runner/config.py."""

import inspect
import re
from pathlib import Path

import pytest
import yaml
from experiment_runner.config import (
    _merge_warn_drop,
    _normalize_train_switch,
    config_resolve,
    config_sanity_check,
    disambiguate,
    validate_user_config,
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
def test_config_resolve_injects_no_constructor_params():
    """Parameter defaults belong to the class, so resolve must not supply them."""
    user = _minimal_user_config()
    user["explanation_config"]["explainers"] = [{"method": "Occlusion"}]
    config = config_resolve(user, MASTER_CONFIG)
    explainer = config["explanation_config"]["explainers"][0]
    assert explainer == {"method": "Occlusion"}, (
        "master.yaml must not override the explainer's own defaults"
    )


@pytest.mark.unit
def test_config_resolve_does_not_inherit_master_component_lists():
    """An empty list stays empty instead of inheriting master's catalogue."""
    user = _minimal_user_config()
    config = config_resolve(user, MASTER_CONFIG)
    assert config["evaluation_config"]["metrics"] == []
    assert config["explanation_config"]["explainers"] == []
    assert config["data_config"]["datasets"] == []
    assert config["train_config"]["models"] == []


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


# ── disambiguate ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_disambiguate_empty():
    assert disambiguate([]) == []


@pytest.mark.unit
def test_disambiguate_all_unique_names_pass_through_unchanged():
    # The common case must be byte-identical to pre-identity behaviour, so that
    # existing configs keep writing to exactly the same paths.
    names = ["Complexity", "Pixel-Flipping", "Sparseness"]
    assert disambiguate(names) == names


@pytest.mark.unit
def test_disambiguate_suffixes_only_the_duplicates():
    assert disambiguate(["TSHAP", "TSHAP", "SIGN"]) == ["TSHAP_1", "TSHAP_2", "SIGN"]


@pytest.mark.unit
def test_disambiguate_three_way_duplicate():
    assert disambiguate(["X", "X", "X"]) == ["X_1", "X_2", "X_3"]


@pytest.mark.unit
def test_disambiguate_interleaved_groups_count_independently():
    # Each name's counter is its own, and positions are preserved.
    assert disambiguate(["A", "B", "A", "C", "B"]) == [
        "A_1",
        "B_1",
        "A_2",
        "C",
        "B_2",
    ]


@pytest.mark.unit
def test_disambiguate_preserves_input_length_and_order():
    names = ["m", "n", "m", "m", "o"]
    out = disambiguate(names)
    assert len(out) == len(names)
    # Every identity still starts with the name it came from.
    assert all(ident.startswith(name) for ident, name in zip(out, names, strict=True))


@pytest.mark.unit
def test_disambiguate_warns_when_a_generated_suffix_hits_a_literal_name(caplog):
    # Two "X" entries generate X_1/X_2; a literal "X_1" then collides with the
    # generated one. The run stays usable, but the user is told.
    with caplog.at_level("WARNING", logger="xai4tsc.runner.config"):
        out = disambiguate(["X", "X", "X_1"])
    assert out == ["X_1", "X_2", "X_1"]
    assert "not unique after disambiguation" in caplog.text
    assert "X_1" in caplog.text


@pytest.mark.unit
def test_disambiguate_does_not_warn_for_a_clean_result(caplog):
    with caplog.at_level("WARNING", logger="xai4tsc.runner.config"):
        disambiguate(["A", "A", "B"])
    assert "not unique" not in caplog.text


# ── _normalize_train_switch ───────────────────────────────────────────────────

_CFG_LOGGER = "xai4tsc.runner.config"


@pytest.mark.unit
@pytest.mark.parametrize("legacy", ["till_epoch", "for_epochs", "Till_Epoch"])
def test_legacy_train_values_normalize_to_true(legacy, caplog):
    train_config = {"default_settings": {"train": legacy}, "models": [{"model": "FCN"}]}

    with caplog.at_level("WARNING", logger=_CFG_LOGGER):
        _normalize_train_switch(train_config)

    assert train_config["default_settings"]["train"] is True
    assert "deprecated" in caplog.text


@pytest.mark.unit
def test_legacy_train_value_warns_once_for_several_entries(caplog):
    train_config = {
        "default_settings": {"train": "till_epoch"},
        "models": [
            {"model": "FCN", "train": "for_epochs"},
            {"model": "LeNet", "train": "till_epoch"},
        ],
    }

    with caplog.at_level("WARNING", logger=_CFG_LOGGER):
        _normalize_train_switch(train_config)

    assert all(e["train"] is True for e in train_config["models"])
    assert caplog.text.count("deprecated") == 1


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, False, None])
def test_switch_values_pass_through_untouched(value):
    train_config = {
        "default_settings": {},
        "models": [{"model": "FCN", "train": value}],
    }

    _normalize_train_switch(train_config)

    assert train_config["models"][0]["train"] is value


@pytest.mark.unit
def test_unknown_train_value_raises_naming_the_entry():
    train_config = {
        "default_settings": {},
        "models": [{"model": "FCN", "train": "untill_epoch"}],
    }

    with pytest.raises(ValueError) as excinfo:
        _normalize_train_switch(train_config)

    message = str(excinfo.value)
    assert "untill_epoch" in message
    assert "FCN" in message
    assert "switch" in message


@pytest.mark.unit
def test_missing_train_key_is_left_absent():
    """Absence must stay absence: the runner keys off it to skip training."""
    train_config = {"default_settings": {}, "models": [{"model": "FCN"}]}

    _normalize_train_switch(train_config)

    assert "train" not in train_config["models"][0]


# ── validate_user_config ──────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("data_config", "datasets"),
        ("train_config", "models"),
        ("explanation_config", "explainers"),
        ("evaluation_config", "metrics"),
    ],
)
def test_missing_component_list_raises_naming_it(section, key):
    cfg = _minimal_user_config()
    del cfg[section][key]

    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        validate_user_config(cfg)


@pytest.mark.unit
def test_missing_section_counts_as_a_missing_list():
    cfg = _minimal_user_config()
    del cfg["evaluation_config"]

    with pytest.raises(ValueError, match=r"evaluation_config\.metrics"):
        validate_user_config(cfg)


@pytest.mark.unit
def test_several_missing_settings_are_reported_together():
    """Forgetting three keys should take one run to learn, not three."""
    cfg = _minimal_user_config()
    del cfg["train_config"]["models"]
    del cfg["evaluation_config"]["metrics"]
    cfg["experiment_name"] = None

    with pytest.raises(ValueError) as excinfo:
        validate_user_config(cfg)

    message = str(excinfo.value)
    for expected in (
        "experiment_name",
        "train_config.models",
        "evaluation_config.metrics",
    ):
        assert expected in message


@pytest.mark.unit
@pytest.mark.parametrize("key", ["experiment_name", "results_rel_path"])
@pytest.mark.parametrize("value", [None, "__delete__"])
def test_missing_required_setting_raises(key, value):
    cfg = _minimal_user_config()
    if value == "__delete__":
        del cfg[key]
    else:
        cfg[key] = value

    with pytest.raises(ValueError, match=key):
        validate_user_config(cfg)


@pytest.mark.unit
def test_explicitly_empty_lists_are_allowed_but_warn(caplog):
    """A deliberate partial run stays possible; silence would hide a typo."""
    cfg = _minimal_user_config()

    with caplog.at_level("WARNING", logger=_CFG_LOGGER):
        validate_user_config(cfg)

    assert caplog.text.count("will do nothing") == 4


@pytest.mark.unit
def test_complete_config_is_silent(caplog):
    cfg = _minimal_user_config()
    cfg["data_config"]["datasets"] = [{"dataset": "GunPoint"}]
    cfg["train_config"]["models"] = [{"model": "FCN"}]
    cfg["explanation_config"]["explainers"] = [{"method": "Occlusion"}]
    cfg["evaluation_config"]["metrics"] = [{"metric": "Complexity"}]

    with caplog.at_level("WARNING", logger=_CFG_LOGGER):
        validate_user_config(cfg)

    assert caplog.text == ""


# ── master.yaml never overrides a code default ────────────────────────────────

# Keys master may carry on a list entry that are NOT parameter defaults: either
# the entry's identity, a structural block with no sensible default, or a value
# the runner resolves before the class sees it.
_STRUCTURAL_KEYS = frozenset(
    {"method", "model", "metric", "base", "transform", "background_data"}
)


@pytest.mark.unit
def test_master_list_entries_never_restate_a_code_default():
    """master.yaml documents parameter defaults; it must not supply them."""
    from xai4tsc.models.models import MODELS
    from xai4tsc.xai.explain import EXPLAINERS

    master = yaml.safe_load(MASTER_CONFIG.read_text())
    sections = [
        (master["explanation_config"]["explainers"], "method", EXPLAINERS),
        (master["train_config"]["models"], "model", MODELS),
    ]
    offenders = []
    for entries, id_key, registry in sections:
        for entry in entries:
            name = entry[id_key]
            cls = registry.get(name.lower())
            if cls is None:
                continue
            params = inspect.signature(cls.__init__).parameters
            for key, value in entry.items():
                if key in _STRUCTURAL_KEYS or key not in params:
                    continue
                default = params[key].default
                if default is not inspect.Parameter.empty and default != value:
                    offenders.append(f"{name}.{key}: master={value!r} code={default!r}")
    assert not offenders, "master.yaml overrides code defaults: " + "; ".join(offenders)


@pytest.mark.unit
def test_master_metric_entries_carry_only_their_identity():
    master = yaml.safe_load(MASTER_CONFIG.read_text())
    for entry in master["evaluation_config"]["metrics"]:
        assert set(entry) == {"metric"}, (
            f"metric entry {entry['metric']!r} carries parameters master cannot apply; "
            "document them in comments instead"
        )


@pytest.mark.unit
def test_master_hyperparams_match_the_package_defaults():
    from xai4tsc.utils.defaults import t_con_defaults

    master = yaml.safe_load(MASTER_CONFIG.read_text())
    hyperparams = master["train_config"]["default_settings"]["hyperparams"]
    for key, expected in t_con_defaults.items():
        got = hyperparams[key]
        assert str(got).lower() == str(expected).lower(), (
            f"hyperparams.{key}: master={got!r} but t_con_defaults={expected!r}"
        )


@pytest.mark.unit
def test_master_is_not_runnable():
    master = yaml.safe_load(MASTER_CONFIG.read_text())
    assert master["experiment_name"] is None
    assert master["results_rel_path"] is None


_ENTRY_LINE = re.compile(r"^\s*- (method|model): (\S.*?)\s*(?:#.*)?$")
_LIST_ITEM = re.compile(r"^\s*- \w+:")
# "#   key: value   description" - three or more spaces separate value from prose,
# which a YAML scalar like "[1, 5]" never contains.
_DOC_LINE = re.compile(r"^\s*#\s{2,}(\w+):\s+(.+?)(?:\s{3,}.*)?$")


@pytest.mark.unit
def test_master_documented_defaults_match_the_code():
    """
    The comments are the only record of a default, so they must be true.

    master.yaml no longer *supplies* parameter defaults, which removes the risk of
    it overriding the code - but a stale comment would still misinform. This walks
    every ``#   key: value`` line under a model or explainer entry and compares it
    to that class's constructor default.
    """
    from xai4tsc.models.models import MODELS
    from xai4tsc.xai.explain import EXPLAINERS

    registries = {"method": EXPLAINERS, "model": MODELS}
    current: tuple[str, dict] | None = None
    problems = []

    for line in MASTER_CONFIG.read_text().splitlines():
        entry = _ENTRY_LINE.match(line)
        if _LIST_ITEM.match(line) and not entry:
            current = None  # a metric entry, or anything else we do not introspect
            continue
        if entry:
            kind, name = entry.group(1), entry.group(2).strip().strip('"')
            cls = registries[kind].get(name.lower())
            current = (
                (name, inspect.signature(cls.__init__).parameters) if cls else None
            )
            continue
        if current is None or not line.lstrip().startswith("#"):
            continue
        documented_line = _DOC_LINE.match(line.rstrip())
        if not documented_line:
            continue

        key, raw = documented_line.group(1), documented_line.group(2).strip()
        name, params = current
        if key not in params:
            continue
        default = params[key].default
        if default is inspect.Parameter.empty:
            continue
        if isinstance(default, tuple):
            default = list(default)
        try:
            documented = yaml.safe_load(raw)
        except yaml.YAMLError:
            problems.append(f"{name}.{key}: unparseable documented value {raw!r}")
            continue
        if documented != default:
            problems.append(
                f"{name}.{key}: comment says {documented!r}, code {default!r}"
            )

    assert not problems, "master.yaml documents stale defaults: " + "; ".join(problems)
