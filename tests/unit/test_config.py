"""Unit tests for experiment_runner/config.py."""

import inspect
import re
from pathlib import Path

import pytest
import yaml
from experiment_runner.config import (
    ConfigIssues,
    _classify_params,
    _merge_warn_drop,
    _normalize_train_switch,
    _report_param_block,
    _validate_entry,
    config_resolve,
    config_sanity_check,
    result_identities,
    strip_runner_keys,
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
    section = config["explanation_config"]
    explainer = section["explainers"][0]
    # Resolve overlays default_settings onto every entry, and nothing beyond it:
    # none of Occlusion's own constructor parameters may arrive from master's
    # catalogue entry, since those defaults belong to the class.
    assert set(explainer) == {"method"} | set(section["default_settings"]), (
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
@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("loss_func", "FocalLoss", "Unknown loss_func"),
        ("optimizer", "lion", "Unknown optimizer"),
        ("loss_params", {"nonsense": 1}, "could not be constructed"),
    ],
)
def test_config_sanity_check_bad_training_lookup_raises(key, value, expected):
    # A name the runner cannot resolve has to fail here, before a dataset is
    # downloaded and a model instantiated, not part-way into a training run.
    cfg = _valid_config()
    cfg["train_config"]["models"] = [{"model": "FCN", "hyperparams": {key: value}}]
    cfg = config_resolve(cfg, MASTER_CONFIG)
    with pytest.raises(ValueError, match=expected):
        config_sanity_check(cfg)


@pytest.mark.unit
def test_config_sanity_check_accepts_a_dotted_loss_path():
    cfg = _valid_config()
    cfg["train_config"]["models"] = [
        {
            "model": "FCN",
            "hyperparams": {
                "loss_func": "torch.nn.BCEWithLogitsLoss",
                "optimizer": "torch.optim.NAdam",
            },
        }
    ]
    cfg = config_resolve(cfg, MASTER_CONFIG)
    assert config_sanity_check(cfg) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("section", "mapping", "key"),
    [
        ("evaluation_config", "metric_class_params", "nr_samples"),
        ("train_config", "hyperparams.loss_params", "pos_weight"),
    ],
)
def test_a_free_form_mapping_keeps_the_keys_master_cannot_list(section, mapping, key):
    # master cannot enumerate the parameters of ~37 Quantus classes or of an
    # arbitrary torch loss, so filtering these two against it dropped whatever it
    # did not happen to list - silently, bar a warning. Measured before the fix:
    # metric_class_params: {nr_samples: 7} resolved to master's five keys alone.
    cfg = _minimal_user_config()
    block = cfg[section].setdefault("default_settings", {})
    for part in mapping.split(".")[:-1]:
        block = block.setdefault(part, {})
    block[mapping.split(".")[-1]] = {key: [1.0, 2.0] if "weight" in key else 7}

    resolved = config_resolve(cfg, MASTER_CONFIG)[section]["default_settings"]
    for part in mapping.split("."):
        resolved = resolved[part]
    assert key in resolved, f"{mapping}.{key} was dropped"


@pytest.mark.unit
def test_master_still_defaults_the_keys_it_does_list_in_a_free_form_mapping():
    # Taking the user's mapping whole must not discard master's own entries.
    cfg = _minimal_user_config()
    cfg["evaluation_config"]["default_settings"] = {
        "metric_class_params": {"nr_samples": 7}
    }
    resolved = config_resolve(cfg, MASTER_CONFIG)
    params = resolved["evaluation_config"]["default_settings"]["metric_class_params"]
    assert params["nr_samples"] == 7
    assert "normalise" in params, "master's declared defaults must survive"


@pytest.mark.unit
def test_an_unknown_key_outside_a_free_form_mapping_is_still_dropped():
    cfg = _minimal_user_config()
    cfg["train_config"]["default_settings"] = {"hyperparams": {"epocs": 5}}
    resolved = config_resolve(cfg, MASTER_CONFIG)
    assert "epocs" not in resolved["train_config"]["default_settings"]["hyperparams"]


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
    from experiment_runner.lookups import EXPLAINERS, METRICS, MODELS

    snaps = [(reg, dict(reg)) for reg in (MODELS, EXPLAINERS, METRICS)]
    yield
    for reg, snap in snaps:
        reg.clear()
        reg.update(snap)


@pytest.mark.unit
def test_sanity_check_loads_external_model(ext_source, clean_registries):
    from experiment_runner.lookups import MODELS

    import xai4tsc.models as package

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
    # Added under the lower-cased name (models are looked up case-insensitively)
    assert "extmodelname" in MODELS
    # The runner's table, not the package catalog, holds it.
    assert "extmodelname" not in package.MODELS


@pytest.mark.unit
def test_sanity_check_loads_external_explainer_and_metric(ext_source, clean_registries):
    from experiment_runner.lookups import EXPLAINERS, METRICS

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
    assert "extexpl" in EXPLAINERS  # explainers are added lower-cased
    assert "ExtMet" in METRICS  # metric keys are exact display names


@pytest.mark.unit
def test_sanity_check_external_blocked_when_flag_false(ext_source, clean_registries):
    from experiment_runner.lookups import MODELS

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


# ── result_identities ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_result_identities_empty():
    assert result_identities([])[0] == []


@pytest.mark.unit
def test_result_identities_all_unique_names_pass_through_unchanged():
    # The common case must be byte-identical to pre-identity behaviour, so that
    # existing configs keep writing to exactly the same paths.
    names = ["Complexity", "Pixel-Flipping", "Sparseness"]
    assert result_identities(names)[0] == names


@pytest.mark.unit
def test_result_identities_suffixes_only_the_duplicates():
    identities, _ = result_identities(["TSHAP", "TSHAP", "SIGN"])
    assert identities == ["TSHAP_1", "TSHAP_2", "SIGN"]


@pytest.mark.unit
def test_result_identities_three_way_duplicate():
    assert result_identities(["X", "X", "X"])[0] == ["X_1", "X_2", "X_3"]


@pytest.mark.unit
def test_result_identities_interleaved_groups_count_independently():
    # Each name's counter is its own, and positions are preserved.
    assert result_identities(["A", "B", "A", "C", "B"])[0] == [
        "A_1",
        "B_1",
        "A_2",
        "C",
        "B_2",
    ]


@pytest.mark.unit
def test_result_identities_preserves_input_length_and_order():
    names = ["m", "n", "m", "m", "o"]
    out = result_identities(names)[0]
    assert len(out) == len(names)
    # Every identity still starts with the name it came from.
    assert all(ident.startswith(name) for ident, name in zip(out, names, strict=True))


@pytest.mark.unit
def test_result_identities_warns_when_a_generated_suffix_hits_a_literal_name(caplog):
    # Two "X" entries generate X_1/X_2; a literal "X_1" then collides with the
    # generated one. The run stays usable, but the user is told.
    with caplog.at_level("WARNING", logger="xai4tsc.runner.config"):
        out = result_identities(["X", "X", "X_1"])[0]
    assert out == ["X_1", "X_2", "X_1"]
    assert "not unique after disambiguation" in caplog.text
    assert "X_1" in caplog.text


@pytest.mark.unit
def test_result_identities_occurrences_are_stable_when_a_duplicate_is_added():
    """The half the derived seeds key on; the identity half is not stable here."""
    assert result_identities(["A"]) == (["A"], [1])
    assert result_identities(["A", "A"]) == (["A_1", "A_2"], [1, 2])


@pytest.mark.unit
def test_result_identities_does_not_warn_for_a_clean_result(caplog):
    with caplog.at_level("WARNING", logger="xai4tsc.runner.config"):
        result_identities(["A", "A", "B"])[0]
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

    assert caplog.text.count("this list is empty") == 4


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
    """
    Every hyperparameter master documents is a real ``train_model`` keyword.

    The values themselves are compared in ``test_defaults.py``, against the
    signature the runner derives them from. What this adds is coverage of the
    other direction: a key master lists that training would reject.

    The two exceptions are the mappings the runner consumes while building the
    loss and the optimizer; they are config keys without being parameters, and
    ``test_defaults.py`` pins that from the other side.
    """
    from xai4tsc.models.base import ModelBase

    accepted = set(inspect.signature(ModelBase.train_model).parameters)
    consumed = {"loss_params", "optimizer_params"}
    master = yaml.safe_load(MASTER_CONFIG.read_text())
    for key in master["train_config"]["default_settings"]["hyperparams"]:
        assert key in accepted or key in consumed, (
            f"master documents hyperparams.{key}, which train_model does not take"
        )


@pytest.mark.unit
def test_master_is_not_runnable():
    master = yaml.safe_load(MASTER_CONFIG.read_text())
    assert master["experiment_name"] is None
    assert master["results_rel_path"] is None


_ENTRY_LINE = re.compile(r"^\s*- (method|model|dataset): (\S.*?)\s*(?:#.*)?$")
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
    every ``#   key: value`` line under a model, explainer or dataset entry and
    compares it to that class's constructor default.
    """
    from xai4tsc.data.datasets import SYNTHETIC_DATASETS
    from xai4tsc.models.models import MODELS
    from xai4tsc.xai.explain import EXPLAINERS

    registries = {
        "method": EXPLAINERS,
        "model": MODELS,
        "dataset": SYNTHETIC_DATASETS,
    }
    current: tuple[str, dict] | None = None
    problems = []

    for line in MASTER_CONFIG.read_text().splitlines():
        entry = _ENTRY_LINE.match(line)
        if _LIST_ITEM.match(line) and not entry:
            current = None  # a metric entry, or anything else we do not introspect
            continue
        if entry:
            kind, name = entry.group(1), entry.group(2).strip().strip('"')
            registry = registries[kind]
            # Synthetic dataset keys are case-sensitive; model/explainer keys are
            # lowercased. A non-synthetic dataset (UCR/UEA, local) has no class to
            # introspect and falls through to `current = None`.
            cls = (
                registry.get(name) if kind == "dataset" else registry.get(name.lower())
            )
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


# ── entry validation: names raise, ignored keys warn (R6, R7, R20) ────────────


def _entry_config(**sections):
    """Return a valid config with one entry per level, overridable per test."""
    cfg = _minimal_user_config()
    cfg["data_config"]["datasets"] = [{"dataset": "GunPoint"}]
    cfg["train_config"]["models"] = [{"model": "FCN"}]
    cfg["explanation_config"]["explainers"] = [{"method": "Integrated_Gradients"}]
    cfg["evaluation_config"]["metrics"] = [{"metric": "Complexity"}]
    for section, entries in sections.items():
        key = {
            "data_config": "datasets",
            "train_config": "models",
            "explanation_config": "explainers",
            "evaluation_config": "metrics",
        }[section]
        cfg[section][key] = entries
    return config_resolve(cfg, MASTER_CONFIG)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("section", "entries", "where"),
    [
        (
            "data_config",
            [{"dataset": "GunPoint", "bogus": 1}],
            "datasets[0] (GunPoint)",
        ),
        ("train_config", [{"model": "FCN", "bogus": 1}], "models[0] (FCN)"),
        (
            "explanation_config",
            [{"method": "Integrated_Gradients", "bogus": 1}],
            "explainers[0] (Integrated_Gradients)",
        ),
        (
            "evaluation_config",
            [{"metric": "Complexity", "bogus": 1}],
            "metrics[0] (Complexity)",
        ),
    ],
)
def test_unknown_entry_key_warns_naming_its_location(section, entries, where, caplog):
    cfg = _entry_config(**{section: entries})

    with caplog.at_level("WARNING", logger=_CFG_LOGGER):
        assert config_sanity_check(cfg) is True

    assert "'bogus'" in caplog.text
    assert where in caplog.text


@pytest.mark.unit
def test_typo_in_an_explainer_parameter_suggests_the_real_name(caplog):
    cfg = _entry_config(
        explanation_config=[{"method": "Integrated_Gradients", "nsteps": 3}]
    )

    with caplog.at_level("WARNING", logger=_CFG_LOGGER):
        config_sanity_check(cfg)

    assert "Did you mean 'n_steps'?" in caplog.text


@pytest.mark.unit
def test_metric_parameter_at_top_level_names_where_it_belongs(caplog):
    """R20: the user did everything but nest it, so say where it goes."""
    cfg = _entry_config(
        evaluation_config=[{"metric": "Pixel-Flipping", "features_in_step": 1}]
    )

    with caplog.at_level("WARNING", logger=_CFG_LOGGER):
        config_sanity_check(cfg)

    assert "'metric_class_params'" in caplog.text
    assert "features_in_step" in caplog.text


@pytest.mark.unit
def test_model_parameter_at_top_level_names_init_params(caplog):
    cfg = _entry_config(train_config=[{"model": "FCN", "filters": [8, 16, 8]}])

    with caplog.at_level("WARNING", logger=_CFG_LOGGER):
        config_sanity_check(cfg)

    assert "'init_params'" in caplog.text


@pytest.mark.unit
def test_unregistered_metric_name_raises():
    """R6: a misspelled metric used to warn and silently shorten metrics.csv."""
    cfg = _entry_config(evaluation_config=[{"metric": "Complexety"}])

    with pytest.raises(ValueError) as excinfo:
        config_sanity_check(cfg)

    assert "Did you mean 'Complexity'?" in str(excinfo.value)


@pytest.mark.unit
def test_metric_name_case_mismatch_names_the_registered_spelling():
    cfg = _entry_config(evaluation_config=[{"metric": "complexity"}])

    with pytest.raises(ValueError, match=r"registered spelling is 'Complexity'"):
        config_sanity_check(cfg)


@pytest.mark.unit
def test_nested_base_entry_is_validated_too():
    cfg = _entry_config(
        explanation_config=[
            {"method": "SIGN", "base": {"method": "NotAnExplainer9999"}},
        ]
    )

    with pytest.raises(ValueError, match="NotAnExplainer9999"):
        config_sanity_check(cfg)


@pytest.mark.unit
def test_default_settings_keys_and_runner_keys_are_accepted(caplog):
    """An entry may override any default_settings key, plus its structural ones."""
    cfg = _entry_config(
        data_config=[
            {"dataset": "UEA", "max_samples": 5, "overrides": {"X": {"max_samples": 1}}}
        ],
        explanation_config=[
            {"method": "Occlusion", "target": "label", "strides": 2},
        ],
        evaluation_config=[
            {"metric": "Complexity", "animate": True, "animate_fps": 3},
        ],
    )

    with caplog.at_level("WARNING", logger=_CFG_LOGGER):
        assert config_sanity_check(cfg) is True

    assert "not a known" not in caplog.text
    assert "not read at" not in caplog.text


@pytest.mark.unit
def test_all_problems_are_reported_in_one_run(caplog):
    cfg = _entry_config(
        train_config=[{"model": "FCN", "bogus_model_key": 1}],
        explanation_config=[{"method": "Integrated_Gradients", "nsteps": 3}],
        evaluation_config=[{"metric": "Nope9999"}, {"metric": "Alsonope9999"}],
    )

    with (
        caplog.at_level("WARNING", logger=_CFG_LOGGER),
        pytest.raises(ValueError) as excinfo,
    ):
        config_sanity_check(cfg)

    # Both errors in one message...
    assert "Nope9999" in str(excinfo.value)
    assert "Alsonope9999" in str(excinfo.value)
    # ...and the warnings still logged, even though the run then failed.
    assert "bogus_model_key" in caplog.text
    assert "nsteps" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_file",
    # master.yaml is a reference rather than a runnable config, and is refused
    # on purpose - test_master_config_is_refused covers that.
    sorted(p for p in (_RUNNER / "configs").glob("*.yaml") if p.name != "master.yaml"),
    ids=lambda p: p.name,
)
def test_shipped_configs_raise_no_entry_warnings(config_file, caplog):
    """The false-positive guard: a shipped config must validate silently."""
    cfg = config_resolve(yaml.safe_load(config_file.read_text()), MASTER_CONFIG)
    with caplog.at_level("WARNING", logger=_CFG_LOGGER):
        assert config_sanity_check(cfg) is True

    assert "not a known" not in caplog.text, caplog.text
    assert "not read at" not in caplog.text, caplog.text


# ── one collected report ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_settings_and_entry_problems_share_one_report(caplog):
    """Every config check appends to one collector, so the count is honest."""
    issues = ConfigIssues()
    user = _minimal_user_config()
    user["bogus_top_level"] = 1
    user["xai4tsc_version"] = ">=0"
    user["general"]["bogus_general"] = 1
    user["explanation_config"]["explainers"] = [
        {"method": "Integrated_Gradients", "nsteps": 3}
    ]
    resolved = config_resolve(user, MASTER_CONFIG, issues=issues)
    config_sanity_check(resolved, issues=issues)

    # Nothing logged yet: the caller decides when to report.
    assert not caplog.text
    assert len(issues.warnings) == 3

    with caplog.at_level("INFO", logger=_CFG_LOGGER):
        issues.report()

    assert "0 error(s), 3 warning(s)" in caplog.text
    for expected in ("bogus_top_level", "bogus_general", "nsteps"):
        assert expected in caplog.text


@pytest.mark.unit
def test_dropped_setting_suggests_the_real_key(caplog):
    user = _minimal_user_config()
    user["data_config"]["default_settings"] = {"trainsplit": 0.5}

    with caplog.at_level("WARNING", logger=_CFG_LOGGER):
        config_resolve(user, MASTER_CONFIG)

    assert "Did you mean 'train_split'?" in caplog.text


@pytest.mark.unit
def test_repeat_restates_warnings_without_raising(caplog):
    issues = ConfigIssues()
    issues.warn("somewhere", "a setting was ignored")
    issues.error("elsewhere", "fatal")

    with caplog.at_level("INFO", logger=_CFG_LOGGER):
        issues.report(repeat=True)

    assert "repeated (1)" in caplog.text
    assert "a setting was ignored" in caplog.text
    assert "fatal" not in caplog.text


@pytest.mark.unit
def test_report_is_silent_when_nothing_was_collected(caplog):
    with caplog.at_level("INFO", logger=_CFG_LOGGER):
        ConfigIssues().report()
    assert caplog.text == ""


@pytest.mark.unit
def test_channel_first_is_a_settable_call_param():
    # Master is also the key whitelist, so a user can only set channel_first if
    # master declares it; dropping it there would silently ignore the override and
    # hand the axis order back to Quantus to guess.
    defaults = config_resolve(_minimal_user_config(), MASTER_CONFIG)[
        "evaluation_config"
    ]["default_settings"]["metric_call_params"]
    assert defaults["channel_first"] is True

    user = _minimal_user_config()
    user["evaluation_config"]["default_settings"] = {
        "metric_call_params": {"channel_first": False}
    }
    overridden = config_resolve(user, MASTER_CONFIG)["evaluation_config"][
        "default_settings"
    ]["metric_call_params"]
    assert overridden["channel_first"] is False


# ── general.seed / general.reproducible types ─────────────────────────────────


@pytest.mark.unit
def test_non_integer_general_seed_is_an_error():
    """A quoted seed would otherwise raise deep inside numpy instead of here."""
    cfg = _valid_config()
    cfg["general"] = {"seed": "2025"}
    cfg = config_resolve(cfg, MASTER_CONFIG)
    with pytest.raises(ValueError, match=r"general\.seed"):
        config_sanity_check(cfg)


@pytest.mark.unit
def test_boolean_general_seed_is_an_error():
    """`seed: true` is a mistake, not seed 1 - bool is an int subclass."""
    cfg = _valid_config()
    cfg["general"] = {"seed": True}
    cfg = config_resolve(cfg, MASTER_CONFIG)
    with pytest.raises(ValueError, match=r"general\.seed"):
        config_sanity_check(cfg)


@pytest.mark.unit
def test_non_boolean_reproducible_is_an_error():
    """A quoted "false" is a non-empty string, so it would be read as true."""
    cfg = _valid_config()
    cfg["general"] = {"reproducible": "false"}
    cfg = config_resolve(cfg, MASTER_CONFIG)
    with pytest.raises(ValueError, match=r"general\.reproducible"):
        config_sanity_check(cfg)


@pytest.mark.unit
def test_null_general_seed_is_accepted():
    """`None` is the documented way to ask for an unseeded run."""
    cfg = _valid_config()
    cfg["general"] = {"seed": None}
    cfg = config_resolve(cfg, MASTER_CONFIG)
    assert config_sanity_check(cfg) is True


# ── xai4tsc_version ───────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spec", "errors", "warnings"),
    [
        (">=2.0.0.dev0,<3", 0, 0),  # pre-release admitted by order
        (">=2.0.0,<3", 1, 0),  # the dev build sorts before 2.0.0
        ("2.0.0.dev", 0, 0),  # bare version means ==
        ("2.0.0", 1, 0),
        (">=3", 1, 0),
        ("two", 1, 0),
        (2.1, 1, 0),  # unquoted YAML number
        (None, 0, 1),
    ],
)
def test_framework_version(monkeypatch, spec, errors, warnings):
    """The installed xai4tsc must fit the config's specifier; none only warns."""
    import xai4tsc

    monkeypatch.setattr(xai4tsc, "__version__", "2.0.0.dev")
    cfg = _valid_config()
    cfg["xai4tsc_version"] = spec
    cfg = config_resolve(cfg, MASTER_CONFIG)
    issues = ConfigIssues()
    config_sanity_check(cfg, issues)
    version_errors = [e for e in issues.errors if e.startswith("xai4tsc_version")]
    version_warnings = [w for w in issues.warnings if w.startswith("xai4tsc_version")]
    assert (len(version_errors), len(version_warnings)) == (errors, warnings)


# ── split fractions and sample selection ──────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"train_split": 0.9, "val_split": 0.5}, "leaves nothing for the test"),
        ({"train_split": 0.9, "val_split": 0.1}, "leaves nothing for the test"),
        ({"train_split": 1.5}, "train_split must be between 0 and 1"),
        ({"val_split": -0.2}, "val_split cannot be negative"),
    ],
)
def test_impossible_split_fractions_are_refused(entry, expected):
    """
    Fractions that leave no test split must fail at config time.

    They used to reach sklearn as a negative test_size, which the splitter caught
    and reported as a stratification problem before raising again - so the error
    named the wrong cause, after the dataset had already been loaded.
    """
    cfg = _entry_config(data_config=[{"dataset": "GunPoint", **entry}])

    with pytest.raises(ValueError, match=expected):
        config_sanity_check(cfg)


@pytest.mark.unit
def test_split_fractions_are_not_checked_for_the_archive_partition():
    # use_predefined_splits never reaches the splitter, so its fractions are
    # ignored rather than impossible; split() reports that separately.
    cfg = _entry_config(
        data_config=[
            {
                "dataset": "GunPoint",
                "train_split": 0.9,
                "val_split": 0.5,
                "use_predefined_splits": True,
            }
        ]
    )

    assert config_sanity_check(cfg) is True


@pytest.mark.unit
def test_unknown_sample_type_is_refused_naming_the_cost():
    """
    A typo in samples.type used to silently explain the whole test split.

    Metrics are computed only on the explained samples, so that quietly changed
    which subset every number in the run was scored over.
    """
    cfg = _entry_config(
        explanation_config=[
            {"method": "Integrated_Gradients", "samples": {"type": "indx"}}
        ]
    )

    with pytest.raises(ValueError) as excinfo:
        config_sanity_check(cfg)
    message = str(excinfo.value)
    assert "indx" in message
    assert "'index'" in message, "should suggest the near-miss"
    assert "whole test split" in message


@pytest.mark.unit
def test_unknown_visualization_type_is_refused_before_training():
    cfg = _entry_config(
        explanation_config=[
            {"method": "Integrated_Gradients", "visualization_type": ["pionts"]}
        ]
    )

    with pytest.raises(ValueError, match=r"visualization_type.*'pionts'"):
        config_sanity_check(cfg)


@pytest.mark.unit
def test_a_single_visualization_type_string_is_checked_as_one_style():
    ok = _entry_config(
        explanation_config=[
            {"method": "Integrated_Gradients", "visualization_type": "bar"}
        ]
    )
    assert config_sanity_check(ok) is True

    bad = _entry_config(
        explanation_config=[
            {"method": "Integrated_Gradients", "visualization_type": "pionts"}
        ]
    )
    with pytest.raises(ValueError, match=r"got 'pionts'"):
        config_sanity_check(bad)


@pytest.mark.unit
@pytest.mark.parametrize("sample_type", ["index", "random", "all"])
def test_documented_sample_types_are_accepted(sample_type):
    samples = {"type": sample_type, "indices": [0], "count": 3}
    cfg = _entry_config(
        explanation_config=[{"method": "Integrated_Gradients", "samples": samples}]
    )

    assert config_sanity_check(cfg) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ({"type": "index", "indices": []}, "non-empty list"),
        ({"type": "index", "indices": [0, -3]}, "non-negative integers"),
        ({"type": "index", "indices": [0, "two"]}, "non-negative integers"),
        ({"type": "random", "count": 0}, "positive integer"),
    ],
)
def test_malformed_sample_selection_is_refused(samples, expected):
    # The bound itself needs the test split, which config validation cannot see;
    # these are the parts that are checkable without any data.
    cfg = _entry_config(
        explanation_config=[{"method": "Integrated_Gradients", "samples": samples}]
    )

    with pytest.raises(ValueError, match=expected):
        config_sanity_check(cfg)


@pytest.mark.unit
def test_split_and_sample_problems_share_one_report(caplog):
    # Both new checks must go through the collector, so one run reports
    # everything rather than failing on whichever is reached first.
    cfg = _entry_config(
        data_config=[{"dataset": "GunPoint", "train_split": 0.9, "val_split": 0.5}],
        explanation_config=[
            {"method": "Integrated_Gradients", "samples": {"type": "indx"}}
        ],
    )

    with caplog.at_level("ERROR", logger=_CFG_LOGGER), pytest.raises(ValueError):
        config_sanity_check(cfg)

    assert "leaves nothing for the test split" in caplog.text
    assert "indx" in caplog.text


# ── unverifiable parameters and general.allow_kwargs ─────────────────────────


def _named_only(a, b=1, *, c=2):
    """Stand in for a target that declares no catch-all."""


def _catch_all(a, b=1, **kwargs):
    """Stand in for named parameters beside a catch-all, as Quantus metrics have."""


@pytest.mark.unit
def test_classify_params_splits_the_three_buckets():
    unverifiable, unknown, reserved = _classify_params(
        {"a": 1, "stray": 2}, _named_only
    )
    assert (unverifiable, unknown, reserved) == ([], ["stray"], [])

    unverifiable, unknown, reserved = _classify_params({"a": 1, "stray": 2}, _catch_all)
    assert (unverifiable, unknown, reserved) == (["stray"], [], [])


@pytest.mark.unit
def test_a_named_parameter_beside_a_catch_all_is_verified():
    """
    The oracle is named parameters, not the presence of **kwargs.

    Quantus metrics declare both, so conflating them would reject
    `disable_warnings` and break every shipped config.
    """
    assert _classify_params({"a": 1, "b": 2}, _catch_all) == ([], [], [])


@pytest.mark.unit
def test_owned_names_are_reserved():
    _, _, reserved = _classify_params({"a": 1}, _catch_all, frozenset({"a"}))
    assert reserved == ["a"]


@pytest.mark.unit
@pytest.mark.parametrize("allow", [False, True])
def test_allow_kwargs_gates_only_the_unverifiable_bucket(allow):
    issues = ConfigIssues()
    _report_param_block(
        {"a": 1, "stray": 2}, _catch_all, "where", "the target", allow, issues
    )

    assert issues.warnings == []
    if allow:
        assert issues.errors == []
    else:
        assert len(issues.errors) == 1
        assert "cannot be verified" in issues.errors[0]
        assert "general.allow_kwargs" in issues.errors[0]


@pytest.mark.unit
@pytest.mark.parametrize("allow", [False, True])
def test_the_flag_never_silences_a_verifiable_mistake(allow):
    """A narrowed signature stays checked however the flag is set."""
    issues = ConfigIssues()
    _report_param_block(
        {"stray": 2}, _named_only, "where", "the target", allow, issues, dropped=False
    )
    assert len(issues.errors) == 1
    assert "is not a parameter of" in issues.errors[0]


@pytest.mark.unit
def test_unknown_is_a_warning_when_the_value_is_filtered_out():
    """Dropped before the call means the run proceeds on a default, not a crash."""
    issues = ConfigIssues()
    _report_param_block(
        {"stray": 2}, _named_only, "where", "the target", False, issues, dropped=True
    )
    assert not issues.errors
    assert len(issues.warnings) == 1


@pytest.mark.unit
def test_a_non_mapping_param_block_is_an_error():
    issues = ConfigIssues()
    _report_param_block(["stray", 2], _catch_all, "where", "the target", True, issues)
    assert len(issues.errors) == 1
    assert "expected a mapping" in issues.errors[0]


@pytest.mark.unit
def test_reserved_call_params_key_is_rejected():
    """`device` is already explain()'s third positional; setting it would raise."""
    from xai4tsc.xai.explain import EXPLAINERS

    entry = {"method": "Integrated_Gradients", "call_params": {"device": "cuda"}}
    issues = ConfigIssues()
    _validate_entry(entry, "method", "e[0]", {}, EXPLAINERS, issues, True)

    assert any("supplied by the framework" in e for e in issues.errors)


@pytest.mark.unit
@pytest.mark.parametrize("allow", [False, True])
def test_call_params_on_a_catch_all_explain_needs_the_flag(allow):
    from xai4tsc.xai.explain import EXPLAINERS

    entry = {"method": "Integrated_Gradients", "call_params": {"scale": 3}}
    issues = ConfigIssues()
    _validate_entry(entry, "method", "e[0]", {}, EXPLAINERS, issues, allow)

    assert bool(issues.errors) is (not allow)


@pytest.mark.unit
@pytest.mark.parametrize("allow", [False, True])
def test_a_catch_all_explainer_constructor_needs_the_flag(allow):
    """The pre-existing hole: dict_to_args hands a VAR_KEYWORD __init__ everything."""
    from xai4tsc.xai.base import ExplainerBase

    class _KwargsInit(ExplainerBase):
        explanation_type = "feature_attribution"

        def __init__(self, **kwargs):
            self.seen = kwargs

        def explain(self, model, exp, device, targets, **kwargs):
            return None

    issues = ConfigIssues()
    _validate_entry(
        {"method": "K", "foo": 7},
        "method",
        "e[0]",
        {},
        {"k": _KwargsInit},
        issues,
        allow,
    )

    assert bool(issues.errors) is (not allow)
    assert not issues.warnings, "a key that does arrive must not be called unknown"


@pytest.mark.unit
def test_metric_params_are_checked_against_the_quantus_class():
    """Not against QuantusEvaluator, which declares **kwargs and verifies nothing."""
    from xai4tsc.evaluation.evaluate import METRICS

    issues = ConfigIssues()
    _validate_entry(
        {"metric": "Complexity", "metric_class_params": {"disable_warnings": True}},
        "metric",
        "m[0]",
        {"metric_class_params": {}},
        METRICS,
        issues,
        False,
    )
    assert not issues.errors, "disable_warnings is a named Quantus parameter"

    issues = ConfigIssues()
    _validate_entry(
        {"metric": "Complexity", "metric_class_params": {"disable_warnigns": True}},
        "metric",
        "m[0]",
        {"metric_class_params": {}},
        METRICS,
        issues,
        False,
    )
    assert any("cannot be verified" in e for e in issues.errors)


@pytest.mark.unit
def test_model_init_params_are_checked():
    from xai4tsc.models.models import MODELS

    issues = ConfigIssues()
    _validate_entry(
        {"model": "FCN", "init_params": {"in_channels": 1, "nope": 2}},
        "model",
        "m[0]",
        {"init_params": {}},
        MODELS,
        issues,
        False,
    )
    assert any("'nope'" in w for w in issues.warnings + issues.errors)


@pytest.mark.unit
def test_dataset_init_params_are_checked_against_the_synthetic_class():
    issues = ConfigIssues()
    _validate_entry(
        {"dataset": "Synth_Multilabel_Freq", "init_params": {"nope": 2}},
        "dataset",
        "d[0]",
        {"init_params": {}},
        None,
        issues,
        False,
    )
    assert any("'nope'" in e for e in issues.errors), (
        "a synthetic dataset is splatted straight into its constructor"
    )


@pytest.mark.unit
@pytest.mark.parametrize("cfg_path", sorted(_RUNNER.glob("configs/*.yaml")))
def test_every_shipped_config_is_free_of_unverifiable_parameters(cfg_path):
    """
    The guard that keeps this honest as components are added.

    Every shipped config must pass with the flag at its default, so a new
    component whose signature ends in **kwargs cannot quietly make one of them
    require `allow_kwargs: true`.
    """
    cfg = config_resolve(yaml.safe_load(cfg_path.read_text()), MASTER_CONFIG)
    cfg.setdefault("general", {})["allow_kwargs"] = False
    issues = ConfigIssues()
    from experiment_runner.config import _validate_entries

    _validate_entries(cfg, issues)
    assert not issues.errors, f"{cfg_path.name}: {issues.errors}"


# ── the runner's own entry keys never reach a constructor ─────────────────────


@pytest.mark.unit
def test_strip_runner_keys_removes_the_structural_keys():
    cfg = config_resolve(
        yaml.safe_load((_RUNNER / "configs/example.yaml").read_text()), MASTER_CONFIG
    )
    entry = dict(cfg["explanation_config"]["explainers"][0], seed=12345)

    stripped = strip_runner_keys(entry, entry["method"], cfg)

    assert set(stripped) == {"seed"}, (
        "only the derived seed is a constructor parameter here"
    )


@pytest.mark.unit
def test_strip_runner_keys_keeps_a_key_the_class_declares():
    """An explainer may legitimately name a parameter the runner also uses."""
    from experiment_runner.lookups import EXPLAINERS

    from xai4tsc.xai.base import ExplainerBase

    class _TakesSamples(ExplainerBase):
        explanation_type = "feature_attribution"

        def __init__(self, samples=None):
            self.samples = samples

        def explain(self, model, exp, device, targets, **kwargs):
            return None

    EXPLAINERS["takessamples"] = _TakesSamples
    try:
        cfg = {"explanation_config": {"default_settings": {"samples": {}}}}
        stripped = strip_runner_keys(
            {"method": "TakesSamples", "samples": {"type": "index"}},
            "TakesSamples",
            cfg,
        )
    finally:
        EXPLAINERS.pop("takessamples", None)

    assert "samples" in stripped
    assert "method" not in stripped


@pytest.mark.unit
def test_strip_runner_keys_descends_into_a_wrapper_base():
    cfg = {"explanation_config": {"default_settings": {"target": None}}}
    stripped = strip_runner_keys(
        {
            "method": "SIGN",
            "target": "predicted",
            "seed": 7,
            "base": {
                "method": "Integrated_Gradients",
                "target": "predicted",
                "n_steps": 8,
            },
        },
        "SIGN",
        cfg,
    )

    assert stripped == {
        "seed": 7,
        "base": {"method": "Integrated_Gradients", "n_steps": 8},
    }, "a wrapper pops `method` off its base to build it, so that one stays"


@pytest.mark.unit
def test_a_catch_all_constructor_receives_only_configured_parameters():
    """
    What R57 was: `dict_to_args` hands a VAR_KEYWORD constructor the whole entry.

    Without the strip this explainer would be built with `method`, `target` and
    `samples` as if they had been configured for it.
    """
    from experiment_runner.explain import build_explainer
    from experiment_runner.lookups import EXPLAINERS

    from xai4tsc.xai.base import ExplainerBase

    class _KwargsInit(ExplainerBase):
        explanation_type = "feature_attribution"

        def __init__(self, **kwargs):
            self.seen = kwargs

        def explain(self, model, exp, device, targets, **kwargs):
            return None

    EXPLAINERS["kwargsinit"] = _KwargsInit
    try:
        entry = {
            "method": "KwargsInit",
            "target": "predicted",
            "samples": {"type": "index"},
            "save_exp_values": False,
            "my_option": 7,
        }
        cfg = {
            "explanation_config": {
                "default_settings": {
                    "target": None,
                    "samples": None,
                    "save_exp_values": None,
                }
            }
        }
        instance = build_explainer(
            "KwargsInit", strip_runner_keys(entry, "KwargsInit", cfg)
        )
    finally:
        EXPLAINERS.pop("kwargsinit", None)

    assert instance.seen == {"my_option": 7}


# ── softmax over independent per-output heads ─────────────────────────────────


@pytest.mark.unit
def test_softmax_over_a_per_output_formulation_warns(caplog):
    # metric_call_params.softmax renormalises the model output across the row
    # before a perturbation metric reads the explained class's score. Over
    # independent per-label heads that turns one label's confidence into a
    # function of the others and moves every perturbation score, with nothing
    # about the numbers looking wrong.
    cfg = _valid_config()
    cfg["data_config"]["datasets"] = [
        {"dataset": "Synth_Multilabel_Freq", "encode": "multihot"}
    ]
    cfg.setdefault("evaluation_config", {}).setdefault("default_settings", {})[
        "metric_call_params"
    ] = {"softmax": True}
    cfg = config_resolve(cfg, MASTER_CONFIG)

    with caplog.at_level("WARNING", logger="xai4tsc"):
        assert config_sanity_check(cfg) is True

    assert "Synth_Multilabel_Freq" in caplog.text
    assert "independent per-label" in caplog.text


@pytest.mark.unit
def test_softmax_is_not_questioned_for_a_single_label_formulation(caplog):
    cfg = _valid_config()
    cfg.setdefault("evaluation_config", {}).setdefault("default_settings", {})[
        "metric_call_params"
    ] = {"softmax": True}
    cfg = config_resolve(cfg, MASTER_CONFIG)

    with caplog.at_level("WARNING", logger="xai4tsc"):
        config_sanity_check(cfg)

    assert "independent per-label" not in caplog.text
