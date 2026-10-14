"""Unit tests for the defaults pointer table and the single resolve pass."""

import copy
import inspect
import re
from pathlib import Path

import pytest
import yaml
from experiment_runner.config import config_resolve, config_sanity_check
from experiment_runner.defaults import (
    _PACKAGE_BACKED,
    _RUNNER_OWNED,
    apply_package_defaults,
    default_settings_leaves,
    hyperparam_names,
    package_default,
)
from experiment_runner.utils import expand_datasets

from xai4tsc.data.base import DatasetBase
from xai4tsc.data.datasets import SynthMultilabelFreqDataset, UcrUeaDataset
from xai4tsc.models.base import ModelBase
from xai4tsc.xai.explain import generate_explanation

_RUNNER = Path(__file__).parent.parent.parent / "experiment_runner"
MASTER_CONFIG = _RUNNER / "configs" / "master.yaml"
_SECTIONS = ("data_config", "train_config", "explanation_config", "evaluation_config")


def _master():
    return yaml.safe_load(MASTER_CONFIG.read_text())


def _dig(block, key):
    for part in key.split("."):
        block = block[part]
    return block


def _minimal_user_config(**sections):
    config = {
        "experiment_name": "unit_test",
        "results_rel_path": "./results",
        "data_config": {"datasets": []},
        "train_config": {"models": []},
        "explanation_config": {"explainers": []},
        "evaluation_config": {"metrics": []},
    }
    for section, value in sections.items():
        config[section] = value
    return config


# ── the pointer table ─────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(("section", "key"), sorted(_PACKAGE_BACKED))
def test_package_backed_defaults_match_the_signatures(section, key):
    """
    master.yaml documents these; the package owns them. They must agree.

    A failure here means the two copies have drifted. The fix is to change the
    default in the package and bring master's documentation along, never the
    other way round: the runner reads the signature, so editing master alone
    changes nothing but the documentation.
    """
    derived = package_default(section, key)
    documented = _dig(_master()[section]["default_settings"], key)
    owner, param = _PACKAGE_BACKED[(section, key)]
    assert documented == derived, (
        f"master.yaml says {section}.default_settings.{key} = {documented!r}, but "
        f"{owner.__qualname__} defaults {param} to {derived!r}. Change it in "
        f"{owner.__qualname__} and update master's comment to match."
    )


@pytest.mark.unit
@pytest.mark.parametrize("section", _SECTIONS)
def test_pointer_table_covers_every_default_settings_key(section):
    """
    Every key is either package-backed or runner-owned, with nothing in between.

    An unclassified key is the R18 shape: declared in master, read by nobody, and
    impossible to spot. Adding one now fails here until it is classified.
    """
    block = _master()[section].get("default_settings") or {}
    unclassified = [
        leaf
        for leaf in default_settings_leaves(section, block)
        if (section, leaf) not in _PACKAGE_BACKED
        and (section, leaf) not in _RUNNER_OWNED
    ]
    assert not unclassified, (
        f"{section}.default_settings keys classified as neither package-backed "
        f"nor runner-owned: {unclassified}"
    )


@pytest.mark.unit
def test_the_two_tables_do_not_overlap():
    assert not set(_PACKAGE_BACKED) & _RUNNER_OWNED


@pytest.mark.unit
def test_stale_pointer_raises_naming_the_key_and_callable(monkeypatch):
    """A renamed parameter must fail here, not leave a stale literal behind."""
    monkeypatch.setitem(
        _PACKAGE_BACKED, ("data_config", "train_split"), (ModelBase.train_model, "gone")
    )
    with pytest.raises(KeyError, match="train_split"):
        package_default("data_config", "train_split")


@pytest.mark.unit
def test_unknown_key_is_not_silently_defaulted():
    with pytest.raises(KeyError, match="_RUNNER_OWNED"):
        package_default("data_config", "not_a_setting")


@pytest.mark.unit
def test_hyperparam_names_are_derived_from_train_model():
    accepted = set(inspect.signature(ModelBase.train_model).parameters)
    assert hyperparam_names() <= accepted
    assert "epochs" in hyperparam_names()
    # Not a hyperparameter: the runner supplies it, so an entry must not.
    assert "save_path" not in hyperparam_names()


@pytest.mark.unit
def test_tuple_defaults_arrive_as_lists():
    """A tuple default must not reach the config, or the run log, as a tuple."""
    signature_default = (
        inspect.signature(generate_explanation).parameters["visualization_type"].default
    )
    assert isinstance(signature_default, tuple), "expected an immutable default"
    assert package_default("explanation_config", "visualization_type") == ["points"]


@pytest.mark.unit
def test_visualization_type_default_is_immutable():
    """
    A list default would trip ruff B006, which is why the None sentinel existed.

    A tuple keeps the default introspectable - the whole point of the table - and
    cannot be mutated by one caller for every later one.
    """
    for func in (generate_explanation,):
        default = inspect.signature(func).parameters["visualization_type"].default
        assert isinstance(default, tuple)


# ── the resolve pass ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_resolve_gives_every_entry_the_section_defaults():
    user = _minimal_user_config(
        data_config={"datasets": [{"dataset": "GunPoint"}]},
        explanation_config={"explainers": [{"method": "Integrated_Gradients"}]},
    )
    config = config_resolve(user, MASTER_CONFIG)

    for section, list_key in (
        ("data_config", "datasets"),
        ("explanation_config", "explainers"),
    ):
        block = config[section]
        entry = block[list_key][0]
        missing = set(block["default_settings"]) - set(entry)
        assert not missing, f"{section}.{list_key}[0] is missing {missing}"


@pytest.mark.unit
def test_resolve_leaves_the_user_config_untouched():
    """Resolution must not rewrite the dict it was handed."""
    user = _minimal_user_config(
        data_config={"datasets": [{"dataset": "GunPoint", "encode": "label"}]}
    )
    before = copy.deepcopy(user)
    config_resolve(user, MASTER_CONFIG)
    assert user == before


@pytest.mark.unit
def test_resolved_entries_share_no_nested_objects():
    """
    Two entries inheriting one nested default must not share the object.

    merge_dicts inserts the default's own dict for a key the entry lacks, so
    without a copy on both sides `samples` would be one object shared by the
    defaults block and every entry, and a later write would reach them all.
    """
    user = _minimal_user_config(
        explanation_config={
            "explainers": [{"method": "Integrated_Gradients"}, {"method": "DeepLift"}]
        }
    )
    block = config_resolve(user, MASTER_CONFIG)["explanation_config"]
    first, second = block["explainers"]

    assert first["samples"] is not block["default_settings"]["samples"]
    assert first["samples"] is not second["samples"]

    first["samples"]["count"] = 99
    assert second["samples"]["count"] != 99
    assert block["default_settings"]["samples"]["count"] != 99


@pytest.mark.unit
def test_user_stated_default_wins_over_the_derived_one():
    user = _minimal_user_config(
        data_config={
            "default_settings": {"train_split": 0.55},
            "datasets": [{"dataset": "GunPoint"}],
        }
    )
    config = config_resolve(user, MASTER_CONFIG)
    assert config["data_config"]["datasets"][0]["train_split"] == 0.55
    # and an unstated sibling still comes from the package
    assert config["data_config"]["datasets"][0]["val_split"] == package_default(
        "data_config", "val_split"
    )


@pytest.mark.unit
def test_per_entry_setting_wins_over_the_section_default():
    user = _minimal_user_config(
        data_config={
            "default_settings": {"encode": "label"},
            "datasets": [{"dataset": "Synth_Multilabel_Freq", "encode": "multihot"}],
        }
    )
    config = config_resolve(user, MASTER_CONFIG)
    assert config["data_config"]["datasets"][0]["encode"] == "multihot"


@pytest.mark.unit
def test_derived_default_does_not_overwrite_a_stated_nested_key():
    """Stating one hyperparameter must not cost the caller the others."""
    user = _minimal_user_config(
        train_config={
            "default_settings": {"hyperparams": {"epochs": 3}},
            "models": [{"model": "FCN"}],
        }
    )
    hyperparams = config_resolve(user, MASTER_CONFIG)["train_config"]["models"][0][
        "hyperparams"
    ]
    assert hyperparams["epochs"] == 3
    assert hyperparams["batchsize"] == package_default(
        "train_config", "hyperparams.batchsize"
    )


@pytest.mark.unit
def test_apply_package_defaults_is_idempotent():
    user = _minimal_user_config()
    config = config_resolve(copy.deepcopy(user), MASTER_CONFIG)
    once = copy.deepcopy(config)
    apply_package_defaults(user, config)
    assert config == once


# ── the checkpoint rule, the one stated-vs-inherited dependency ───────────────


@pytest.mark.unit
def test_checkpoint_entry_does_not_train_on_an_inherited_switch(tmp_path):
    """
    `train: true` in default_settings describes the models being trained.

    It must not drag a model that is only being loaded for evaluation into a
    retrain, which is the one decision that needs to tell a stated setting from
    an inherited one.
    """
    ckpt = str(tmp_path / "FCN_epoch_1.pt")
    user = _minimal_user_config(
        train_config={
            "default_settings": {"train": True},
            "models": [
                {"model": "FCN", "model_ckpt_path": ckpt},
                {"model": "FCN", "model_ckpt_path": ckpt, "train": True},
            ],
        }
    )
    models = config_resolve(user, MASTER_CONFIG)["train_config"]["models"]

    assert models[0]["train"] is None, "an inherited switch must not retrain"
    assert models[1]["train"] is True, "the entry asked for it, so it trains"


@pytest.mark.unit
def test_entry_without_a_checkpoint_keeps_the_inherited_switch():
    user = _minimal_user_config(
        train_config={
            "default_settings": {"train": True},
            "models": [{"model": "FCN"}],
        }
    )
    models = config_resolve(user, MASTER_CONFIG)["train_config"]["models"]
    assert models[0]["train"] is True


# ── expansion preserves the resolved-entry invariant ──────────────────────────


@pytest.mark.unit
def test_expansion_preserves_resolved_settings_and_override_precedence():
    """
    Wildcards expand after resolution, so members inherit the resolved base.

    An override still wins, because it is the priority dict of the merge inside
    expand_datasets - which is what makes resolving first safe.
    """
    user = _minimal_user_config(
        data_config={
            "datasets": [
                {
                    "dataset": "UCR",
                    "max_samples": 500,
                    "overrides": {"GunPoint": {"max_samples": 50}},
                }
            ]
        }
    )
    config = config_resolve(user, MASTER_CONFIG)
    members = {
        m["dataset"]: m for m in expand_datasets(config["data_config"]["datasets"])
    }

    inherited = set(config["data_config"]["default_settings"])
    assert inherited <= set(members["GunPoint"]), "a member lost its resolved settings"
    assert members["GunPoint"]["max_samples"] == 50, "the override must win"
    assert members["Coffee"]["max_samples"] == 500, "the entry's own value applies"


# ── master.yaml documents what it declares ────────────────────────────────────


def _described_keys(section):
    """
    Map each ``default_settings`` key of *section* to whether master describes it.

    A key counts as described when a comment sits on its own line or anywhere in
    the block beneath it, which is where a list value (``visualization_type``) and
    a multi-line explanation put theirs.
    """
    lines = MASTER_CONFIG.read_text().splitlines()
    start = lines.index(f"{section}:") + 1
    while lines[start].strip() != "default_settings:":
        start += 1

    key_at = re.compile(r"^(\s+)([A-Za-z_]\w*):(.*)$")
    described, stack, current = {}, [], None
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith("    "):
            break  # back out to the section's next key
        match = key_at.match(line)
        if match:
            indent, key, rest = len(match.group(1)), match.group(2), match.group(3)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            path = ".".join([*(k for _, k in stack), key])
            stack.append((indent, key))
            current = path
            described[path] = "#" in rest and rest.split("#", 1)[1].strip() != ""
        elif current and "#" in line and line.split("#", 1)[1].strip():
            described[current] = True
    return described


@pytest.mark.unit
@pytest.mark.parametrize("section", _SECTIONS)
def test_package_backed_keys_are_described(section):
    """
    A declared key with no description is how `channels` and `length` went dead.

    Presence only - the prose is prose. The owning callable is deliberately not
    named in master: that pointer lives in _PACKAGE_BACKED, where it is
    executable, and repeating it here would be a second unverifiable copy.
    """
    described = _described_keys(section)
    missing = [
        key
        for (sect, key) in _PACKAGE_BACKED
        if sect == section and not described.get(key, False)
    ]
    assert not missing, (
        f"{section}.default_settings keys with no description: {missing}"
    )


@pytest.mark.unit
def test_master_names_no_callable_in_default_settings():
    """The pointer belongs in the table, not in prose a test cannot verify."""
    described = {
        section: _described_keys(section) for section in _SECTIONS
    }  # parse must succeed for every section
    assert described

    lines = MASTER_CONFIG.read_text().splitlines()
    owners = {owner.__qualname__.split(".")[0] for owner, _ in _PACKAGE_BACKED.values()}
    offenders = [
        line.strip() for line in lines if any(f"{owner}." in line for owner in owners)
    ]
    assert not offenders, (
        f"master.yaml names a package symbol: {offenders}. Descriptions say what a "
        "key does; the owning signature is recorded in _PACKAGE_BACKED."
    )


# ── the split signatures must not drift apart again ───────────────────────────


@pytest.mark.unit
def test_split_signatures_agree():
    """
    An override restating a different default is how val_split drifted.

    `UcrUeaDataset.split` and the synthetic dataset's own `split` both re-declare
    the base signature, so each is a place a default can quietly diverge from the
    one the pointer table reads.
    """
    base = inspect.signature(DatasetBase.split).parameters
    for subclass in (UcrUeaDataset, SynthMultilabelFreqDataset):
        overriding = inspect.signature(subclass.split).parameters
        for name, param in overriding.items():
            if name == "self" or param.default is inspect.Parameter.empty:
                continue
            assert name in base, f"{subclass.__name__}.split adds {name!r}"
            assert param.default == base[name].default, (
                f"{subclass.__name__}.split defaults {name} to {param.default!r}, "
                f"but DatasetBase.split uses {base[name].default!r}"
            )


# ── a mistyped hyperparameter is reported, not silently defaulted ─────────────


@pytest.mark.unit
def test_hyperparams_typo_is_a_named_config_error():
    """
    The failure class R6/R7/R20 closed on the YAML side, closed on this one too.

    `epochss` used to reach `hyperparams.get("epochs", ...)` and train the default
    number of epochs without a word.
    """
    user = _minimal_user_config(
        train_config={"models": [{"model": "FCN", "hyperparams": {"epochss": 50}}]}
    )
    config = config_resolve(user, MASTER_CONFIG)

    with pytest.raises(ValueError, match="epochss") as excinfo:
        config_sanity_check(config)
    assert "epochs" in str(excinfo.value), "the near-miss should be suggested"


@pytest.mark.unit
def test_hyperparams_typo_would_also_raise_at_the_call():
    """Why the config error is worth having: the call itself is unforgiving."""
    accepted = inspect.signature(ModelBase.train_model).parameters
    assert "epochss" not in accepted


# ── the consequence of val_split defaulting to 0.1 ────────────────────────────


@pytest.mark.unit
def test_inherited_val_split_is_reported_on_the_archive_path(caplog):
    """
    The default asks for a validation split the archive partition cannot give.

    Saying so is the point: that path produces no validation set, so early
    stopping falls back to training loss. The comparison there is against zero
    rather than against this default, precisely so an inherited value is caught.
    """
    dataset = UcrUeaDataset("GunPoint", use_predefined_splits=True)
    default = package_default("data_config", "val_split")

    with caplog.at_level("WARNING", logger="xai4tsc.data.datasets"):
        dataset._report_ignored_split_args(
            package_default("data_config", "train_split"), default, True
        )

    assert f"val_split={default:.4g}" in caplog.text
    assert "no validation set is produced" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_file",
    sorted(p for p in (_RUNNER / "configs").glob("*.yaml") if p.name != "master.yaml"),
    ids=lambda p: p.name,
)
def test_shipped_configs_on_the_archive_path_ask_for_no_val_split(config_file):
    """
    A shipped config must not ask for what its split path cannot deliver.

    Inheriting the 0.1 default on the archive path is a warning per run, so the
    benchmark configs state 0.0 instead. This is the guard against one of them
    quietly losing that line.
    """
    user = yaml.safe_load(config_file.read_text())
    defaults = (user.get("data_config") or {}).get("default_settings") or {}
    if not defaults.get("use_predefined_splits"):
        pytest.skip(f"{config_file.name} splits the pooled data itself")
    assert defaults.get("val_split") == 0.0, (
        f"{config_file.name} uses the archive partition, so it should state "
        "val_split: 0.0 rather than inherit a validation split it will not get"
    )
