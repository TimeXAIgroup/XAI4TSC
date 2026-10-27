"""
Where the runner's default settings come from.

The package's own function signatures are the single source of truth for every
default a component consumes. This module records which signature owns which
config key and reads the value off it, so a renamed or removed parameter fails
here at startup instead of leaving a stale literal behind. ``master.yaml``
documents the same values for people writing configs, and a test holds the two in
step.

Keys the package knows nothing about - the runner's own control flow, and the
Quantus parameter blocks - are listed in :data:`_RUNNER_OWNED` instead, where
``master.yaml`` stays authoritative. Between them the two tables must cover every
key in every ``default_settings`` block; a test enforces that, so a new key has
to be classified rather than quietly joining neither group.
"""

import copy
import inspect
import logging
from collections.abc import Callable

from xai4tsc.data.base import DatasetBase
from xai4tsc.data.datasets import LocalDataset, UcrUeaDataset
from xai4tsc.models.base import ModelBase
from xai4tsc.xai.explain import generate_explanation

logger = logging.getLogger("xai4tsc.runner.defaults")

_PACKAGE_BACKED: dict[tuple[str, str], tuple[Callable, str]] = {
    # (section, key within default_settings) -> (owning callable, parameter name)
    ("data_config", "train_split"): (DatasetBase.split, "train_split"),
    ("data_config", "val_split"): (DatasetBase.split, "val_split"),
    ("data_config", "stratify"): (DatasetBase.split, "stratify"),
    ("data_config", "allow_imputation"): (DatasetBase.split, "impute_missing"),
    ("data_config", "max_samples"): (DatasetBase.__init__, "max_samples"),
    ("data_config", "sample_strategy"): (DatasetBase.__init__, "sample_strategy"),
    ("data_config", "max_series_length"): (DatasetBase.__init__, "max_series_length"),
    ("data_config", "series_position"): (DatasetBase.__init__, "series_position"),
    ("data_config", "download"): (UcrUeaDataset.__init__, "download"),
    ("data_config", "allow_padding"): (UcrUeaDataset.__init__, "pad_series"),
    ("data_config", "use_predefined_splits"): (
        DatasetBase.split,
        "use_predefined_splits",
    ),
    ("data_config", "format"): (LocalDataset.__init__, "data_format"),
    ("data_config", "layout"): (LocalDataset.__init__, "layout"),
    # The string form, which is what a config carries; `generate_explanation`
    # takes the same setting as a `Domain` member and so cannot own this key.
    ("data_config", "data_domain"): (LocalDataset.__init__, "data_domain"),
    ("train_config", "hyperparams.epochs"): (ModelBase.train_model, "epochs"),
    ("train_config", "hyperparams.batchsize"): (ModelBase.train_model, "batchsize"),
    ("train_config", "hyperparams.learn_rate"): (ModelBase.train_model, "learn_rate"),
    ("train_config", "hyperparams.patience"): (ModelBase.train_model, "patience"),
    ("train_config", "hyperparams.min_delta"): (ModelBase.train_model, "min_delta"),
    ("train_config", "hyperparams.save_best"): (ModelBase.train_model, "save_best"),
    ("explanation_config", "target"): (generate_explanation, "targets"),
    ("explanation_config", "threshold"): (generate_explanation, "threshold"),
    ("explanation_config", "visualization_type"): (
        generate_explanation,
        "visualization_type",
    ),
    ("explanation_config", "samples.count"): (generate_explanation, "n_samples"),
}
"""Config keys whose default is read off the package parameter that consumes it."""

_RUNNER_OWNED: frozenset[tuple[str, str]] = frozenset(
    {
        # Pipeline control flow: the runner reads these, no package function sees
        # them.
        ("data_config", "skip_dataset"),
        ("data_config", "skip_models"),
        ("data_config", "skip_explainers"),
        ("data_config", "skip_metrics"),
        ("data_config", "path"),
        ("data_config", "is_presplit"),
        # The encoder is an object in the package signature, so there is no
        # parameter default to read: the string is the runner's own, and it also
        # names the split-cache folder. `encode_params` is free-form - a
        # third-party encoder has arguments master cannot enumerate - and is
        # consumed by the runner rather than passed to the package.
        ("data_config", "encode"),
        ("data_config", "encode_params"),
        ("explanation_config", "save_exp_values"),
        ("train_config", "train"),
        # The loss and the optimizer are objects in the package signature, whose
        # default is None, so there is no value to read off it: the names and
        # their constructor arguments are the runner's own. The two `*_params`
        # mappings are free-form - a third-party loss has parameters master
        # cannot enumerate - and are consumed by the runner rather than passed to
        # `train_model`.
        ("train_config", "hyperparams.loss_func"),
        ("train_config", "hyperparams.loss_params"),
        ("train_config", "hyperparams.optimizer"),
        ("train_config", "hyperparams.optimizer_params"),
        # Which of `indices` / `count` applies is a runner dispatch, and the
        # indices in master are a placeholder: an entry selecting by index has to
        # supply its own, which config validation requires.
        ("explanation_config", "samples.type"),
        ("explanation_config", "samples.indices"),
        # Quantus parameters fan out to ~37 metric classes and are per-metric
        # rather than per-section, so they are not derived; the partial-wrapped
        # registry entries expose only the adapter's own parameters anyway.
        ("evaluation_config", "metric_class_params"),
        ("evaluation_config", "metric_call_params"),
    }
)
"""Config keys with no package owner, for which ``master.yaml`` is the source."""


def package_default(section: str, key: str) -> object:
    """
    Read one package-backed default off the signature that owns it.

    Parameters
    ----------
    section : str
        Config section, e.g. ``"data_config"``.
    key : str
        Key within that section's ``default_settings``, dotted for a nested one
        (e.g. ``"hyperparams.epochs"``).

    Returns
    -------
    object
        The owning parameter's default. A tuple default is returned as a list, so
        the value stays plain YAML rather than reaching the resolved config and
        the run log as a Python tuple.

    Raises
    ------
    KeyError
        If the key is not package-backed, or if the owning callable no longer has
        that parameter - which is the point of the table: a rename fails loudly
        here rather than leaving the runner on a stale literal.
    """
    try:
        owner, param = _PACKAGE_BACKED[(section, key)]
    except KeyError:
        raise KeyError(
            f"'{section}.{key}' is not a package-backed default. Add it to "
            "_PACKAGE_BACKED, or to _RUNNER_OWNED if the package never sees it."
        ) from None

    params = inspect.signature(owner).parameters
    if param not in params:
        raise KeyError(
            f"'{section}.{key}' points at {owner.__qualname__}.{param}, which no "
            f"longer exists. Update _PACKAGE_BACKED, and change the default in "
            f"{owner.__qualname__} rather than in master.yaml."
        )
    default = params[param].default
    if default is inspect.Parameter.empty:
        raise KeyError(
            f"'{section}.{key}' points at {owner.__qualname__}.{param}, which has "
            "no default to read. Give it one, or make the key runner-owned."
        )
    return list(default) if isinstance(default, tuple) else default


def hyperparam_names() -> frozenset[str]:
    """
    Return the training hyperparameters a model entry may set.

    Derived from both pointer tables rather than listed again, so adding a
    hyperparameter to :meth:`~xai4tsc.models.base.ModelBase.train_model` and the
    table is enough to make it configurable.

    Both tables, not just the package-backed one: ``loss_func`` and ``optimizer``
    name library objects the runner builds, and ``loss_params`` /
    ``optimizer_params`` are consumed by the runner and never reach
    ``train_model`` at all. Deriving from ``_PACKAGE_BACKED`` alone would make
    all four "not a training parameter" in config validation.

    Returns
    -------
    frozenset[str]
        The accepted key names inside a ``hyperparams`` block.
    """
    prefix = "hyperparams."
    return frozenset(
        key.removeprefix(prefix)
        for section, key in (*_PACKAGE_BACKED, *_RUNNER_OWNED)
        if section == "train_config" and key.startswith(prefix)
    )


def _stated(block: dict | None, key: str) -> bool:
    """Return whether *block* states the (possibly dotted) *key* itself."""
    node = block
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _assign(block: dict, key: str, value: object) -> None:
    """Write *value* at the (possibly dotted) *key*, creating missing levels."""
    parts = key.split(".")
    for part in parts[:-1]:
        node = block.get(part)
        if not isinstance(node, dict):
            node = {}
            block[part] = node
        block = node
    block[parts[-1]] = copy.deepcopy(value)


def apply_package_defaults(user_config: dict, resolved: dict) -> None:
    """
    Replace master's copy of every package-backed default with the derived one.

    Modifies *resolved* in place. A key the user states in their own
    ``default_settings`` is left alone, so user settings still win; for every
    other key master's value is overwritten by the package's, which is what makes
    master documentation rather than a second authority. The two agree by test, so
    this changes no value - it removes the possibility of them disagreeing.

    Parameters
    ----------
    user_config : dict
        The raw user config, consulted only to see which keys the user stated.
    resolved : dict
        The config after master has been merged in, modified in place.
    """
    for section, key in _PACKAGE_BACKED:
        block = (resolved.get(section) or {}).get("default_settings")
        if block is None:
            continue
        user_block = (user_config.get(section) or {}).get("default_settings")
        if _stated(user_block, key):
            continue
        _assign(block, key, package_default(section, key))


def default_settings_leaves(section: str, block: dict) -> list[str]:
    """
    List the (possibly dotted) keys of one ``default_settings`` block.

    Nested mappings are descended into, so ``hyperparams`` contributes one key per
    hyperparameter - except where the parent itself is runner-owned, which stops
    the descent, since the whole subtree then belongs to ``master.yaml``.

    Parameters
    ----------
    section : str
        Config section the block belongs to.
    block : dict
        The ``default_settings`` mapping.

    Returns
    -------
    list[str]
        One entry per leaf, in declaration order.
    """
    leaves: list[str] = []

    def walk(node: dict, prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}{key}"
            if (section, path) in _RUNNER_OWNED:
                leaves.append(path)
            elif isinstance(value, dict) and value:
                walk(value, f"{path}.")
            else:
                leaves.append(path)

    walk(block, "")
    return leaves
