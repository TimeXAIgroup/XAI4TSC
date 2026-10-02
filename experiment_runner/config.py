"""
Config loading and validation for the experiment runner.

Resolves user YAML against ``master.yaml``, applies per-method defaults,
and validates model and explainer names against the live runtime registries.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger("xai4tsc.runner.config")


def disambiguate(names: list[str]) -> list[str]:
    """
    Make a list of display names unique by suffixing only the duplicates.

    Output paths and ``metrics.csv`` rows are keyed on an entry's display name,
    so two config entries sharing a name overwrite each other's artifacts and
    produce indistinguishable result rows. Suffixing ``_1.._N`` onto just the
    repeated names keeps every unique name - the overwhelmingly common case -
    byte-identical to what earlier versions wrote.

    Parameters
    ----------
    names : list of str
        Display names in config order, one per entry at a single level
        (datasets, models, explainers or metrics).

    Returns
    -------
    list of str
        One identity per input name, positionally aligned. Names occurring once
        are returned unchanged; names occurring more than once gain a ``_1``,
        ``_2``, ... suffix in config order.

    Examples
    --------
    >>> disambiguate(["TSHAP", "TSHAP", "SIGN"])
    ['TSHAP_1', 'TSHAP_2', 'SIGN']
    """
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1

    seen: dict[str, int] = {}
    identities: list[str] = []
    for name in names:
        if counts[name] == 1:
            identities.append(name)
            continue
        seen[name] = seen.get(name, 0) + 1
        identities.append(f"{name}_{seen[name]}")

    # A generated suffix can collide with a name that was already spelled that
    # way (e.g. entries "X", "X" and a literal "X_1"). Warn rather than raise:
    # the run is still usable, the user just cannot tell those two apart.
    duplicates = {i for i in identities if identities.count(i) > 1}
    if duplicates:
        logger.warning(
            "Result identities are not unique after disambiguation: %s. Rename "
            "or remove the conflicting entries to keep their outputs separate.",
            ", ".join(sorted(duplicates)),
        )
    return identities


def register_external_components(config: dict) -> None:
    """
    Load and register external model/explainer/metric classes from the config.

    Silent extension fallback used by :func:`config_sanity_check`: when a
    configured component name is absent from its runtime registry but the item
    supplies ``class_name`` and ``class_path``, the class is imported from that
    source file and registered, so the rest of the pipeline treats it like any
    built-in. This lets users add components without editing package source.

    Gated by ``general.allow_external_code`` (default ``False``). If a path is
    supplied while the flag is off, a warning is logged and the component is
    left unregistered, so the subsequent validation in
    :func:`config_sanity_check` raises a clear error.

    Parameters
    ----------
    config : dict
        Resolved configuration dict.
    """
    # Lazy imports to avoid circular dependencies at module level
    from xai4tsc.evaluation.evaluate import METRICS, register_metric
    from xai4tsc.models.models import MODELS, register_model
    from xai4tsc.utils.utils import load_class_from_path
    from xai4tsc.xai.explain import EXPLAINERS, register_explainer

    allow = config.get("general", {}).get("allow_external_code", False)

    def _load(kind: str, name: str, item: dict) -> type | None:
        path, class_name = item.get("class_path"), item.get("class_name")
        if not (path and class_name):
            return None
        if not allow:
            logger.warning(
                "External source '%s' specified for %s '%s' but "
                "general.allow_external_code is false; not loading.",
                path,
                kind,
                name,
            )
            return None
        cls = load_class_from_path(path, class_name)
        logger.info("Registered external %s '%s' from %s", kind, name, path)
        return cls

    for model_conf in config.get("train_config", {}).get("models", []):
        name = model_conf["model"]
        if name.lower() in MODELS:
            continue
        # Models are looked up case-insensitively, so register under lower case.
        if (cls := _load("model", name, model_conf)) is not None:
            register_model(name.lower(), cls)

    for exp_conf in config.get("explanation_config", {}).get("explainers", []):
        name = exp_conf["method"]
        if name.lower() in EXPLAINERS:
            continue
        # register_explainer lower-cases the key internally.
        if (cls := _load("explainer", name, exp_conf)) is not None:
            register_explainer(name, cls)

    for metric_conf in config.get("evaluation_config", {}).get("metrics", []):
        name = metric_conf["metric"]
        if name in METRICS:
            continue
        # METRICS keys are exact display names, not lower-cased.
        if (cls := _load("metric", name, metric_conf)) is not None:
            register_metric(name, cls)


def _merge_warn_drop(user: dict, master: dict, context: str) -> dict:
    """
    Merge user config into master config, warning and dropping unknown keys.

    Unknown keys (present in user but not in master) are logged as warnings
    and excluded from the result. Known keys from user take priority over master.

    Parameters
    ----------
    user : dict
        User-supplied config section.
    master : dict
        Master config section defining known keys.
    context : str
        Human-readable path for warning messages (e.g. ``"general"``).

    Returns
    -------
    dict
        Merged dict starting from master, overlaid with valid user keys.
    """
    result = dict(master)
    for key, value in user.items():
        if key not in master:
            logger.warning("Unknown key '%s' in '%s', ignoring.", key, context)
            continue
        if isinstance(value, dict) and isinstance(master[key], dict):
            result[key] = _merge_warn_drop(value, master[key], f"{context}.{key}")
        else:
            result[key] = value
    return result


_REQUIRED_SETTINGS = ("experiment_name", "results_rel_path")
"""Top-level settings every user config must supply."""

_REQUIRED_LISTS = (
    ("data_config", "datasets"),
    ("train_config", "models"),
    ("explanation_config", "explainers"),
    ("evaluation_config", "metrics"),
)
"""The four component lists: what the experiment runs, which cannot be defaulted."""


def validate_user_config(user_config: dict) -> None:
    """
    Check that a user config supplies everything that cannot be defaulted.

    Operates on the **raw** config, before :func:`config_resolve` merges master in.
    That ordering is essential: master declares the four component lists as a
    catalogue, so after resolution an omitted list is indistinguishable from an
    inherited one.  Settings describe *how* an experiment runs and are defaulted
    from master; the four lists describe *what* it runs and must be stated.

    Parameters
    ----------
    user_config : dict
        Raw config dict as loaded from the user YAML.

    Raises
    ------
    ValueError
        If any required setting or component list is missing, naming all of them.
    """
    missing = [key for key in _REQUIRED_SETTINGS if not user_config.get(key)]
    empty = []
    for section, key in _REQUIRED_LISTS:
        block = user_config.get(section)
        if not isinstance(block, dict) or key not in block:
            missing.append(f"{section}.{key}")
        elif not block[key]:
            empty.append(f"{section}.{key}")

    if missing:
        why = ""
        if any("." in name for name in missing):
            why = (
                " The component lists describe what the experiment runs and so have "
                "no sensible default; set one to [] to skip that level deliberately."
            )
        err_msg = (
            f"Config is missing required setting(s): {', '.join(missing)}.{why} See "
            "experiment_runner/configs/master.yaml for every available option and "
            "configs/example.yaml for a working starting point."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    for name in empty:
        logger.warning(
            "'%s' is empty - that level of the pipeline will do nothing.", name
        )


def config_resolve(user_config: dict, master_path: Path) -> dict:
    """
    Resolve a user config against the master config.

    Steps performed:

    1. Load master.yaml (annotated reference with all known keys and the default
       values of the ``general`` and ``default_settings`` blocks).
    2. Warn + drop any unknown top-level or nested keys found in user_config.

    Master's four component lists (``datasets``, ``models``, ``explainers``,
    ``metrics``) are a **catalogue**, not a source of values: parameter defaults
    belong to the classes that consume them, so an omitted parameter falls
    through to its constructor default via
    :func:`~xai4tsc.utils.utils.dict_to_args` rather than being filled in here.
    That keeps the package and the runner in agreement about what a given
    method is.  Because :func:`validate_user_config` requires all four lists,
    master's own entries are always overlaid and never leak into a run.

    Parameters
    ----------
    user_config : dict
        Raw config dict as loaded from the user YAML.
    master_path : Path
        Path to master.yaml.

    Returns
    -------
    dict
        Resolved config with defaults applied and unknown keys removed.
    """
    with open(master_path) as f:
        master = yaml.safe_load(f)

    # Warn + drop at the top level and recursively inside known sections
    return _merge_warn_drop(user_config, master, "root")


_LEGACY_TRAIN_VALUES = frozenset({"till_epoch", "for_epochs"})


def _normalize_train_switch(train_config: dict) -> None:
    """
    Normalise every ``train`` value in *train_config* to a bool or ``None``.

    ``train`` is a switch: truthy trains, ``false`` or absent does not.  The
    legacy values ``"till_epoch"`` and ``"for_epochs"`` never differed in
    behaviour - neither was ever read - so both normalise to ``True`` with one
    deprecation warning per run.  Rewriting here means the runner only ever sees
    a bool or ``None``.

    Parameters
    ----------
    train_config : dict
        The ``train_config`` section, modified in place.

    Raises
    ------
    ValueError
        If a ``train`` value is a string that is not a recognised legacy value.
    """
    targets = [("train_config.default_settings", train_config.get("default_settings"))]
    targets += [
        (f"train_config.models[{i}] ('{entry.get('model')}')", entry)
        for i, entry in enumerate(train_config.get("models", []))
    ]

    deprecated_used = False
    for where, block in targets:
        if not isinstance(block, dict) or "train" not in block:
            continue
        value = block["train"]
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, str) and value.lower() in _LEGACY_TRAIN_VALUES:
            block["train"] = True
            deprecated_used = True
            continue
        err_msg = (
            f"Invalid 'train' value {value!r} in {where}. "
            "'train' is a switch: true trains, false (or omitting it) does not."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    if deprecated_used:
        logger.warning(
            "'train' values 'till_epoch' / 'for_epochs' are deprecated and were "
            "treated as true. Neither was ever read, so they behaved identically; "
            "use 'train: true'. 'epochs' is the number of epochs to run in this "
            "call, and a run resuming from a checkpoint continues its epoch count."
        )


def config_sanity_check(config: dict) -> bool:
    """
    Check whether the supplied config contains errors.

    First registers any external components declared via ``class_name`` /
    ``class_path`` (see :func:`register_external_components`), then validates
    models and explainers against the live runtime registries so that both
    custom entries added via ``register_model()`` / ``register_explainer()`` and
    external classes loaded from source are accepted.

    Parameters
    ----------
    config : dict
        Configuration to check.

    Returns
    -------
    bool
        ``True`` if the configuration contains no errors.
    """
    # Last-resort silent extension: load external classes into the registries
    # before validating, so configured-but-unregistered names can still resolve.
    register_external_components(config)

    _normalize_train_switch(config["train_config"])

    # Lazy imports to avoid circular dependencies at module level
    from xai4tsc.models.models import MODELS
    from xai4tsc.xai.explain import EXPLAINERS

    # Validate models against live registry
    for model_conf in config["train_config"]["models"]:
        name = model_conf["model"]
        if name.lower() not in MODELS:
            err_msg = (
                f"Model '{name}' is not registered. "
                "Use register_model() or check MODELS."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    # Validate explainers against live registry (case-insensitive)
    for exp_conf in config["explanation_config"]["explainers"]:
        name = exp_conf["method"].lower()
        if name not in EXPLAINERS:
            err_msg = (
                f"Explainer '{name}' is not registered. "
                "Use register_explainer() or check EXPLAINERS."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    # Domain compatibility: a domain-restricted metric (e.g. a frequency
    # perturbation metric) is pointless unless at least one configured explainer
    # produces a matching domain. Validate up front against the static capability
    # sets (incompatible explainer/metric pairs are skipped at runtime).
    from xai4tsc.evaluation.evaluate import METRICS

    explainer_domains: set = set()
    for exp_conf in config["explanation_config"]["explainers"]:
        cls = EXPLAINERS.get(exp_conf["method"].lower())
        if cls is not None:
            explainer_domains |= getattr(cls, "explanation_domains", set())
    for metric_conf in config.get("evaluation_config", {}).get("metrics", []):
        required = getattr(METRICS.get(metric_conf["metric"]), "required_domains", None)
        if required and not (required & explainer_domains):
            err_msg = (
                f"Metric '{metric_conf['metric']}' requires an explanation in "
                f"domain(s) {{{', '.join(d.value for d in required)}}}, but no "
                "configured explainer produces one."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    return True
