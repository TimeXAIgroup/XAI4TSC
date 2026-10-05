"""
Config loading and validation for the experiment runner.

Resolves user YAML against ``master.yaml``, applies per-method defaults,
and validates model and explainer names against the live runtime registries.
"""

import difflib
import inspect
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


def _merge_warn_drop(
    user: dict, master: dict, context: str, issues: "ConfigIssues | None" = None
) -> dict:
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
    issues : ConfigIssues, optional
        Collector for the unknown keys found.  When omitted, one is created and
        reported immediately, which keeps standalone calls self-contained.

    Returns
    -------
    dict
        Merged dict starting from master, overlaid with valid user keys.
    """
    own_report = issues is None
    issues = ConfigIssues() if issues is None else issues

    result = dict(master)
    for key, value in user.items():
        if key not in master:
            issues.warn(
                context,
                f"'{key}' is not a known setting and was dropped."
                f"{_suggest(key, set(master))}",
            )
            continue
        if isinstance(value, dict) and isinstance(master[key], dict):
            result[key] = _merge_warn_drop(
                value, master[key], f"{context}.{key}", issues
            )
        else:
            result[key] = value
    if own_report:
        issues.report()
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


def validate_user_config(
    user_config: dict, issues: "ConfigIssues | None" = None
) -> None:
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
    issues : ConfigIssues, optional
        Collector for the empty-list warnings.  This check must run before the
        results directory exists, hence before file logging is attached, so
        deferring its warnings is what gets them into ``experiment.log``.

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
        message = "this list is empty, so that level of the pipeline will do nothing."
        if issues is None:
            logger.warning("'%s': %s", name, message)
        else:
            issues.warn(name, message)


def config_resolve(
    user_config: dict, master_path: Path, issues: "ConfigIssues | None" = None
) -> dict:
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
    issues : ConfigIssues, optional
        Collector for the unknown keys dropped here, so they are reported with
        every other config problem instead of on their own.

    Returns
    -------
    dict
        Resolved config with defaults applied and unknown keys removed.
    """
    with open(master_path) as f:
        master = yaml.safe_load(f)

    # Warn + drop at the top level and recursively inside known sections
    return _merge_warn_drop(user_config, master, "root", issues)


_LEGACY_TRAIN_VALUES = frozenset({"till_epoch", "for_epochs"})


def _normalize_train_switch(
    train_config: dict, issues: "ConfigIssues | None" = None
) -> None:
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
    issues : ConfigIssues, optional
        Collector for the deprecation warning.  When omitted it is logged
        directly.

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
        message = (
            "'train' values 'till_epoch' / 'for_epochs' are deprecated and were "
            "treated as true. Neither was ever read, so they behaved identically; "
            "use 'train: true'. 'epochs' is the number of epochs to run in this "
            "call, and a run resuming from a checkpoint continues its epoch count."
        )
        if issues is None:
            logger.warning(message)
        else:
            issues.warn("train_config", message)


class ConfigIssues:
    """
    Collector for config problems, so one run reports all of them in one place.

    Split by consequence: an **error** means the experiment cannot run as
    configured, a **warning** means the run proceeds but not as the user wrote it.
    Every config check appends here rather than logging as it goes - unknown keys
    in the settings blocks, deprecated values, unknown entry keys, unregistered
    component names - so the report is a single block with an honest count, rather
    than warnings scattered through the setup output.

    Warnings are logged even when errors follow, so a config with several mistakes
    takes one run to fix rather than one run per mistake.
    """

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, message: str) -> None:
        """Record a problem that prevents the run."""
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        """Record something the run will proceed without honouring."""
        self.warnings.append(f"{where}: {message}")

    def report(self, repeat: bool = False) -> None:
        """
        Log everything collected as one block, then raise if anything was fatal.

        Parameters
        ----------
        repeat : bool
            ``True`` when re-printing at the end of a finished run, which only
            re-states the warnings and never raises.  A long run scrolls its
            configuration report out of view, so the warnings are shown again
            where the user will actually be looking.

        Raises
        ------
        ValueError
            If any error was recorded and *repeat* is false; the message carries
            every one of them.
        """
        if not self.errors and not self.warnings:
            return
        headline = (
            "--- Configuration warnings, repeated (%d) ---"
            if repeat
            else "--- Configuration check: %d error(s), %d warning(s) ---"
        )
        logger.info(
            headline % (len(self.warnings),)
            if repeat
            else headline % (len(self.errors), len(self.warnings))
        )
        if self.warnings:
            logger.warning(
                "The run proceeds, but these settings were not honoured:\n  %s",
                "\n  ".join(self.warnings),
            )
        if self.errors and not repeat:
            err_msg = "The run cannot proceed:\n  " + "\n  ".join(self.errors)
            logger.error(err_msg)
            raise ValueError(err_msg)


# Per-entry keys that no `default_settings` block declares: structural, or read by
# the runner rather than passed to the component.
_ENTRY_EXTRAS = {
    "dataset": frozenset({"init_params", "overrides"}),
    "model": frozenset({"init_params", "model_ckpt_path", "class_name", "class_path"}),
    "method": frozenset({"class_name", "class_path"}),
    "metric": frozenset(
        {"animate", "animate_sample", "animate_fps", "class_name", "class_path"}
    ),
}

# identity key -> (config section, list key)
_ENTRY_SECTIONS = {
    "dataset": ("data_config", "datasets"),
    "model": ("train_config", "models"),
    "method": ("explanation_config", "explainers"),
    "metric": ("evaluation_config", "metrics"),
}

# Where a component's parameters belong. Explainers take theirs at the entry's top
# level and so are absent here; the other two nest them
# describes.
_PARAM_HOME = {
    "model": "'init_params'",
    "metric": "'metric_class_params' (constructor) or 'metric_call_params' (call time)",
}

_ENTRY_LABEL = {
    "dataset": "dataset setting",
    "model": "model entry key",
    "method": "explainer entry key",
    "metric": "metric entry key",
}


def _suggest(key: str, candidates: set[str]) -> str:
    """Return a ' Did you mean ...?' hint for *key*, or an empty string."""
    close = difflib.get_close_matches(key, sorted(candidates), n=1, cutoff=0.7)
    return f" Did you mean '{close[0]}'?" if close else ""


def _check_component_name(
    kind: str, name: str, registry: dict, where: str, issues: ConfigIssues
) -> type | None:
    """
    Resolve an entry's component name against its registry, recording an error.

    Models and explainers are looked up case-insensitively; metric keys are
    user-facing display names and are matched exactly, so a case-only mismatch is
    reported with the registered spelling.

    Returns
    -------
    type or None
        The registered class, or ``None`` when the name does not resolve.
    """
    exact = name if kind == "metric" else name.lower()
    if exact in registry:
        return registry[exact]

    case_match = [key for key in registry if key.lower() == name.lower()]
    if case_match:
        # Only metrics can reach this: model and explainer lookups are already
        # case-insensitive, so a case-only mismatch there would have resolved.
        hint = (
            " Metric names are case-sensitive display names; the registered "
            f"spelling is '{case_match[0]}'."
        )
    else:
        hint = (
            f"{_suggest(name, set(registry))} Register it at runtime with "
            f"register_{'explainer' if kind == 'method' else kind}(), or check the "
            "available names."
        )
    issues.error(where, f"{kind} '{name}' is not registered.{hint}")
    return None


def _validate_entry(
    entry: dict,
    kind: str,
    where: str,
    overridable: set[str],
    registry: dict | None,
    issues: ConfigIssues,
) -> None:
    """Check one list entry's name and keys; see :func:`config_sanity_check`."""
    name = entry.get(kind)
    where = f"{where} ({name})"
    cls = None
    if registry is not None and isinstance(name, str):
        cls = _check_component_name(kind, name, registry, where, issues)

    # Explainers take their parameters at the entry's top level; a partial-wrapped
    # metric has no introspectable signature, and model parameters are nested.
    params: set[str] = set()
    if cls is not None and kind in ("method", "model"):
        params = set(inspect.signature(cls.__init__).parameters) - {"self"}

    accepted = {kind} | overridable | _ENTRY_EXTRAS[kind]
    if kind == "method":
        accepted |= params

    for key in entry:
        if key in accepted:
            continue
        if kind in _PARAM_HOME and (key in params or kind == "metric"):
            issues.warn(
                where,
                f"'{key}' is not read at an entry's top level; "
                f"{'parameters of this ' + kind} go under {_PARAM_HOME[kind]}.",
            )
        else:
            label = _ENTRY_LABEL[kind]
            issues.warn(
                where, f"'{key}' is not a known {label}.{_suggest(key, accepted)}"
            )

    # A wrapper explainer nests the method it wraps, with its own parameters.
    if kind == "method" and isinstance(entry.get("base"), dict):
        _validate_entry(
            entry["base"], kind, f"{where}.base", overridable, registry, issues
        )


def _validate_entries(config: dict, issues: ConfigIssues) -> None:
    """Validate every component list entry's name and keys."""
    from xai4tsc.evaluation.evaluate import METRICS
    from xai4tsc.models.models import MODELS
    from xai4tsc.xai.explain import EXPLAINERS

    registries = {"model": MODELS, "method": EXPLAINERS, "metric": METRICS}
    for kind, (section, list_key) in _ENTRY_SECTIONS.items():
        block = config.get(section) or {}
        # The resolved default_settings is exactly the set of keys an entry may
        # override: _merge_warn_drop has already dropped anything master
        # does not declare.
        overridable = set(block.get("default_settings") or {})
        for index, entry in enumerate(block.get(list_key) or []):
            if not isinstance(entry, dict):
                issues.error(
                    f"{section}.{list_key}[{index}]",
                    f"expected a mapping, got {entry!r}",
                )
                continue
            _validate_entry(
                entry,
                kind,
                f"{section}.{list_key}[{index}]",
                overridable,
                registries.get(kind),
                issues,
            )


def _configured_data_domains(config: dict) -> set:
    """
    Return the signal domains the configured datasets' arrays are expressed in.

    A dataset entry's ``data_domain`` is only honoured for a local (``path``)
    dataset; UCR/UEA data is always raw series and a synthetic dataset declares its
    own domain on the class, so those contribute whatever they declare.

    Parameters
    ----------
    config : dict
        A resolved configuration.

    Returns
    -------
    set of Domain
        One entry per distinct domain, defaulting to the time domain.
    """
    from xai4tsc.data.datasets import SYNTHETIC_DATASETS
    from xai4tsc.xai._types import Domain, resolve_data_domain

    domains = set()
    for entry in config.get("data_config", {}).get("datasets") or []:
        name = entry.get("dataset")
        synthetic = SYNTHETIC_DATASETS.get(name)
        if synthetic is not None:
            domains.add(synthetic.data_domain)
        elif entry.get("path"):
            domains.add(resolve_data_domain(entry.get("data_domain", "time")))
        else:
            domains.add(Domain.TIME)
    return domains or {Domain.TIME}


def config_sanity_check(config: dict, issues: "ConfigIssues | None" = None) -> bool:
    """
    Check whether the supplied config contains errors.

    First registers any external components declared via ``class_name`` /
    ``class_path`` (see :func:`register_external_components`), then validates every
    component list entry against the live runtime registries, so that both custom
    entries added via ``register_model()`` / ``register_explainer()`` /
    ``register_metric()`` and external classes loaded from source are accepted.

    Problems are **split by consequence** and collected into one report:

    * an unregistered component name is an **error** - the entry cannot run;
    * a key the runner will never read is a **warning** - the entry runs, but not
      as the user wrote it. This is where most real typos land, because a parameter
      name that does not match is dropped by
      :func:`~xai4tsc.utils.utils.dict_to_args` and the component silently uses its
      own default.

    Keys in the *mapping* sections are handled separately, by
    :func:`_merge_warn_drop` during resolution; this function owns the *list*
    entries, which that merge cannot see into.

    Parameters
    ----------
    config : dict
        Configuration to check.
    issues : ConfigIssues, optional
        Collector to append to, so these problems are reported alongside those
        found during resolution.  When omitted, one is created and reported here.

    Returns
    -------
    bool
        ``True`` if the configuration contains no errors.

    Raises
    ------
    ValueError
        If any entry names an unregistered component, or a domain-restricted
        metric has no explainer that can satisfy it.
    """
    # Last-resort silent extension: load external classes into the registries
    # before validating, so configured-but-unregistered names can still resolve.
    register_external_components(config)

    own_report = issues is None
    issues = ConfigIssues() if issues is None else issues

    _normalize_train_switch(config["train_config"], issues)

    # Lazy imports to avoid circular dependencies at module level
    from xai4tsc.xai.explain import EXPLAINERS

    # Component names and entry keys, collected so one run reports all of them.
    _validate_entries(config, issues)
    if own_report:
        issues.report()

    # Domain compatibility. Every configured dataset declares the domain its arrays
    # are in, which is the domain the model consumes, so it decides two things:
    # which explainers can read it, and what a plain attributor's explanation ends up
    # in. Both are static, so both are checked here rather than failing per dataset.
    from xai4tsc.evaluation.evaluate import required_domains_for

    data_domains = _configured_data_domains(config)

    explainer_domains: set = set()
    for exp_conf in config["explanation_config"]["explainers"]:
        cls = EXPLAINERS.get(exp_conf["method"].lower())
        if cls is None:
            continue
        accepted = getattr(cls, "accepts_data_domains", set())
        unreadable = data_domains - accepted if accepted else set()
        if accepted and not (data_domains & accepted):
            err_msg = (
                f"Explainer '{exp_conf['method']}' consumes "
                f"{{{', '.join(sorted(d.value for d in accepted))}}} data, but no "
                "configured dataset provides it."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        if unreadable:
            logger.warning(
                "Explainer '%s' cannot consume %s data; those datasets will fail.",
                exp_conf["method"],
                ", ".join(sorted(d.value for d in unreadable)),
            )
        # A plain attributor explains in whichever domain the data is, so its
        # capability is the set of configured data domains, not its declared default.
        if getattr(cls, "explains_in_data_domain", False):
            explainer_domains |= data_domains & accepted if accepted else data_domains
        else:
            explainer_domains |= getattr(cls, "explanation_domains", set())
    for metric_conf in config.get("evaluation_config", {}).get("metrics", []):
        # A metric tied to the model-native domain is satisfiable by any configured
        # data domain; the per-pair runtime check uses the explanation's own.
        required: set = set()
        for data_domain in data_domains:
            required |= required_domains_for(metric_conf["metric"], data_domain)
        if required and not (required & explainer_domains):
            err_msg = (
                f"Metric '{metric_conf['metric']}' requires an explanation in "
                f"domain(s) {{{', '.join(d.value for d in required)}}}, but no "
                "configured explainer produces one."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    return True
