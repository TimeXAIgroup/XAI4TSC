"""
Config loading and validation for the experiment runner.

Resolves user YAML against ``master.yaml``, applies per-method defaults,
and validates model and explainer names against the live runtime registries.
"""

import copy
import difflib
import inspect
import logging
from pathlib import Path

import yaml

from xai4tsc.utils.utils import merge_dicts

from .defaults import _RUNNER_OWNED, apply_package_defaults, hyperparam_names
from .encoders import build_encoder
from .losses import build_loss
from .optimizers import build_optimizer

logger = logging.getLogger("xai4tsc.runner.config")


def result_identities(names: list[str]) -> tuple[list[str], list[int]]:
    """
    Derive the display identity and the occurrence number of every entry.

    One pass over the names serves both, because both answer the same question -
    which config entry is this - for two consumers that need different trade-offs:

    * the **display identity** names output paths and ``metrics.csv`` rows, so two
      entries sharing a name would otherwise overwrite each other's artifacts and
      produce indistinguishable rows. Only the repeated names are suffixed
      ``_1.._N``, which keeps every unique name - the overwhelmingly common case -
      byte-identical to what earlier versions wrote.
    * the **occurrence number** keys a result's derived seed
      (:func:`~experiment_runner.seeding.boundary_seed`), and is always present for
      exactly that reason. The identity is not usable there: it suffixes only
      duplicates, so ``["A"]`` yields ``["A"]`` while ``["A", "A"]`` yields
      ``["A_1", "A_2"]``, and adding a second copy of an entry would silently change
      the first copy's seed. A running count leaves the first entry's number alone
      while still separating the copies, so duplicating an entry to observe variance
      still shows variance.

    A name's total count is its highest occurrence number, which is what decides
    whether it needs a suffix - so the identity is derived from the occurrences
    rather than counted separately.

    Parameters
    ----------
    names : list of str
        Display names in config order, one per entry at a single level
        (datasets, models, explainers or metrics).

    Returns
    -------
    tuple[list[str], list[int]]
        ``(identities, occurrences)``, both positionally aligned with *names*.

    Examples
    --------
    >>> result_identities(["TSHAP", "TSHAP", "SIGN"])
    (['TSHAP_1', 'TSHAP_2', 'SIGN'], [1, 2, 1])
    >>> result_identities(["TSHAP"])[1] == result_identities(["TSHAP", "TSHAP"])[1][:1]
    True
    """
    occurrences: list[int] = []
    seen: dict[str, int] = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        occurrences.append(seen[name])

    # The last occurrence number recorded for a name is its total count, so no
    # second counting pass is needed to tell a duplicate from a unique name.
    identities = [
        name if seen[name] == 1 else f"{name}_{occurrence}"
        for name, occurrence in zip(names, occurrences, strict=True)
    ]

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
    return identities, occurrences


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


def _free_form_mapping(context: str, key: str) -> bool:
    """
    Return whether ``context.key`` names a runner-owned free-form mapping.

    Such a mapping is one whose keys ``master.yaml`` cannot enumerate: the
    Quantus parameter blocks fan out to ~37 metric classes, and ``loss_params``
    takes whatever its loss constructor does, including a third-party one. They
    are listed in :data:`~experiment_runner.defaults._RUNNER_OWNED`, and
    :func:`~experiment_runner.defaults.default_settings_leaves` already stops its
    descent at exactly these keys; this is the same rule for the merge, which
    otherwise filters each level against master and drops what it does not find.

    Parameters
    ----------
    context : str
        The dotted merge context, e.g. ``"root.train_config.default_settings"``.
    key : str
        The key being merged inside *context*.

    Returns
    -------
    bool
        ``True`` when the user's mapping should be taken whole.
    """
    parts = context.split(".")
    if len(parts) < 3 or parts[0] != "root" or parts[2] != "default_settings":
        return False
    path = ".".join([*parts[3:], key])
    return (parts[1], path) in _RUNNER_OWNED


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
            if _free_form_mapping(context, key):
                # Master cannot enumerate these keys, so filtering against it
                # would silently drop every parameter it does not happen to list.
                # Its own entries still apply as defaults underneath.
                result[key] = merge_dicts(copy.deepcopy(value), master[key])
            else:
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
    Resolve a user config into the form the rest of the run reads.

    Steps performed:

    1. Load master.yaml (annotated reference with all known keys and the default
       values of the ``general`` and ``default_settings`` blocks).
    2. Warn + drop any unknown top-level or nested keys found in user_config.
    3. Replace every package-backed ``default_settings`` value the user did not
       state with the one read off the package parameter that consumes it
       (:func:`~experiment_runner.defaults.apply_package_defaults`).
    4. Overlay each section's ``default_settings`` onto every one of its list
       entries (:func:`_resolve_entries`).

    After this returns, **every list entry carries its effective settings**, so
    nothing downstream has to merge again or know which layer a value came from.
    Validation, the cache key and the pipeline all read final values.

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
        Resolved config: defaults applied, unknown keys removed, entries merged.
    """
    with open(master_path) as f:
        master = yaml.safe_load(f)

    # Warn + drop at the top level and recursively inside known sections
    resolved = _merge_warn_drop(user_config, master, "root", issues)
    apply_package_defaults(user_config, resolved)
    _resolve_entries(resolved)
    return resolved


def _resolve_entries(config: dict) -> None:
    """
    Overlay each section's ``default_settings`` onto every one of its entries.

    Modifies *config* in place. This is the one place the per-entry merge
    happens; the four component lists are walked here because the merge cannot be
    expressed as a single top-level one. :func:`~xai4tsc.utils.utils.merge_dicts`
    descends into dict values and treats a list as an opaque leaf - the same
    reason a user's list replaces master's catalogue wholesale - so nothing can
    push ``<section>.default_settings`` into the elements of its sibling list
    without walking that list. The four blocks also have disjoint key spaces, so
    there is no single block to merge from: ``train_split`` has no business in a
    model entry.

    Both sides are deep-copied, and both copies are needed.
    :func:`~xai4tsc.utils.utils.merge_dicts` writes into its first argument, so
    without copying the entry the raw config would be mutated; and for a key the
    entry does not have it inserts the default's own object, so without copying
    the defaults every entry that inherited ``samples`` or ``hyperparams`` would
    share one dict with the defaults block and with its siblings, and a later
    write to one would reach them all.

    Parameters
    ----------
    config : dict
        The config after master has been merged in, modified in place.
    """
    for section, list_key in _ENTRY_SECTIONS.values():
        block = config.get(section) or {}
        defaults = block.get("default_settings") or {}
        entries = block.get(list_key)
        if not isinstance(entries, list):
            continue

        resolved_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                # Left as-is for _validate_entries to report by position.
                resolved_entries.append(entry)
                continue
            resolved = merge_dicts(copy.deepcopy(entry), copy.deepcopy(defaults))
            # Set correct train values for checkpoints without training:
            # Presence of `train` on the raw entry decides whether a checkpoint
            # retrains, and the merge is about to make every entry carry it.
            if (
                section == "train_config"
                and "model_ckpt_path" in resolved
                and "train" not in entry
            ):
                resolved["train"] = None
            resolved_entries.append(resolved)
        # A new list rather than an in-place replacement: _merge_warn_drop copies
        # each section shallowly, so this list is still the user's own object.
        block[list_key] = resolved_entries


_LEGACY_TRAIN_VALUES = frozenset({"till_epoch", "for_epochs"})


def _normalize_train_switch(
    train_config: dict, issues: "ConfigIssues | None" = None
) -> None:
    """
    Normalise every ``train`` value in *train_config* to a bool or ``None``.

    ``train`` is a switch: any value read as true trains, ``false`` or absent does
    not.  The
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
    "method": frozenset({"call_params", "class_name", "class_path"}),
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


# Parameter names the framework supplies positionally at an explainer's call site,
# so a `call_params` block must never set them (the splat would raise).
_EXPLAIN_OWNED = frozenset({"model", "exp", "device", "targets"})


def strip_runner_keys(
    params: dict, method: str, config: dict | None = None, nested: bool = False
) -> dict:
    """
    Drop the runner's own entry keys from an explainer's construction parameters.

    An explainer takes its parameters at the entry's top level, so the dict handed
    to :func:`~xai4tsc.xai.explain.build_explainer` is the whole resolved entry:
    the identity key, the ``default_settings`` the runner reads itself, and the
    structural extras. :func:`~xai4tsc.utils.utils.dict_to_args` filters those out
    against a named signature, but it returns the dict **unchanged** for a
    constructor declaring ``**kwargs``, which would then receive ``method``,
    ``target``, ``samples`` and the rest as if they had been configured for it.

    A key the class itself names is kept, so an explainer that genuinely takes a
    parameter called ``samples`` still gets it; for every component that declares a
    named signature this function therefore changes nothing at all.

    The owned names are derived, never listed: the identity key, the resolved
    ``explanation_config.default_settings`` (which master's key whitelist produced)
    and :data:`_ENTRY_EXTRAS`. A wrapper's nested ``base`` is stripped the same way
    with one exception: ``method`` stays, because there it is not an identity key
    the runner reads but an input
    :class:`~xai4tsc.xai.base.WrapperExplainer` itself consumes - it pops that key
    to build the method it wraps, and raises ``KeyError`` without it.

    Parameters
    ----------
    params : dict
        The entry copy about to be used as construction parameters.
    method : str
        The explainer's registry name, used to look up what it declares.
    config : dict, optional
        The resolved config, read for the ``default_settings`` names. Without it
        only the identity key and the structural extras are stripped.
    nested : bool, optional
        ``True`` for a wrapper's ``base`` sub-dict, which keeps its ``method``.

    Returns
    -------
    dict
        A copy of *params* without the keys the runner owns.
    """
    from xai4tsc.xai.explain import EXPLAINERS

    owned = set(_ENTRY_EXTRAS["method"])
    if not nested:
        owned.add("method")
    if config:
        owned |= set(
            (config.get("explanation_config") or {}).get("default_settings") or {}
        )

    cls = EXPLAINERS.get(str(method).lower())
    named: set[str] = set()
    if cls is not None and "__init__" in cls.__dict__:
        named = set(inspect.signature(cls.__init__).parameters)

    stripped = {
        key: value for key, value in params.items() if key not in owned or key in named
    }
    base = stripped.get("base")
    if isinstance(base, dict):
        stripped["base"] = strip_runner_keys(
            base, base.get("method", ""), config, nested=True
        )
    return stripped


def _has_catch_all(func: object) -> bool:
    """
    Report whether *func* declares ``**kwargs``.

    Parameters
    ----------
    func : callable
        The signature to inspect.

    Returns
    -------
    bool
        ``True`` when a ``VAR_KEYWORD`` parameter is declared.
    """
    return any(
        param.kind == param.VAR_KEYWORD
        for param in inspect.signature(func).parameters.values()
    )


def _classify_params(
    block: dict, func: object, owned: frozenset[str] = frozenset()
) -> tuple[list[str], list[str], list[str]]:
    """
    Split a config parameter block's keys by what the target's signature says.

    The oracle is the set of **named** parameters, never the presence of
    ``**kwargs``. A catch-all sits beside named parameters often enough that
    conflating the two would reject keys that plainly work: every Quantus metric
    declares ``**kwargs`` *and* names between ten and nineteen parameters, so
    ``disable_warnings`` is verifiable even though the signature ends in a
    catch-all.

    Parameters
    ----------
    block : dict
        The parameter block as configured.
    func : callable
        The signature the block is handed to.
    owned : frozenset of str, optional
        Names the framework supplies itself, which a user must not set.

    Returns
    -------
    tuple of (list, list, list)
        ``(unverifiable, unknown, reserved)``. *unverifiable* keys match no named
        parameter but the target declares ``**kwargs``, so nothing can say whether
        they are read; *unknown* keys match nothing and there is no catch-all to
        absorb them; *reserved* keys collide with *owned*.
    """
    sig = inspect.signature(func)
    named = {
        param.name
        for param in sig.parameters.values()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    } - {"self"}
    catch_all = _has_catch_all(func)

    reserved = [key for key in block if key in owned]
    unmatched = [key for key in block if key not in named and key not in owned]
    if catch_all:
        return unmatched, [], reserved
    return [], unmatched, reserved


def _report_param_block(
    block: object,
    func: object,
    where: str,
    label: str,
    allow_kwargs: bool,
    issues: ConfigIssues,
    owned: frozenset[str] = frozenset(),
    dropped: bool = True,
    inherited: frozenset[str] = frozenset(),
) -> None:
    """
    Report the keys in one parameter block that the target will not read.

    Three outcomes, split by what the key would actually do:

    * **reserved** - the framework passes this itself, so setting it raises at the
      call site. Always an error.
    * **unverifiable** - the target's signature ends in ``**kwargs``, so no
      inspection can tell whether the key is read. An error naming
      ``general.allow_kwargs`` while that flag is false, and passed through in
      silence once it is true: a config that wants to reach a catch-all has to say
      so, because nothing can be promised about it afterwards.
    * **unknown** - nothing named matches and there is no catch-all. An error when
      the block is splatted straight in (it would raise), a warning when the value
      is filtered out first and the component falls back to its own default.

    Parameters
    ----------
    block : object
        The configured block, which need not be a mapping.
    func : callable
        The signature the block is handed to.
    where : str
        Config location, for the message.
    label : str
        How to name the target in a message, e.g. ``"this explainer's explain()"``.
    allow_kwargs : bool
        ``general.allow_kwargs``.
    issues : ConfigIssues
        Collector to append to.
    owned : frozenset of str, optional
        Names the framework supplies itself.
    dropped : bool, optional
        ``True`` when an unreadable key is filtered out before the call, which
        decides whether *unknown* is a warning or an error.
    inherited : frozenset of str, optional
        Keys this entry inherited from its section's ``default_settings``, which
        are not this entry's to answer for.
    """
    if block is None:
        # A documented-but-empty block, as master writes it. Nothing to check.
        return
    if not isinstance(block, dict):
        issues.error(
            where,
            f"expected a mapping of parameters, got "
            f"{type(block).__name__}; remove it or give it key/value pairs.",
        )
        return
    # Only what this entry itself declares. A key inherited from the section's
    # `default_settings` is a global choice, already checked against master's key
    # whitelist, and is deliberately absorbed by the entries it does not apply to:
    # `softmax` reaches every metric although only the perturbation ones read it.
    own = {key: value for key, value in block.items() if key not in inherited}
    if not own:
        return

    unverifiable, unknown, reserved = _classify_params(own, func, owned)
    accepted = set(inspect.signature(func).parameters) - {"self"} - owned

    for key in reserved:
        issues.error(
            where,
            f"'{key}' is supplied by the framework and must not be set here.",
        )
    for key in unknown:
        message = f"'{key}' is not a parameter of {label}.{_suggest(key, accepted)}"
        (issues.error if not dropped else issues.warn)(where, message)
    if allow_kwargs:
        return
    for key in unverifiable:
        issues.error(
            where,
            f"'{key}' cannot be verified: {label} declares **kwargs, so no "
            f"signature says whether it is read.{_suggest(key, accepted)} Set "
            "general.allow_kwargs: true to pass unchecked parameters through, or "
            "declare it as a named argument.",
        )


def _param_blocks(
    kind: str, name: str, cls: type | None, entry: dict
) -> list[tuple[str, object, str, frozenset[str], bool]]:
    """
    Return the parameter blocks of one entry, each with the signature it reaches.

    One table so every component type is checked the same way. Each row is
    ``(block key, callable, label, owned names, dropped)``; see
    :func:`_report_param_block` for what the last two decide. A block whose
    oracle cannot be resolved is left out rather than guessed at.

    The oracle is the signature the value actually arrives at, which for a Quantus
    metric is the library class rather than the adapter: ``QuantusEvaluator``
    declares ``**kwargs`` and re-filters per metric, so checking the adapter would
    verify nothing.

    Parameters
    ----------
    kind : str
        Entry identity key: ``"dataset"``, ``"model"``, ``"method"`` or
        ``"metric"``.
    name : str
        The component's configured name.
    cls : type or None
        The registry class, where the registry has one.
    entry : dict
        The resolved entry.

    Returns
    -------
    list of tuple
        One row per parameter block present on *entry*.
    """
    blocks: list[tuple[str, object, str, frozenset[str], bool]] = []
    if kind == "method" and cls is not None:
        if "call_params" in entry:
            blocks.append(
                (
                    "call_params",
                    cls.explain,
                    "this explainer's explain()",
                    _EXPLAIN_OWNED,
                    False,
                )
            )
    elif kind == "model" and cls is not None and "init_params" in entry:
        blocks.append(
            ("init_params", cls.__init__, "this model's constructor", frozenset(), True)
        )
    elif kind == "dataset":
        from xai4tsc.data.datasets import SYNTHETIC_DATASETS

        synthetic = SYNTHETIC_DATASETS.get(name)
        if synthetic is not None and "init_params" in entry:
            blocks.append(
                (
                    "init_params",
                    synthetic.__init__,
                    "this dataset's constructor",
                    frozenset(),
                    False,
                )
            )
    elif kind == "metric":
        from xai4tsc.evaluation.evaluate import METRICS, QUANTUS_METRICS

        quantus = QUANTUS_METRICS.get(name)
        target = quantus if quantus is not None else METRICS.get(name)
        if isinstance(target, type):
            if "metric_class_params" in entry:
                blocks.append(
                    (
                        "metric_class_params",
                        target.__init__,
                        "this metric's constructor",
                        frozenset(),
                        True,
                    )
                )
            if "metric_call_params" in entry:
                call = target.__call__ if quantus is not None else target.evaluate
                blocks.append(
                    (
                        "metric_call_params",
                        call,
                        "this metric at call time",
                        frozenset(),
                        True,
                    )
                )
    return blocks


def _validate_entry(
    entry: dict,
    kind: str,
    where: str,
    defaults: dict,
    registry: dict | None,
    issues: ConfigIssues,
    allow_kwargs: bool = False,
) -> None:
    """Check one list entry's name and keys; see :func:`config_sanity_check`."""
    overridable = set(defaults)
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

    # An explainer takes its construction parameters at the entry's top level, so a
    # catch-all __init__ makes an unmatched key unverifiable rather than unknown:
    # dict_to_args hands the whole entry to a VAR_KEYWORD constructor.
    # `build_explainer` passes params only when the class defines its own
    # __init__; for one that does not, nothing is handed over and an unmatched key
    # is plainly unknown rather than unverifiable.
    catch_all_init = (
        kind == "method"
        and cls is not None
        and "__init__" in cls.__dict__
        and _has_catch_all(cls.__init__)
    )

    for key in entry:
        if key in accepted:
            continue
        if kind in _PARAM_HOME and (key in params or kind == "metric"):
            issues.warn(
                where,
                f"'{key}' is not read at an entry's top level; "
                f"{'parameters of this ' + kind} go under {_PARAM_HOME[kind]}.",
            )
        elif catch_all_init and not allow_kwargs:
            issues.error(
                where,
                f"'{key}' cannot be verified: this explainer's constructor declares "
                f"**kwargs, so no signature says whether it is read."
                f"{_suggest(key, accepted)} Set general.allow_kwargs: true to pass "
                "unchecked parameters through, or declare it as a named argument.",
            )
        elif not catch_all_init:
            label = _ENTRY_LABEL[kind]
            issues.warn(
                where, f"'{key}' is not a known {label}.{_suggest(key, accepted)}"
            )

    for block_key, func, label, owned, dropped in _param_blocks(
        kind, name if isinstance(name, str) else "", cls, entry
    ):
        _report_param_block(
            entry[block_key],
            func,
            f"{where}.{block_key}",
            label,
            allow_kwargs,
            issues,
            owned,
            dropped,
            frozenset(defaults.get(block_key) or ()),
        )

    # A wrapper explainer nests the method it wraps, with its own parameters.
    if kind == "method" and isinstance(entry.get("base"), dict):
        _validate_entry(
            entry["base"],
            kind,
            f"{where}.base",
            defaults,
            registry,
            issues,
            allow_kwargs,
        )


def _validate_entries(config: dict, issues: ConfigIssues) -> None:
    """Validate every component list entry's name and keys."""
    from xai4tsc.evaluation.evaluate import METRICS
    from xai4tsc.models.models import MODELS
    from xai4tsc.xai.explain import EXPLAINERS

    registries = {"model": MODELS, "method": EXPLAINERS, "metric": METRICS}
    allow_kwargs = bool((config.get("general") or {}).get("allow_kwargs", False))
    for kind, (section, list_key) in _ENTRY_SECTIONS.items():
        block = config.get(section) or {}
        # The resolved default_settings is exactly the set of keys an entry may
        # override: _merge_warn_drop has already dropped anything master
        # does not declare.
        defaults = block.get("default_settings") or {}
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
                defaults,
                registries.get(kind),
                issues,
                allow_kwargs,
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


def _validate_general_seeding(general: dict, issues: ConfigIssues) -> None:
    """
    Check that ``general.seed`` and ``general.reproducible`` have usable types.

    Both reach the RNG unchecked otherwise: a quoted ``seed: "2025"`` raises deep
    inside numpy rather than at config time, and ``reproducible: "false"`` is a
    non-empty string, so it is read as true and silently turns seeding back on.

    Parameters
    ----------
    general : dict
        The resolved ``general`` section.
    issues : ConfigIssues
        Collector to append to.
    """
    seed = general.get("seed")
    # bool is an int subclass, and `seed: true` is a mistake rather than seed 1.
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        issues.error(
            "general.seed",
            f"must be an integer or null, got {type(seed).__name__} ({seed!r}). "
            "Every result's seed is derived from this value.",
        )

    reproducible = general.get("reproducible")
    if reproducible is not None and not isinstance(reproducible, bool):
        issues.error(
            "general.reproducible",
            f"must be true or false, got {type(reproducible).__name__} "
            f'({reproducible!r}). A quoted "false" is a non-empty string and '
            "would be read as true.",
        )


_SAMPLE_TYPES = ("index", "random", "all")
"""Accepted ``samples.type`` values, in the order ``master.yaml`` documents them."""


def _validate_data_splits(config: dict, issues: ConfigIssues) -> None:
    """
    Check that each dataset's split fractions can produce three splits.

    It is pure arithmetic on config values, so it is caught here instead before a
    dataset is even downloaded. Entries using the archive partition or an already-split
    directory never reach the splitter, so their fractions are not checked here; they
    are reported as ignored when the split runs.

    Parameters
    ----------
    config : dict
        The resolved config.
    issues : ConfigIssues
        Collector to append to.
    """
    section = config.get("data_config") or {}
    for index, entry in enumerate(section.get("datasets") or []):
        if not isinstance(entry, dict):
            continue
        if entry.get("use_predefined_splits") or entry.get("is_presplit"):
            continue

        name = entry.get("dataset", "?")
        where = f"data_config.datasets[{index}] ('{name}')"
        train_split = entry.get("train_split")
        val_split = entry.get("val_split")
        if not isinstance(train_split, (int, float)) or isinstance(train_split, bool):
            continue
        if not isinstance(val_split, (int, float)) or isinstance(val_split, bool):
            continue

        if not 0 < train_split < 1:
            issues.error(
                where,
                f"train_split must be between 0 and 1, got {train_split!r}.",
            )
        elif train_split + val_split >= 1:
            issues.error(
                where,
                f"train_split ({train_split:.4g}) + val_split ({val_split:.4g}) "
                f"= {train_split + val_split:.4g} leaves nothing for the test "
                "split; together they must stay below 1.",
            )
        elif val_split < 0:
            issues.error(where, f"val_split cannot be negative, got {val_split!r}.")


def _validate_encoders(config: dict, issues: ConfigIssues) -> None:
    """
    Check that each dataset's ``encode`` names an encoder that can be built.

    Resolving it here means a retired or misspelled name fails before anything is
    downloaded. It used to fall back to ``LabelEncoder`` in silence, which for
    ``encode: "multhot"`` trained a different problem and reported plausible
    numbers for it.

    Parameters
    ----------
    config : dict
        The resolved config.
    issues : ConfigIssues
        Collector to append to.
    """
    section = config.get("data_config") or {}
    for index, entry in enumerate(section.get("datasets") or []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("encode")
        try:
            build_encoder(name, entry.get("encode_params"))
        except ValueError as exc:
            issues.error(
                f"data_config.datasets[{index}] ('{entry.get('dataset', '?')}')",
                str(exc),
            )


def _validate_perturbation_softmax(config: dict, issues: ConfigIssues) -> None:
    """
    Warn when perturbation metrics would softmax a per-output model's outputs.

    ``metric_call_params.softmax`` makes the perturbation metrics normalise the
    model's output before reading the explained class's score. Over a
    one-of-K head that is right. Over independent per-output heads
    (``encode: "multihot"``, trained with BCE) it is not: the outputs do not
    compete, so renormalising them across the row turns one label's confidence
    into a function of the others and moves every perturbation score.

    A warning rather than an error, because the flag is also meaningful for a
    model whose ``forward`` already ends in a softmax, which the config cannot
    tell apart from a mistake. Silence is what would be wrong: nothing about the
    resulting numbers looks off.

    Parameters
    ----------
    config : dict
        The resolved config.
    issues : ConfigIssues
        Collector to append to.
    """
    evaluation = config.get("evaluation_config") or {}
    call_params = (evaluation.get("default_settings") or {}).get(
        "metric_call_params"
    ) or {}
    if not call_params.get("softmax"):
        return
    datasets = (config.get("data_config") or {}).get("datasets") or []
    per_output = sorted(
        {
            str(entry.get("dataset", "?"))
            for entry in datasets
            if isinstance(entry, dict) and str(entry.get("encode", "")) == "multihot"
        }
    )
    if per_output:
        issues.warn(
            "evaluation_config.default_settings.metric_call_params.softmax",
            f"true, but {', '.join(per_output)} uses encode: 'multihot', whose "
            "outputs are independent per-label probabilities rather than a "
            "distribution over classes. Renormalising them moves every "
            "perturbation metric's score. Set it to false for a per-output "
            "formulation",
        )


def _validate_training_lookups(config: dict, issues: ConfigIssues) -> None:
    """
    Check that each model entry's loss and optimizer can be built.

    The counterpart to :func:`_validate_encoders`: resolving the names here means
    a misspelling, a retired name or a bad constructor argument fails before a
    dataset is downloaded or a model is instantiated, instead of part-way into a
    training run.

    Parameters
    ----------
    config : dict
        The resolved config, whose model entries already carry their merged
        ``hyperparams``.
    issues : ConfigIssues
        Collector to append to.
    """
    for index, entry in enumerate(
        (config.get("train_config") or {}).get("models") or []
    ):
        if not isinstance(entry, dict):
            continue
        hyperparams = entry.get("hyperparams")
        if not isinstance(hyperparams, dict):
            continue
        where = f"train_config.models[{index}] ('{entry.get('model', '?')}')"
        for build, name_key, params_key in (
            (build_loss, "loss_func", "loss_params"),
            (build_optimizer, "optimizer", "optimizer_params"),
        ):
            try:
                build(hyperparams.get(name_key), hyperparams.get(params_key))
            except ValueError as exc:
                issues.error(where, str(exc))


def _validate_explanation_samples(config: dict, issues: ConfigIssues) -> None:
    """
    Check each explainer's ``samples`` block as far as the data allows.

    Only the data-independent part of ``indices`` is checked here, because
    ``config_sanity_check`` runs before any dataset is loaded and so cannot know
    how many test samples there will be. The bound itself is checked once the
    split exists.

    Parameters
    ----------
    config : dict
        The resolved config.
    issues : ConfigIssues
        Collector to append to.
    """
    section = config.get("explanation_config") or {}
    for index, entry in enumerate(section.get("explainers") or []):
        if not isinstance(entry, dict):
            continue
        method = entry.get("method", "?")
        where = f"explanation_config.explainers[{index}] ('{method}')"
        samples = entry.get("samples")
        if not isinstance(samples, dict):
            continue

        sample_type = samples.get("type")
        if sample_type not in _SAMPLE_TYPES:
            issues.error(
                f"{where}.samples.type",
                f"must be one of {list(_SAMPLE_TYPES)}, got {sample_type!r}."
                f"{_suggest(str(sample_type), _SAMPLE_TYPES)} An unrecognised "
                "value would explain the whole test split, which is also the "
                "subset every metric would then score.",
            )
            continue

        if sample_type == "index":
            indices = samples.get("indices")
            if not isinstance(indices, list) or not indices:
                issues.error(
                    f"{where}.samples.indices",
                    "must be a non-empty list of sample indices when "
                    f"samples.type is 'index', got {indices!r}.",
                )
                continue
            bad = [
                i
                for i in indices
                if isinstance(i, bool) or not isinstance(i, int) or i < 0
            ]
            if bad:
                issues.error(
                    f"{where}.samples.indices",
                    f"must contain non-negative integers; {bad} is not usable.",
                )
        elif sample_type == "random":
            count = samples.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                issues.error(
                    f"{where}.samples.count",
                    "must be a positive integer when samples.type is "
                    f"'random', got {count!r}.",
                )


def _validate_hyperparams(config: dict, issues: ConfigIssues) -> None:
    """
    Check every model entry's ``hyperparams`` block against what training takes.

    The block is handed to
    :meth:`~xai4tsc.models.base.ModelBase.train_model` as keyword arguments - all
    but ``loss_params`` and ``optimizer_params``, which the runner consumes when
    it builds those two objects - so an unrecognised key raises there. Reporting
    it here instead names the entry and offers the near-miss, and does so before
    a dataset is downloaded. It is
    an error rather than a warning because the keys decide how the model trains:
    a silently dropped ``epochs`` moves every number the run produces.

    The accepted names come from :func:`~experiment_runner.defaults.hyperparam_names`,
    which derives them from the pointer table, so they cannot drift from the
    signature.

    Parameters
    ----------
    config : dict
        The resolved config.
    issues : ConfigIssues
        Collector to append to.
    """
    accepted = hyperparam_names()
    for index, entry in enumerate(
        (config.get("train_config") or {}).get("models") or []
    ):
        if not isinstance(entry, dict):
            continue
        hyperparams = entry.get("hyperparams")
        if not isinstance(hyperparams, dict):
            continue
        where = (
            f"train_config.models[{index}] ('{entry.get('model', '?')}').hyperparams"
        )
        for key in hyperparams:
            if key not in accepted:
                issues.error(
                    where,
                    f"'{key}' is not a training parameter.{_suggest(key, accepted)}",
                )


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
    _validate_general_seeding(config.get("general", {}), issues)
    _validate_data_splits(config, issues)
    _validate_encoders(config, issues)
    _validate_perturbation_softmax(config, issues)
    _validate_explanation_samples(config, issues)
    _validate_hyperparams(config, issues)
    _validate_training_lookups(config, issues)

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
