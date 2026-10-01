"""
Pure helpers for the experiment runner.

Small, side-effect-free utilities shared by the runner entry point: archive
wildcard expansion, metric-score aggregation, and the ``metrics.csv`` row and
log-label formatting. Kept apart from :mod:`experiment_runner.main` so the
orchestration module holds only the pipeline itself.

This is the runner's own helper module and is unrelated to
:mod:`xai4tsc.utils`, which serves the importable package.
"""

import copy
import logging

import numpy as np
from sktime.datasets.tsc_dataset_names import (
    multivariate as _UEA_NAMES,  # noqa: N812 — uppercase denotes a constant
)
from sktime.datasets.tsc_dataset_names import (
    univariate as _UCR_NAMES,  # noqa: N812 — uppercase denotes a constant
)

from xai4tsc.utils.utils import merge_dicts

logger = logging.getLogger("xai4tsc.runner.utils")


# Reserved keywords that expand to all datasets in the corresponding archive.
_ARCHIVE_KEYWORDS = {"UCR": _UCR_NAMES, "UEA": _UEA_NAMES}


def expand_datasets(datasets: list[dict]) -> list[dict]:
    """
    Expand archive wildcards and resolve per-member ``overrides``.

    Each entry is either a concrete dataset or an archive keyword
    (``UCR`` / ``UEA``) that expands to all its member datasets. An entry's
    inline settings apply to every member it produces; an optional
    ``overrides`` map (``{dataset_name: {setting: value}}``) sets per-member
    exceptions that take precedence over the entry's inline settings.

    Parameters
    ----------
    datasets : list[dict]
        Raw ``data_config["datasets"]`` entries.

    Returns
    -------
    list[dict]
        Concrete dataset entries with ``overrides`` resolved and stripped.
    """
    expanded: list[dict] = []
    for entry in datasets:
        overrides = entry.get("overrides") or {}
        base = {k: v for k, v in entry.items() if k != "overrides"}
        archive = _ARCHIVE_KEYWORDS.get(base.get("dataset"))
        if archive is not None:
            for name in set(overrides) - set(archive):
                logger.warning(
                    "Ignoring override for '%s': not a member of archive '%s'.",
                    name,
                    base["dataset"],
                )
            for name in archive:
                member = {**base, "dataset": name}
                if name in overrides:
                    # overrides[name] wins over the entry's inline settings.
                    member = merge_dicts(copy.deepcopy(overrides[name]), member)
                expanded.append(member)
        else:
            if overrides:
                logger.warning(
                    "Ignoring 'overrides' on non-archive dataset '%s'; "
                    "set per-dataset values inline instead.",
                    base.get("dataset"),
                )
            expanded.append(base)
    return expanded


def _aggregate_metric(evaluation: object) -> float | None:
    """
    Reduce a Quantus metric result to a single float.

    Most Quantus metrics return one score per explained instance; average them
    so the recorded value reflects the whole sample, not just the first element.
    Scalars pass through unchanged. Returns ``None`` for empty/all-NaN results.
    """
    if evaluation is None:
        return None
    arr = np.asarray(evaluation, dtype=float).ravel()
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


_METRICS_COLUMNS = (
    "dataset",
    "model",
    "explainer",
    "metric",
    "status",
    "score",
    "detail",
)


def _result_row(
    dataset_id: str,
    model_id: str,
    explainer_id: str,
    metric_id: str,
    status: str,
    score: float | None = None,
    detail: str = "",
) -> dict:
    """
    Build one ``metrics.csv`` row.

    Every configured (dataset, model, explainer, metric) combination the runner
    actually attempts gets a row, so a result file always accounts for the whole
    configuration rather than only its successes.

    Parameters
    ----------
    dataset_id, model_id, explainer_id, metric_id : str
        Result identities (see :func:`~experiment_runner.config.disambiguate`),
        not raw registry names.
    status : str
        One of ``"ok"`` (a score was produced), ``"nan"`` (the metric ran but
        could not produce a number), ``"error"`` (the metric raised) or
        ``"skipped-domain"`` (the metric requires an explanation domain this
        explainer does not produce).
    score : float or None
        The aggregated score; ``None`` for any status other than ``"ok"`` and
        rendered as an empty cell.
    detail : str
        Why, for a non-``ok`` status. Flattened to a single line so one result
        stays one CSV row.

    Returns
    -------
    dict
        A row matching the ``metrics.csv`` schema.
    """
    flat = " ".join(str(detail).split())
    return {
        "dataset": dataset_id,
        "model": model_id,
        "explainer": explainer_id,
        "metric": metric_id,
        "status": status,
        "score": score,
        "detail": flat[:300],
    }


def _format_label_list(labels: list[str], limit: int = 10) -> str:
    """
    Join outcome labels for a single log line, truncating long lists.

    An archive wildcard expands to 128 (UCR) or 33 (UEA) entries, and a broad
    metric list multiplies out per model and explainer, so a run that skips or
    fails many of them would otherwise emit a line thousands of characters wide.

    Parameters
    ----------
    labels : list of str
        Outcome labels, already formatted for display.
    limit : int
        Maximum number of labels to name before summarising the remainder.

    Returns
    -------
    str
        Comma-separated labels, suffixed with ``, ... and N more`` when there
        are more than *limit* of them.
    """
    if len(labels) <= limit:
        return ", ".join(labels)
    return f"{', '.join(labels[:limit])}, ... and {len(labels) - limit} more"
