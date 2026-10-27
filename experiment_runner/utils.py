"""
Pure helpers for the experiment runner.

Small, side-effect-free utilities shared by the runner entry point: archive
wildcard expansion, metric-score aggregation, the ``metrics.csv`` and
``models.csv`` row builders, and log-label formatting. Kept apart from
:mod:`experiment_runner.main` so the orchestration module holds only the
pipeline itself.

This is the runner's own helper module and is unrelated to
:mod:`xai4tsc.utils`, which serves the importable package.
"""

import logging

import numpy as np
import pandas as pd
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

    Runs after :func:`~experiment_runner.config.config_resolve`, so each entry
    already carries its effective settings, and after
    :func:`~experiment_runner.config.config_sanity_check`, so one mistake in a
    wildcard entry is reported once rather than once per member. Expansion
    preserves the resolved-entry invariant: a member is built from an
    already-resolved base, and an override still wins because it is the priority
    dict of the merge below.

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
                    member = merge_dicts(overrides[name], member)
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
    "target",
    "samples",
    "seed",
    "status",
    "score",
    "detail",
)


_INTEGER_COLUMNS = frozenset({"epoch", "samples", "outputs", "pos", "neg", "seed"})
"""
Columns of either result file that count something and cannot be fractional.

Held in one place because both files have gaps in them for the same reason: a
row that does not define a column leaves the cell empty.
"""


def _as_integer_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Cast the counting columns of a result frame to a nullable integer dtype.

    A row with no value for a column - a failed model, a formulation that does
    not define a metric, an unseeded run - leaves a gap, and pandas represents a
    gap in an ``int64`` column by widening the column to ``float64``. The counts
    then reach the CSV as ``9.0`` and ``1.0``, reading as measurements precise to
    one decimal rather than as a number of samples or an epoch. ``Int64`` holds
    the gap itself, so a count stays a count and the empty cell stays empty.

    Done per column and tolerant of failure: these files are written once, at the
    end of a run, so a column that will not cast - a custom summary putting a
    fractional value under one of these names - is left as it was and warned
    about rather than taking the whole result file down with it.

    Parameters
    ----------
    frame : pd.DataFrame
        A frame of :func:`_result_row` or :func:`_model_row` rows.

    Returns
    -------
    pd.DataFrame
        The same frame, with every present member of
        :data:`_INTEGER_COLUMNS` cast where the cast succeeded.
    """
    for column in frame.columns:
        if column not in _INTEGER_COLUMNS:
            continue
        try:
            frame[column] = frame[column].astype("Int64")
        except (TypeError, ValueError):
            logger.warning(
                "Column '%s' holds values that are not whole numbers, so it is "
                "written as-is rather than as a count.",
                column,
            )
    return frame


def _result_row(
    dataset_id: str,
    model_id: str,
    explainer_id: str,
    metric_id: str,
    status: str,
    score: float | None = None,
    detail: str = "",
    seed: int | None = None,
    target: str = "",
    samples: int | None = None,
) -> dict:
    """
    Build one ``metrics.csv`` row.

    Every configured (dataset, model, explainer, metric) combination the runner
    actually attempts gets a row, so a result file always accounts for the whole
    configuration rather than only its successes.

    Parameters
    ----------
    dataset_id, model_id, explainer_id, metric_id : str
        Result identities (see
        :func:`~experiment_runner.config.result_identities`),
        not raw registry names.
    status : str
        One of ``"ok"`` (a score was produced), ``"nan"`` (the metric ran but
        could not produce a number), ``"error"`` (the metric raised),
        ``"skipped-domain"`` (the metric requires an explanation domain this
        explainer does not produce), ``"skipped-no-ground-truth"`` (the metric
        localizes against ground truth the dataset does not provide) or
        ``"skipped-no-targets"`` (no explained sample carries this target).
    score : float or None
        The aggregated score; ``None`` for any status other than ``"ok"`` and
        rendered as an empty cell.
    detail : str
        Why, for a non-``ok`` status. Flattened to a single line so one result
        stays one CSV row.
    seed : int or None
        The derived seed this result was computed with, so a row can be
        reproduced from the row alone. Empty under ``reproducible: false``,
        where no seed was applied.
    target : str
        The class this row's attributions explain, when the explanation holds
        one per class (``targets="all"``, or a per-output encoding). Empty for a
        single-target explanation, where the explained class varies per sample.
        A row per target rather than their mean is what makes a per-class spread
        visible at all.
    samples : int or None
        How many samples were handed to the metric. Not decoration: under a
        per-output encoding this varies per target and, with
        ``targets="predicted"``, with the model and the decision threshold too,
        so two scores without their denominators are not comparable. It is an
        upper bound rather than the exact count, because a ground-truth metric
        drops samples with no region of its own accord - which is also why those
        metrics score the same with or without a target mask, while the
        perturbation metrics do not.

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
        "target": target,
        "samples": samples,
        "seed": seed,
        "status": status,
        "score": score,
        "detail": flat[:300],
    }


_MODEL_COLUMNS = (
    "dataset",
    "model",
    "epoch",
    "status",
    "detail",
)
"""
The identity prefix of ``models.csv``; the metric columns follow it.

Which metrics those are depends on the formulation -
:meth:`~xai4tsc.models.base.ModelBase.evaluate_model` reports ``pos`` / ``neg``
and the positive class's rates for a binary problem, macro rates for a
multiclass one, and subset / per-output accuracy plus the Hamming loss for a
multi-label one. So the columns are a union over the rows actually produced
rather than a fixed schema, and a cell is empty where that formulation does not
define the metric.
"""


def _model_row(
    dataset_id: str,
    model_id: str,
    status: str,
    summary: dict | None = None,
    epoch: int | None = None,
    detail: str = "",
) -> dict:
    """
    Build one ``models.csv`` row from an ``evaluate_model`` summary.

    This is test-set *model performance*, which is a different measurement from
    the XAI metric scores in ``metrics.csv``: one row per (dataset, model)
    rather than per explanation, and its own columns. The two files are kept
    apart for that reason rather than merged.

    Every model the runner builds is scored on the test split, trained in this
    run or not, and one whose iteration raised gets a ``failed`` row rather
    than none, so the file accounts for the whole configuration the way
    ``metrics.csv`` does.

    Parameters
    ----------
    dataset_id, model_id : str
        Result identities (see
        :func:`~experiment_runner.config.result_identities`), not raw registry
        names - so two entries of the same model are distinguishable.
    status : str
        ``"ok"`` for a scored model, or ``"failed"`` for one whose iteration
        raised and was skipped. A non-``ok`` row explains itself in *detail*.
    summary : dict, optional
        What :meth:`~xai4tsc.models.base.ModelBase.evaluate_model` returned.
        Its keys become columns. Identity columns win on a name collision.
    epoch : int or None
        The epoch the evaluated weights come from (``best_epoch``), empty when
        none is known. The model's own result files are named after
        :attr:`~xai4tsc.models.base.ModelBase.epoch_label`, which is ``0`` in
        that case rather than empty.
    detail : str
        Why, for a non-``ok`` status. Flattened to a single line.

    Returns
    -------
    dict
        A row for the ``models.csv`` schema.
    """
    flat = " ".join(str(detail).split())
    return {
        **(summary or {}),
        "dataset": dataset_id,
        "model": model_id,
        "epoch": epoch,
        "status": status,
        "detail": flat[:300],
    }


def _model_frame(rows: list[dict]) -> pd.DataFrame:
    """
    Collect model rows into a frame with the identity columns first.

    The metric columns vary by formulation, so they are ordered by first
    appearance rather than sorted: that keeps each summary's own order
    (``samples``, then accuracy, then the rates) readable instead of
    interleaving unrelated measurements alphabetically.

    Parameters
    ----------
    rows : list of dict
        Rows from :func:`_model_row`.

    Returns
    -------
    pd.DataFrame
        One row per model, columns being :data:`_MODEL_COLUMNS` followed by
        every metric any row carried.
    """
    ordered = list(_MODEL_COLUMNS)
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return _as_integer_columns(pd.DataFrame(rows, columns=ordered))


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


def _check_sample_indices(
    explanation_config: dict, n_test: int, dataset_id: str
) -> None:
    """
    Check configured sample indices against the test split.

    ``config_sanity_check`` runs before any dataset is loaded, so it can only
    check that the indices are non-negative integers.  The bound needs the split,
    and the split exists before any model trains - which is where this runs, so a
    typo costs seconds instead of a training run plus a bare ``IndexError`` from
    inside the explainer.

    Parameters
    ----------
    explanation_config : dict
        The resolved ``explanation_config`` section.
    n_test : int
        Number of samples in the test split.
    dataset_id : str
        Identity of the dataset being set up, for the error message.

    Raises
    ------
    ValueError
        If any explainer requests an index outside the test split.
    """
    for index, entry in enumerate(explanation_config.get("explainers") or []):
        if not isinstance(entry, dict):
            continue
        samples = entry.get("samples")
        if not isinstance(samples, dict) or samples.get("type") != "index":
            continue
        indices = samples.get("indices")
        if not isinstance(indices, list):
            continue
        out_of_range = [i for i in indices if isinstance(i, int) and i >= n_test]
        if out_of_range:
            method = entry.get("method", "?")
            msg = (
                f"Dataset '{dataset_id}': explanation_config.explainers[{index}] "
                f"('{method}') requests sample indices {out_of_range}, but the "
                f"test split has {n_test} samples (valid: 0-{n_test - 1})."
            )
            logger.error(msg)
            raise ValueError(msg)
