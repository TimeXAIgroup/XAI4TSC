"""
Runner adapter for :mod:`xai4tsc.evaluation`.

Unpacks metric configuration dicts and delegates to the package
:func:`~xai4tsc.evaluation.evaluate` function.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from xai4tsc.evaluation import evaluate
from xai4tsc.evaluation.evaluate import METRICS
from xai4tsc.utils.utils import dict_to_args

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from torch import nn

    from xai4tsc.xai._types import Explanation

logger = logging.getLogger("xai4tsc.runner.evaluate")


def _evaluate(
    model: nn.Module,
    metric_conf: dict,
    explanation: Explanation,
    data: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    device: str = "cpu",
) -> float | np.ndarray | None:
    """
    Runner wrapper: unpack metric_conf dict and delegate to evaluate().

    Parameters
    ----------
    model :
        The PyTorch model being evaluated.
    metric_conf : dict
        Metric configuration dict with keys ``"metric"``,
        ``"metric_class_params"``, and ``"metric_call_params"``.
    explanation : Explanation
        Explanation dataclass containing ``exp_values`` and metadata.
    data : np.ndarray, optional
        Input samples for the metric.
    labels : np.ndarray, optional
        Ground-truth labels.
    device : str
        Compute device.

    Returns
    -------
    float or np.ndarray or None
        Metric score returned by :func:`~xai4tsc.evaluation.evaluate`.
    """
    return evaluate(
        model,
        metric_conf["metric"],
        explanation=explanation,
        data=data,
        labels=labels,
        metric_class_params=metric_conf["metric_class_params"],
        metric_call_params=metric_conf["metric_call_params"],
        device=device,
    )


def _animate(
    model: nn.Module,
    metric_conf: dict,
    explanation: Explanation,
    data: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
) -> Path | None:
    """
    Render a perturbation-process GIF for a metric that supports animation.

    Only the perturbation metrics (frequency / time-frequency) expose an
    ``animate`` method; for any other metric this is a no-op. The metric is built
    with the explanation's domain transform and the configured
    ``metric_class_params`` (filtered to its constructor).

    Parameters
    ----------
    model :
        The model being explained.
    metric_conf : dict
        Metric configuration (``"metric"``, optional ``"metric_class_params"``,
        optional ``"animate_sample"`` / ``"animate_fps"``).
    explanation : Explanation
        The explanation to animate (provides ``transform`` and ``exp_values``).
    data, labels : np.ndarray
        The samples and target labels the explanation was computed on.
    save_path : Path
        Output GIF path.

    Returns
    -------
    Path or None
        The written GIF path, or ``None`` if the metric does not animate.
    """
    metric_cls = METRICS.get(metric_conf["metric"])
    if metric_cls is None or not hasattr(metric_cls, "animate"):
        logger.warning(
            "Metric '%s' does not support animation; skipping.", metric_conf["metric"]
        )
        return None

    init_params = dict(metric_conf.get("metric_class_params", {}))
    init_params["transform"] = explanation.transform
    init_params = dict_to_args(init_params, metric_cls.__init__)
    metric = metric_cls(**(init_params or {}))

    return metric.animate(
        model,
        data,
        labels,
        explanation.exp_values,
        save_path=save_path,
        sample=metric_conf.get("animate_sample", 0),
        fps=metric_conf.get("animate_fps", 10),
    )
