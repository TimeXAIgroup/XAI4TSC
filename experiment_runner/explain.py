"""
Runner adapter for :mod:`xai4tsc.xai`.

Handles sample selection and optional saving of raw explanation values,
then delegates to the package :func:`~xai4tsc.xai.generate_explanation`
function.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from xai4tsc.xai.explain import generate_explanation

if TYPE_CHECKING:
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

    from xai4tsc.models.base import ModelBase
    from xai4tsc.xai._types import Explanation

logger = logging.getLogger("xai4tsc.runner.explain")


def _explainer_display_name(explainer_config: dict) -> str:
    """
    Return ``"WRAPPER-BASE"`` for wrapper explainers, else the method name.

    Wrapper entries carry a nested ``base`` config; surfacing the base method
    distinguishes e.g. ``SIGN-Integrated_Gradients`` from ``SIGN-DeepLift`` in
    logs and result folders (and avoids same-name folders colliding).

    Parameters
    ----------
    explainer_config : dict
        A single explainer entry from ``explanation_config.explainers``.

    Returns
    -------
    str
        ``"<method>-<base method>"`` when a nested ``base`` method is present,
        otherwise ``"<method>"``.
    """
    method = explainer_config["method"]
    base = explainer_config.get("base")
    if isinstance(base, dict) and base.get("method"):
        return f"{str(method).upper()}-{base['method']}"
    return method


def _resolve_background_data(
    selector: str | None, splits: dict[str, np.ndarray | None] | None
) -> np.ndarray | None:
    """
    Resolve a YAML ``background_data`` selector to a split array.

    Selectors are a runner-only concept (an ndarray cannot live in YAML): the
    table maps ``"train_set"`` / ``"test_set"`` / ``"val_set"`` to the arrays in
    scope at the call site.  Unknown selectors warn and resolve to ``None`` so
    the package falls back to a baseline that needs no background.

    Parameters
    ----------
    selector : str or None
        The selector string from the explainer config.
    splits : dict or None
        Mapping of selector name to the corresponding data array.

    Returns
    -------
    np.ndarray or None
        The selected array, or ``None`` when unset or unresolvable.
    """
    if selector is None:
        return None
    if not isinstance(selector, str):
        # Already an array (library-style use) — pass through unchanged.
        return selector
    table = splits or {}
    if selector in table:
        return table[selector]
    logger.warning(
        "Unknown background_data selector '%s'; using no background.", selector
    )
    return None


def _generate_explanation(
    model: ModelBase,
    data: np.ndarray,
    labels: np.ndarray,
    encoder: OneHotEncoder | LabelEncoder | OrdinalEncoder,
    prng: np.random.Generator | None = None,
    device: str = "cpu",
    config: dict | None = None,
    explainer_config: dict | None = None,
    data_config: dict | None = None,
    save_path: Path | None = None,
    splits: dict[str, np.ndarray | None] | None = None,
    metadata: list | None = None,
) -> Explanation:
    """
    Runner wrapper: unpacks config dicts and delegates to generate_explanation().

    Parameters
    ----------
    model : ModelBase
        Model to use
    data : np.ndarray
        Whole dataset
    labels : np.ndarray, optional
        All labels
    encoder : OneHotEncoder | LabelEncoder | OrdinalEncoder
        Encoder used to decode labels when saving explanation values.
    prng : np.random.Generator, optional
        Random number generator to use, by default None
    device : str, optional
        Compute device, by default ``"cpu"``.
    config : dict, optional
        General configuration, by default None
    explainer_config : dict, optional
        Explainer configuration, by default None
    data_config : dict, optional
        Dataset configuration, by default None
    save_path : Path, optional
        Base directory for saved artifacts; falls back to the configured
        results path when ``None``.
    splits : dict, optional
        Mapping of selector name (``train_set`` / ``test_set`` / ``val_set``) to
        data array, used to resolve a ``background_data`` selector for explainers
        that need a background set (e.g. TSHAP).
    metadata : list, optional
        Per-sample test-split metadata (ground-truth localization). When present,
        the subset for the explained ``indices`` is attached to
        ``explanation.metadata`` for ground-truth metrics (e.g. TimeFrequencyAUC).

    Returns
    -------
    Explanation
        An object of the Explanation dataclass.
    """
    # Resolve runtime-only params into a copy; never mutate explainer_config.
    params = dict(explainer_config)
    if "background_data" in explainer_config:
        params["background_data"] = _resolve_background_data(
            explainer_config["background_data"], splits
        )

    base = save_path if save_path is not None else Path(config["results_rel_path"])
    explanation_dir = (
        base
        / "explanations"
        / f"explanations_{_explainer_display_name(explainer_config)}"
    )
    visualization_type = explainer_config.get("visualization_type", ["bubbles"])

    # Explain only certain samples addressed by index
    if explainer_config["samples"]["type"] == "by_index":
        explanation = generate_explanation(
            method=explainer_config["method"],
            model=model,
            data=data,
            labels=labels,
            targets=explainer_config["target"],
            params=params,
            encoder=encoder,
            indices=explainer_config["samples"]["indices"],
            prng=prng,
            device=device,
            save_path=explanation_dir,
            visualization_type=visualization_type,
        )
    # Explain random samples
    elif explainer_config["samples"]["type"] == "random":
        explanation = generate_explanation(
            method=explainer_config["method"],
            model=model,
            data=data,
            labels=labels,
            targets=explainer_config["target"],
            params=params,
            encoder=encoder,
            indices=None,
            samples=explainer_config["samples"]["count"],
            prng=prng,
            device=device,
            save_path=explanation_dir,
            visualization_type=visualization_type,
        )
    # Attach per-sample ground-truth metadata for the explained samples (aligned
    # to explanation.indices) so ground-truth metrics (e.g. TimeFrequencyAUC) can
    # read it via the evaluator's metadata injection.
    if metadata is not None and explanation is not None:
        explanation.metadata = [metadata[int(i)] for i in explanation.indices]

    if explainer_config.get("save_exp_values", False) and explanation is not None:
        explanation_dir.mkdir(parents=True, exist_ok=True)
        np.save(explanation_dir / "exp_values.npy", explanation.exp_values)
        np.save(explanation_dir / "indices.npy", explanation.indices)
        try:
            decoded_labels = explanation.encoder.inverse_transform(explanation.labels)
        except Exception:
            decoded_labels = explanation.labels
        with open(explanation_dir / "labels.json", "w") as f:
            json.dump(decoded_labels.tolist(), f)
        logger.info("Explanation values saved to %s", explanation_dir)

    return explanation
