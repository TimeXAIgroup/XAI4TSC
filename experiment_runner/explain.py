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

from xai4tsc.data.base import decode_labels
from xai4tsc.utils.utils import dict_to_args
from xai4tsc.xai.explain import generate_explanation

from .config import strip_runner_keys
from .lookups import EXPLAINERS, build_transform

if TYPE_CHECKING:
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

    from xai4tsc.models.base import ModelBase
    from xai4tsc.xai._types import Domain, Explanation
    from xai4tsc.xai.base import ExplainerBase

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


def _inject_seed(params: dict, seed: int | None) -> dict:
    """
    Fill in a derived seed wherever the config did not state one.

    Applied to the explainer entry and, recursively, to the ``base`` of a wrapper,
    because :func:`_build_base` builds the wrapped method from that sub-dict alone -
    a seed left at the top level would never reach a stochastic base. A bare string
    base is promoted to a dict so it can carry one.

    The test is key *presence*, not truthiness: ``seed: null`` is the documented way
    to opt a component out of seeding, so it has to survive. An explicitly
    configured seed always wins over the derived one.

    Parameters
    ----------
    params : dict
        Explainer params to fill in; mutated and returned.
    seed : int or None
        The derived seed, or ``None`` under ``reproducible: false``.

    Returns
    -------
    dict
        *params*, with a ``seed`` at every level that lacked one.
    """
    if "seed" not in params:
        params["seed"] = seed
    base = params.get("base")
    if isinstance(base, str):
        base = {"method": base}
    if isinstance(base, dict):
        params["base"] = _inject_seed(dict(base), seed)
    return params


def build_explainer(method: str, params: dict | None) -> ExplainerBase:
    """
    Instantiate a registered explainer, filtering *params* to its ``__init__``.

    Parameters
    ----------
    method : str
        Explainer name; looked up case-insensitively in
        :data:`~experiment_runner.lookups.EXPLAINERS`.
    params : dict or None
        Parameters to pass to the explainer's ``__init__``. Keys not matching
        the constructor signature are dropped via
        :func:`~xai4tsc.utils.dict_to_args`. If the class defines no
        ``__init__``, no parameters are passed.

    Returns
    -------
    ExplainerBase
        A ready-to-use explainer instance.

    Raises
    ------
    NotImplementedError
        If *method* is not in :data:`~experiment_runner.lookups.EXPLAINERS`.
    """
    explainer_key = method.lower()
    if explainer_key not in EXPLAINERS:
        raise NotImplementedError(f"Explainer '{explainer_key}' not supported.")
    # Only pass config params if the class defines its own __init__ - otherwise
    # object.__init__ is used, which would receive the full config dict and fail.
    explainer_class = EXPLAINERS[explainer_key]
    if "__init__" in explainer_class.__dict__:
        init_params = dict_to_args(params, explainer_class.__init__)
        return explainer_class(**(init_params or {}))
    return explainer_class()


def _build_base(entry: dict) -> ExplainerBase:
    """
    Build a wrapper's ``base`` explainer from its config sub-dict.

    A base that is itself a wrapper has its own ``base`` built first.

    Parameters
    ----------
    entry : dict
        ``{"method": <name>, **params}`` for the wrapped explainer.

    Returns
    -------
    ExplainerBase
        The explainer instance to pass to the wrapper.
    """
    params = dict(entry)
    method = params.pop("method")
    if isinstance(params.get("base"), dict):
        params["base"] = _build_base(params["base"])
    return build_explainer(method, params)


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
    explanation_dir: Path | None = None,
    data_domain: str | Domain = "time",
    seed: int | None = None,
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
    data_domain : str or Domain, optional
        Signal domain the data is expressed in, taken from the loaded dataset
        (``"time"`` for every dataset shipped here).
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
    explanation_dir : Path, optional
        Directory to write this explainer's artifacts into. The caller supplies it
        so that repeated entries for one method can be kept apart (see
        :func:`~experiment_runner.config.result_identities`); when ``None`` it falls
        back to ``<save_path>/explanations/explanations_<display name>``, which
        keeps direct callers working but cannot disambiguate duplicates.
    seed : int or None
        Seed derived for this explainer, filled into the explainer's params (and
        its wrapped base's) when the config did not state one. The explainer
        built with it is stored on ``Explanation.explainer``, which the
        re-explaining metrics copy.

    Returns
    -------
    Explanation
        An object of the Explanation dataclass.

    Notes
    -----
    An entry's top-level keys are the explainer's construction parameters; a
    nested ``call_params`` block holds per-call options, which are passed to
    ``explain()`` and recorded on ``Explanation.explainer_call_params`` so the
    re-explaining metrics replay the same call. The keys the runner reads itself
    are removed first (:func:`~experiment_runner.config.strip_runner_keys`), so
    they cannot arrive as construction parameters.
    """
    # Resolve runtime-only params into a copy; never mutate explainer_config.
    params = _inject_seed(dict(explainer_config), seed)
    # Popped, not read: `dict_to_args` passes the whole dict to a constructor
    # declaring **kwargs, so a `call_params` left in would arrive as a
    # construction argument.
    call_params = params.pop("call_params", None)
    # A non-mapping is reported by the config check; coerce rather than hand it to
    # `dict()`, which would raise an error naming neither the key nor the entry.
    call_params = call_params if isinstance(call_params, dict) else {}
    # The entry doubles as the construction params, so the keys the runner reads
    # itself have to come off before the explainer sees them: `dict_to_args` drops
    # them for a named signature but hands the lot to a `**kwargs` constructor.
    params = strip_runner_keys(params, explainer_config["method"], config)
    if "transform" in params:
        params["transform"] = build_transform(params["transform"])
    if isinstance(params.get("base"), dict):
        params["base"] = _build_base(params["base"])
    if "background_data" in explainer_config:
        params["background_data"] = _resolve_background_data(
            explainer_config["background_data"], splits
        )
    explainer = build_explainer(explainer_config["method"], params)

    if explanation_dir is None:
        base = save_path if save_path is not None else Path(config["results_rel_path"])
        explanation_dir = (
            base
            / "explanations"
            / f"explanations_{_explainer_display_name(explainer_config)}"
        )
    visualization_type = explainer_config["visualization_type"]

    # Explain only certain samples addressed by index
    if explainer_config["samples"]["type"] == "index":
        explanation = generate_explanation(
            explainer=explainer,
            model=model,
            data=data,
            labels=labels,
            targets=explainer_config["target"],
            call_params=call_params,
            encoder=encoder,
            indices=explainer_config["samples"]["indices"],
            prng=prng,
            device=device,
            save_path=explanation_dir,
            visualization_type=visualization_type,
            data_domain=data_domain,
            threshold=explainer_config["threshold"],
        )
    # Explain random samples
    elif explainer_config["samples"]["type"] == "random":
        explanation = generate_explanation(
            explainer=explainer,
            model=model,
            data=data,
            labels=labels,
            targets=explainer_config["target"],
            call_params=call_params,
            encoder=encoder,
            indices=None,
            n_samples=explainer_config["samples"]["count"],
            prng=prng,
            device=device,
            save_path=explanation_dir,
            visualization_type=visualization_type,
            data_domain=data_domain,
            threshold=explainer_config["threshold"],
        )
    else:
        # "all" case
        explanation = generate_explanation(
            explainer=explainer,
            model=model,
            data=data,
            labels=labels,
            targets=explainer_config["target"],
            call_params=call_params,
            encoder=encoder,
            indices=list(np.arange(len(data)).tolist()),
            prng=prng,
            device=device,
            save_path=explanation_dir,
            visualization_type=visualization_type,
            data_domain=data_domain,
            threshold=explainer_config["threshold"],
        )
    # Attach per-sample ground-truth metadata for the explained samples (aligned
    # to explanation.indices) so ground-truth metrics (e.g. TimeFrequencyAUC) can
    # read it via the evaluator's metadata injection.
    if metadata is not None and explanation is not None:
        explanation.metadata = [metadata[int(i)] for i in explanation.indices]

    if explainer_config["save_exp_values"] and explanation is not None:
        explanation_dir.mkdir(parents=True, exist_ok=True)
        np.save(explanation_dir / "exp_values.npy", explanation.exp_values)
        np.save(explanation_dir / "indices.npy", explanation.indices)
        if explanation.target_mask is not None:
            # Without it the stack is uninterpretable: it holds an attribution
            # for every (target, sample) pair, and only the mask says which of
            # them the sample actually carries.
            np.save(
                explanation_dir / "target_mask.npy",
                np.asarray(explanation.target_mask),
            )
        try:
            decoded_labels = decode_labels(explanation.encoder, explanation.labels)
        except Exception:
            decoded_labels = explanation.labels
        with open(explanation_dir / "labels.json", "w") as f:
            json.dump(decoded_labels.tolist(), f)
        logger.info("Explanation values saved to %s", explanation_dir)

    return explanation
