"""Low-level utilities: ``dict_to_args``, ``merge_dicts``, and ``rescale_array``."""

import copy
import hashlib
import importlib.machinery
import importlib.util
import inspect
import logging
import re
from pathlib import Path
from types import FunctionType

import numpy as np

logger = logging.getLogger(__name__)


def dict_to_args(dict_to_filter: dict, func: FunctionType) -> dict:
    """
    Filter a dictionary to only contain arguments specified in func.

    Supported arguments are POSITIONAL_OR_KEYWORD, KEYWORD_ONLY,
    and VAR_KEYWORDS (in which case all arguments remain in the dict).
    Positional arguments without keyword (such as ``*args``) are not supported.

    Parameters
    ----------
    dict_to_filter : dict
        Dictionary to filter.
    func : FunctionType
        Function whose arguments will be kept.

    Returns
    -------
    dict
        Filtered dictionary containing only arguments of func.
    """
    if dict_to_filter is None:
        return None

    sig = inspect.signature(func)

    has_kwargs = any(
        param.kind == param.VAR_KEYWORD for param in sig.parameters.values()
    )
    if has_kwargs:
        return dict_to_filter

    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD or param.kind == param.KEYWORD_ONLY
    ]
    filtered_dict = {
        filter_key: dict_to_filter[filter_key]
        for filter_key in filter_keys
        if filter_key in dict_to_filter
    }
    return filtered_dict


def merge_dicts(prio: dict, base: dict) -> dict:
    """
    Recursively merge two dictionaries where the values of prio take precedence.

    Neither input is modified, and the result shares no objects with them. Two
    dicts under one key are merged recursively; any other value in *prio*,
    including ``None``, replaces the one in *base*. Keys keep *prio*'s order,
    followed by the keys only *base* has.

    Parameters
    ----------
    prio : dict
        Dictionary whose values have higher priority.
    base : dict
        Dictionary whose values are used when a key is absent from prio.

    Returns
    -------
    dict
        A new merged dictionary containing all keys.
    """
    merged = copy.deepcopy(prio)
    for key, value in base.items():
        if key not in merged:
            merged[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = merge_dicts(merged[key], value)
    return merged


def load_class_from_path(path: str | Path, class_name: str) -> type:
    """
    Import a class by name from an external Python source file.

    Executes the module at *path* (running arbitrary Python) and returns the
    attribute named *class_name*. The module is registered under a name suffixed
    with a hash of its path, so two different external files never collide in
    ``sys.modules``.

    This runs untrusted code. Callers are responsible for obtaining user consent
    before invoking it, and so for gating it behind a setting the user controls
    rather than calling it on a path that merely happens to be present.

    Parameters
    ----------
    path : str or Path
        Path to the ``.py`` file containing the class definition.
    class_name : str
        Name of the class to retrieve from the loaded module.

    Returns
    -------
    type
        The requested class object.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    ImportError
        If the module cannot be imported or does not define *class_name*.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"External source file not found: {path}")

    logger.warning(
        "Loading external source from %s — this executes arbitrary Python code.",
        path,
    )

    # Unique module name so two external files never clash in sys.modules.
    name_suffix = hashlib.sha1(str(path).encode()).hexdigest()[:10]
    src_name = f"{path.stem}_{name_suffix}"
    loader = importlib.machinery.SourceFileLoader(src_name, str(path))
    spec = importlib.util.spec_from_loader(src_name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Class '{class_name}' not found in external source {path}"
        ) from exc


def rescale_array(
    array: np.ndarray, new_min: float = 0.0, new_max: float = 1.0
) -> np.ndarray:
    """Min-max rescale array values to [new_min, new_max]."""
    a_min, a_max = array.min(), array.max()
    if a_min == a_max:
        return np.full_like(array, new_min, dtype=float)
    return (array - a_min) / (a_max - a_min) * (new_max - new_min) + new_min


def _filename_slug(name: object, fallback: str) -> str:
    """
    Reduce a class name to something safe to put in a filename.

    A generated class name - an integer index, or a multi-hot row joined with
    dashes - is usable as-is, but a name supplied through an encoder's
    ``classes`` argument is arbitrary text: a ``/`` would silently write into a
    subdirectory or fail, and a space makes the path awkward to pass around.

    Parameters
    ----------
    name : object
        The display name, of any type; it is stringified first.
    fallback : str
        Used when *name* has no usable characters at all.

    Returns
    -------
    str
        *name* with every character outside ``[A-Za-z0-9._-]`` replaced by an
        underscore, runs collapsed and the ends trimmed.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return slug or fallback
