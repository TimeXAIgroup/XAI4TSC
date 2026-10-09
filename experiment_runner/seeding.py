"""
Per-result seed derivation for the standalone runner.

The runner seeds at four boundaries - dataset, model, explainer, metric - rather
than once per run, so a result's randomness depends on where it sits in the config
and not on what happened to run before it. Re-running one dataset to check a number
then gives that number back.

Three channels carry randomness through a run and all three are reset at every
boundary, because only re-deriving all of them makes a boundary independent of its
predecessor:

* the process-global ``numpy`` stream, which the Quantus metrics draw from;
* the process-global ``torch`` stream, which model weight init and dropout draw from;
* generator objects, which the split and the explained-sample selection are handed
  explicitly. These are objects, not globals, so reseeding the globals leaves them
  exactly where they were - they have to be rebuilt from a derived seed.

Resetting process globals is an application-level decision, so it lives here rather
than in the package: a library that reseeds globals changes results its caller never
asked it to change. The package side is :mod:`xai4tsc._random`.
"""

import logging

import numpy as np
import torch

from xai4tsc.utils import derive_seed

logger = logging.getLogger("xai4tsc.runner.seeding")


def seed_globals(seed: int | None) -> None:
    """
    Reset the process-global ``numpy`` and ``torch`` streams.

    Parameters
    ----------
    seed : int or None
        Seed to apply. ``None`` is a no-op, which leaves both streams advancing
        from wherever they were - what ``reproducible: false`` wants, since an
        unseeded run should not be reproducible.
    """
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)


def boundary_seed(
    parent: int | None,
    kind: str,
    name: str,
    occurrence: int = 1,
    *,
    log: bool = True,
) -> int | None:
    """
    Derive one boundary's seed from its parent's, and record it.

    Parameters
    ----------
    parent : int or None
        Seed of the enclosing boundary. ``None`` propagates: an unseeded run stays
        unseeded all the way down rather than becoming reproducible halfway.
    kind : str
        Which boundary this is (``"dataset"``, ``"model"``, ``"explainer"``,
        ``"metric"``), so a name reused across levels cannot collide.
    name : str
        The entry's configured name.
    occurrence : int
        1-based count of how many entries of this name precede it, from
        :func:`~experiment_runner.config.result_identities`. Used instead of the
        display identity because the latter changes for existing entries when a
        duplicate is added.
    log : bool
        Whether to log the derived seed. The per-model log file is attached before
        the model boundary is derived, so that seed lands in it.

    Returns
    -------
    int or None
        The derived seed, or ``None`` when *parent* was ``None``.
    """
    if parent is None:
        return None
    seed = derive_seed(parent, kind, name, occurrence)
    if log:
        logger.info("Seed for %s '%s' (#%d): %d", kind, name, occurrence, seed)
    return seed
