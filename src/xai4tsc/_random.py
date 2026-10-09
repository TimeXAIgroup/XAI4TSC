"""
The package's default seed and the helpers that derive child seeds from it.

Randomness reaches the package through two channels this module owns: a generator
object passed in explicitly, and a seed passed to a component's ``__init__``.
Centralising them means every layer agrees on what an unspecified seed means, and
that a component's randomness can always be pinned from the outside.

The module depends on ``hashlib`` and ``numpy`` only, so the data, explainer and
evaluation layers can all import it without pulling in the plotting stack that
:mod:`xai4tsc.utils` imports.
"""

import hashlib

import numpy as np

DEFAULT_SEED: int = 0xDEADBEEF
"""
Seed used wherever a caller supplies none.

A fixed default rather than ``None``: an unseeded ``np.random.default_rng(None)``
draws from OS entropy and cannot be pinned from the outside at all, not even by
``np.random.seed``, so the documented package path would be irreproducible with no
way to fix it. Pass ``None`` explicitly to opt into entropy.

The value fits in 32 bits, which is the tightest of the consumers it has to satisfy
(``np.random.seed`` rejects anything larger; ``default_rng``, ``torch.manual_seed``
and scikit-learn's ``random_state`` all accept more).
"""


def derive_seed(parent: int, *parts: object) -> int:
    """
    Derive a child seed from *parent* and an identity, independently of call order.

    The identity is hashed rather than drawn from a generator, so the result depends
    only on what it names: deriving the same identity twice gives the same seed, and
    nothing derived earlier can shift it. That is what makes a single result
    reproducible on its own rather than only as part of the run that produced it.

    Each part is length-prefixed before being hashed, so the serialisation is
    injective - no byte inside a part can be mistaken for a boundary between parts,
    and distinct identities therefore cannot collide.

    ``hashlib`` is used rather than the builtin :func:`hash`, whose string hashing is
    salted per process by ``PYTHONHASHSEED`` and would hand out different seeds on
    every run.

    Parameters
    ----------
    parent : int
        Seed of the enclosing scope. Chaining through it means a child's seed changes
        when its parent's does, so one run seed still governs the whole run.
    *parts : object
        Identity of the child within that scope, stringified in the order given.
        Order is part of the identity.

    Returns
    -------
    int
        A seed in ``[0, 2**32]``, accepted by every consumer in the package and by
        ``np.random.seed``.
    """
    digest = hashlib.blake2b(digest_size=4)
    for part in (parent, *parts):
        encoded = str(part).encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "big")


def resolve_rng(
    seed: int | None,
    rng: np.random.Generator | None = None,
) -> np.random.Generator:
    """
    Return the generator to draw from, preferring a caller-supplied one.

    Unifies the two ways a caller can control randomness: hand over a generator to
    share a single stream, or give a seed and let one be built. A supplied *rng*
    wins.

    Parameters
    ----------
    seed : int or None
        Seed for a fresh generator, used only when *rng* is ``None``. ``None`` means
        draw from OS entropy, which is not reproducible.
    rng : np.random.Generator, optional
        An existing generator to use instead. Takes precedence over *seed*.

    Returns
    -------
    np.random.Generator
        *rng* if it was given, else a fresh generator seeded with *seed*.
    """
    if rng is not None:
        return rng
    return np.random.default_rng(seed)
