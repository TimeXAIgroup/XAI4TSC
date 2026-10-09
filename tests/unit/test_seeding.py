"""Unit tests for seed derivation - the package helpers and the runner boundaries."""

import numpy as np
import pytest
from experiment_runner.config import result_identities
from experiment_runner.seeding import boundary_seed, seed_globals

from xai4tsc import DEFAULT_SEED, derive_seed, resolve_rng

# ── derive_seed ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_derive_seed_is_stable_across_processes():
    """Hard-coded values: a hash() regression would be salted and differ per run."""
    assert derive_seed(2025, "dataset", "GunPoint", 1) == 3683578805
    assert derive_seed(DEFAULT_SEED, "metric", "Complexity", 1) == 623464173


@pytest.mark.unit
def test_derive_seed_is_in_range_for_every_consumer():
    """np.random.seed is the tightest sink: it rejects anything >= 2**32."""
    seeds = [
        derive_seed(DEFAULT_SEED, kind, name, occ)
        for kind in ("dataset", "model", "explainer", "metric")
        for name in ("GunPoint", "FCN", "SIGN-Integrated_Gradients", "Pixel-Flipping")
        for occ in (1, 2, 17)
    ]
    assert all(0 <= s < 2**32 for s in seeds)
    for s in seeds:
        np.random.seed(s)  # would raise if out of range


@pytest.mark.unit
def test_derive_seed_is_injective_for_separator_characters():
    """A joined key would collide here; the length prefix is what prevents it."""
    assert derive_seed(2025, "metric", "Pixel|Flipping", 1) != derive_seed(
        2025, "metric|Pixel", "Flipping", 1
    )


@pytest.mark.unit
def test_derive_seed_separates_kinds_and_parents():
    """A name reused at another level, or under another parent, must not collapse."""
    assert derive_seed(1, "dataset", "FCN", 1) != derive_seed(1, "model", "FCN", 1)
    assert derive_seed(1, "model", "FCN", 1) != derive_seed(2, "model", "FCN", 1)


# ── result_identities: the occurrence half ────────────────────────────────────


@pytest.mark.unit
def test_occurrence_index_is_stable_when_a_duplicate_is_added():
    """The reason seeds key on occurrence rather than on the display identity."""
    assert result_identities(["A"])[1][0] == result_identities(["A", "A"])[1][0] == 1


@pytest.mark.unit
def test_occurrences_count_each_name_independently():
    assert result_identities(["TSHAP", "SIGN", "TSHAP"])[1] == [1, 1, 2]
    assert result_identities([])[1] == []


@pytest.mark.unit
def test_duplicate_entries_get_different_seeds():
    """Duplicating an entry to observe variance has to still show variance."""
    names = ["TSHAP", "TSHAP"]
    _, occ = result_identities(names)
    first = boundary_seed(2025, "explainer", names[0], occ[0], log=False)
    second = boundary_seed(2025, "explainer", names[1], occ[1], log=False)
    assert first != second


# ── boundary_seed ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_boundary_seed_is_independent_of_sibling_entries():
    """
    The unit-level regression test for R31.

    One result's seed must not move when unrelated entries are added beside it,
    which is what made a single result irreproducible in isolation.
    """

    def chain(datasets, models, explainers, metrics):
        d_i, m_i = datasets.index("GunPoint"), models.index("FCN")
        e_i, k_i = explainers.index("TSHAP"), metrics.index("Complexity")
        seed = boundary_seed(
            2025, "dataset", "GunPoint", result_identities(datasets)[1][d_i], log=False
        )
        seed = boundary_seed(
            seed, "model", "FCN", result_identities(models)[1][m_i], log=False
        )
        seed = boundary_seed(
            seed, "explainer", "TSHAP", result_identities(explainers)[1][e_i], log=False
        )
        return boundary_seed(
            seed, "metric", "Complexity", result_identities(metrics)[1][k_i], log=False
        )

    alone = chain(["GunPoint"], ["FCN"], ["TSHAP"], ["Complexity"])
    crowded = chain(
        ["ECG200", "GunPoint", "Coffee"],
        ["LeNet", "FCN"],
        ["Integrated_Gradients", "TSHAP", "DeepLift"],
        ["Sparseness", "Complexity", "Pixel-Flipping"],
    )
    assert alone == crowded


@pytest.mark.unit
def test_boundary_seed_propagates_none():
    """`reproducible: false` must stay unseeded the whole way down."""
    assert boundary_seed(None, "dataset", "GunPoint", 1, log=False) is None


@pytest.mark.unit
def test_boundary_seed_logs_the_derived_value(caplog):
    """Every boundary records its seed, so a result can be traced from the log."""
    with caplog.at_level("INFO", logger="xai4tsc.runner.seeding"):
        seed = boundary_seed(2025, "model", "FCN", 1)
    assert str(seed) in caplog.text


# ── resolve_rng / seed_globals ────────────────────────────────────────────────


@pytest.mark.unit
def test_resolve_rng_prefers_a_supplied_generator():
    shared = np.random.default_rng(7)
    assert resolve_rng(DEFAULT_SEED, shared) is shared


@pytest.mark.unit
def test_resolve_rng_is_reproducible_for_a_seed():
    assert resolve_rng(11).integers(2**31) == resolve_rng(11).integers(2**31)


@pytest.mark.unit
def test_seed_globals_none_leaves_the_streams_advancing():
    """A no-op on None, so an unseeded run does not become reproducible."""
    import torch

    torch.manual_seed(3)
    np.random.seed(3)
    first = (np.random.rand(), torch.rand(1).item())
    seed_globals(None)
    second = (np.random.rand(), torch.rand(1).item())
    assert first != second
