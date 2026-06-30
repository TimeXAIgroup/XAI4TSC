"""
Unit tests for src/xai4tsc/utils/plot.py — headless (Agg), saves to tmp_path.

These render with the non-interactive Agg backend and write PNGs into
``tmp_path`` only, so they stay fast and need no display, model, or real data.
"""

import matplotlib
import numpy as np
import pytest

from xai4tsc.utils.plot import plot_relevance


@pytest.fixture(autouse=True)
def _agg_backend():
    """Force the headless Agg backend and close figures after each test."""
    matplotlib.use("Agg", force=True)
    yield
    from matplotlib import pyplot as plt

    plt.close("all")


def _signal_relevance(shape=(1, 1, 20), seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(shape).astype(np.float32)
    relevance = rng.random(shape).astype(np.float32)
    return signal, relevance


# ── rel_type branches ───────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "rel_type", ["bubbles", "background", "intensity", "graph", "bar", None]
)
def test_plot_relevance_rel_types(tmp_path, rel_type):
    signal, relevance = _signal_relevance()
    out = plot_relevance(
        signal, relevance, rel_type=rel_type, save_path=tmp_path / "r.png"
    )
    assert out is not None
    assert out.exists()


# ── shape handling ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_plot_relevance_dimension_mismatch_returns_none(tmp_path, caplog):
    signal = np.zeros((1, 1, 20), dtype=np.float32)
    relevance = np.zeros((1, 20), dtype=np.float32)  # one fewer axis
    with caplog.at_level("ERROR", logger="xai4tsc.utils.plot"):
        out = plot_relevance(signal, relevance, save_path=tmp_path / "r.png")
    assert out is None
    assert "mismatch" in caplog.text.lower()


@pytest.mark.unit
def test_plot_relevance_2d_autoexpands_with_warning(tmp_path, caplog):
    signal = np.zeros((1, 20), dtype=np.float32)
    relevance = np.zeros((1, 20), dtype=np.float32)
    with caplog.at_level("WARNING", logger="xai4tsc.utils.plot"):
        out = plot_relevance(signal, relevance, save_path=tmp_path / "r.png")
    assert "missing one dimension" in caplog.text.lower()
    assert out is not None
    assert out.exists()


@pytest.mark.unit
def test_plot_relevance_1d_autoexpands_with_warning(tmp_path, caplog):
    signal = np.zeros((20,), dtype=np.float32)
    relevance = np.zeros((20,), dtype=np.float32)
    with caplog.at_level("WARNING", logger="xai4tsc.utils.plot"):
        out = plot_relevance(signal, relevance, save_path=tmp_path / "r.png")
    assert "missing two dimensions" in caplog.text.lower()
    assert out is not None
    assert out.exists()


# ── colour-map / decoration branches ─────────────────────────────────────────────


@pytest.mark.unit
def test_plot_relevance_negative_relevance_uses_bwr(tmp_path):
    signal = np.zeros((1, 1, 20), dtype=np.float32)
    relevance = np.linspace(-1, 1, 20, dtype=np.float32).reshape(1, 1, 20)
    out = plot_relevance(
        signal, relevance, rel_type="bubbles", save_path=tmp_path / "r.png"
    )
    assert out is not None
    assert out.exists()


@pytest.mark.unit
def test_plot_relevance_intensity_negative_relevance(tmp_path):
    # "intensity" has its own diverging colour-map branch for negative values.
    signal = np.sin(np.linspace(0, 6, 20)).astype(np.float32).reshape(1, 1, 20)
    relevance = np.linspace(-1, 1, 20, dtype=np.float32).reshape(1, 1, 20)
    out = plot_relevance(
        signal, relevance, rel_type="intensity", save_path=tmp_path / "r.png"
    )
    assert out is not None
    assert out.exists()


@pytest.mark.unit
def test_plot_relevance_graph_only_and_colorbar(tmp_path):
    signal, relevance = _signal_relevance()
    out = plot_relevance(
        signal,
        relevance,
        rel_type="bubbles",
        graph_only=True,
        colorbar=True,
        title="demo",
        save_path=tmp_path / "r.png",
    )
    assert out is not None
    assert out.exists()


@pytest.mark.unit
def test_plot_relevance_graph_type_skips_colorbar(tmp_path):
    # colorbar is suppressed for rel_type="graph" (the guard branch).
    signal, relevance = _signal_relevance()
    out = plot_relevance(
        signal, relevance, rel_type="graph", colorbar=True, save_path=tmp_path / "r.png"
    )
    assert out is not None
    assert out.exists()


# ── batch handling ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_plot_relevance_batch_gt_1_creates_dir(tmp_path):
    signal, relevance = _signal_relevance(shape=(2, 1, 20))
    out = plot_relevance(
        signal, relevance, rel_type="bubbles", save_path=tmp_path / "r.png"
    )
    assert out is not None
    # A directory named after the save stem is created for multi-sample output.
    out_dir = tmp_path / "r"
    assert out_dir.is_dir()
    assert (out_dir / "r0.png").exists()
    assert (out_dir / "r1.png").exists()
