"""Unit tests for experiment_runner/utils.py - pure helpers, no I/O, no training."""

import numpy as np
import pytest
from experiment_runner.utils import (
    _aggregate_metric,
    _format_label_list,
    expand_datasets,
)


@pytest.mark.unit
def test_aggregate_metric_none_returns_none():
    assert _aggregate_metric(None) is None


@pytest.mark.unit
def test_aggregate_metric_empty_array_returns_none():
    assert _aggregate_metric(np.array([])) is None


@pytest.mark.unit
def test_aggregate_metric_all_nan_returns_none():
    assert _aggregate_metric(np.array([np.nan, np.nan])) is None


@pytest.mark.unit
def test_aggregate_metric_vector_returns_nanmean():
    # NaNs are ignored; mean of the finite values (1.0, 3.0) is 2.0.
    assert _aggregate_metric([1.0, 3.0, np.nan]) == pytest.approx(2.0)


@pytest.mark.unit
def test_aggregate_metric_scalar_passthrough():
    assert _aggregate_metric(0.5) == pytest.approx(0.5)


@pytest.mark.unit
def test_aggregate_metric_zero_dim_array_passthrough():
    assert _aggregate_metric(np.array(0.75)) == pytest.approx(0.75)


# ── expand_datasets ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_expand_datasets_explicit_entry_passthrough():
    out = expand_datasets([{"dataset": "GunPoint", "max_samples": 500}])
    assert out == [{"dataset": "GunPoint", "max_samples": 500}]


@pytest.mark.unit
def test_expand_datasets_wildcard_inline_applies_to_all_members():
    out = expand_datasets([{"dataset": "UEA", "allow_padding": True}])
    assert len(out) > 1
    assert all(d["allow_padding"] is True for d in out)
    assert all("overrides" not in d for d in out)
    # The wildcard keyword itself is never emitted as a dataset.
    assert all(d["dataset"] != "UEA" for d in out)


@pytest.mark.unit
def test_expand_datasets_override_wins_over_inline():
    out = expand_datasets(
        [
            {
                "dataset": "UEA",
                "allow_padding": True,
                "skip_explainers": False,
                "overrides": {"InsectWingbeat": {"skip_explainers": True}},
            }
        ]
    )
    ins = next(d for d in out if d["dataset"] == "InsectWingbeat")
    assert ins["skip_explainers"] is True  # override wins
    assert ins["allow_padding"] is True  # inherited inline setting
    others = [d for d in out if d["dataset"] != "InsectWingbeat"]
    assert all(d["skip_explainers"] is False for d in others)


@pytest.mark.unit
def test_expand_datasets_unknown_override_member_warns(caplog):
    with caplog.at_level("WARNING", logger="xai4tsc.runner"):
        out = expand_datasets(
            [
                {
                    "dataset": "UEA",
                    "overrides": {"NotADataset": {"skip_explainers": True}},
                }
            ]
        )
    assert all(d["dataset"] != "NotADataset" for d in out)
    assert "NotADataset" in caplog.text


@pytest.mark.unit
def test_expand_datasets_overrides_on_non_archive_warns_and_drops(caplog):
    with caplog.at_level("WARNING", logger="xai4tsc.runner"):
        out = expand_datasets(
            [{"dataset": "GunPoint", "overrides": {"GunPoint": {"max_samples": 5}}}]
        )
    assert out == [{"dataset": "GunPoint"}]
    assert "overrides" in caplog.text.lower()


# ── _format_label_list ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_format_label_list_empty():
    assert _format_label_list([]) == ""


@pytest.mark.unit
def test_format_label_list_short_lists_are_named_in_full():
    assert _format_label_list(["A (#1)", "B (#2)"]) == "A (#1), B (#2)"


@pytest.mark.unit
def test_format_label_list_at_limit_is_not_truncated():
    labels = [f"D{i}" for i in range(10)]
    out = _format_label_list(labels, limit=10)
    assert out == ", ".join(labels)
    assert "more" not in out


@pytest.mark.unit
def test_format_label_list_truncates_beyond_limit():
    # An archive wildcard expands to 128 entries; the line must stay readable.
    out = _format_label_list([f"D{i}" for i in range(128)], limit=10)
    assert out.startswith("D0, D1,")
    assert out.endswith(", ... and 118 more")
    assert "D10" not in out.split("...")[0]


@pytest.mark.unit
def test_format_label_list_truncation_counts_remainder_exactly():
    assert _format_label_list(["a", "b", "c"], limit=2) == "a, b, ... and 1 more"
