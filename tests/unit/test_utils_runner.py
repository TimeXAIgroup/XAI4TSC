"""Unit tests for experiment_runner/utils.py - pure helpers, no I/O, no training."""

import numpy as np
import pytest
from experiment_runner.config import config_resolve
from experiment_runner.main import MASTER_CONFIG
from experiment_runner.utils import (
    _METRICS_COLUMNS,
    _aggregate_metric,
    _check_sample_indices,
    _format_label_list,
    _result_row,
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


# ── the seed column makes a result row self-describing ────────────────────────


@pytest.mark.unit
def test_metrics_columns_include_the_seed():
    """A row has to carry the seed that produced it, or it cannot be re-run."""
    assert "seed" in _METRICS_COLUMNS


@pytest.mark.unit
def test_result_row_carries_the_seed():
    row = _result_row("ds", "m", "e", "k", "ok", score=1.0, seed=12345)
    assert row["seed"] == 12345
    assert set(row) == set(_METRICS_COLUMNS)


@pytest.mark.unit
def test_result_row_seed_is_empty_when_unseeded():
    """`reproducible: false` applies no seed, so the cell stays empty."""
    assert _result_row("ds", "m", "e", "k", "ok", score=1.0)["seed"] is None


# ── sample index bounds ───────────────────────────────────────────────────────


def _explanation_config(samples, method="Integrated_Gradients"):
    return {
        "default_settings": {"samples": {"type": "index", "indices": [0]}},
        "explainers": [{"method": method, "samples": samples}],
    }


@pytest.mark.unit
def test_out_of_range_index_raises_before_training():
    """
    The bound is checked once the split exists and before any model trains.

    Config validation cannot do it: it runs before a dataset is loaded, so the
    test-split size does not exist yet.
    """
    cfg = _explanation_config({"type": "index", "indices": [0, 1, 999999]})

    with pytest.raises(ValueError) as excinfo:
        _check_sample_indices(cfg, n_test=55, dataset_id="Chinatown")
    message = str(excinfo.value)
    assert "999999" in message
    assert "55 samples" in message
    assert "0-54" in message
    assert "Chinatown" in message


@pytest.mark.unit
def test_last_valid_index_is_accepted():
    cfg = _explanation_config({"type": "index", "indices": [54]})

    assert _check_sample_indices(cfg, n_test=55, dataset_id="Chinatown") is None


@pytest.mark.unit
@pytest.mark.parametrize("samples", [{"type": "random", "count": 5}, {"type": "all"}])
def test_other_sample_types_are_not_bound_checked(samples):
    # Only explicit indices can be out of range; the other modes derive theirs
    # from the data itself.
    cfg = _explanation_config(samples)

    assert _check_sample_indices(cfg, n_test=3, dataset_id="Coffee") is None


@pytest.mark.unit
def test_indices_inherited_from_default_settings_are_checked(tmp_path):
    # An entry that states no samples block of its own still gets checked,
    # because config_resolve has already overlaid default_settings onto it.
    # Driving this through the real resolve is the point: it is what makes the
    # bound check able to read the effective value straight off the entry.
    user = {
        "experiment_name": "resolve_then_check",
        "results_rel_path": str(tmp_path),
        "data_config": {"datasets": []},
        "train_config": {"models": []},
        "evaluation_config": {"metrics": []},
        "explanation_config": {
            "default_settings": {"samples": {"type": "index", "indices": [0, 4242]}},
            "explainers": [{"method": "Integrated_Gradients"}],
        },
    }
    config = config_resolve(user, MASTER_CONFIG)

    with pytest.raises(ValueError, match="4242"):
        _check_sample_indices(
            config["explanation_config"], n_test=10, dataset_id="GunPoint"
        )
