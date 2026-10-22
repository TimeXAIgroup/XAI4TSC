"""Unit tests for experiment_runner/utils.py - pure helpers, no I/O, no training."""

import numpy as np
import pandas as pd
import pytest
from experiment_runner.config import config_resolve
from experiment_runner.main import MASTER_CONFIG
from experiment_runner.utils import (
    _METRICS_COLUMNS,
    _MODEL_COLUMNS,
    _aggregate_metric,
    _as_integer_columns,
    _check_sample_indices,
    _format_label_list,
    _model_frame,
    _model_row,
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


# ── per-target metric rows ────────────────────────────────────────────────────


@pytest.mark.unit
def test_the_schema_carries_the_target_and_the_sample_count():
    assert "target" in _METRICS_COLUMNS
    assert "samples" in _METRICS_COLUMNS
    # The dict and the column tuple have to agree, since the CSV is written from
    # the dict against this tuple.
    row = _result_row("d", "m", "x", "Complexity", "ok", score=0.5)
    assert set(row) == set(_METRICS_COLUMNS)


@pytest.mark.unit
def test_a_per_target_row_names_its_target_and_its_denominator():
    # One row per target rather than their mean is what makes a per-class spread
    # visible; `samples` is what makes two of those rows comparable, since under
    # target: "predicted" it moves with the model and the threshold.
    row = _result_row(
        "Synth",
        "FCN",
        "freqrise",
        "Time-Frequency AUC",
        "ok",
        score=0.91,
        seed=7,
        target="7Hz",
        samples=18,
    )
    assert row["target"] == "7Hz"
    assert row["samples"] == 18
    assert row["score"] == 0.91


@pytest.mark.unit
def test_a_single_target_row_leaves_the_target_blank():
    # The explained class varies per sample there, so there is no one target to
    # name - and an invented one would read as a real scope.
    row = _result_row("d", "m", "x", "Complexity", "ok", score=0.5, samples=40)
    assert row["target"] == ""
    assert row["samples"] == 40


# ── models.csv: test-set model performance ────────────────────────────────────

# A different measurement from the XAI metric scores, so a separate file: one
# row per (dataset, model), and columns that depend on the formulation.


@pytest.mark.unit
def test_a_model_row_starts_with_the_identity_then_the_summary():
    row = _model_row(
        "GunPoint",
        "FCN",
        "ok",
        summary={"samples": 20, "accuracy": "75.00", "auc": "0.9100"},
        epoch=2,
    )
    frame = _model_frame([row])

    assert list(frame.columns)[: len(_MODEL_COLUMNS)] == list(_MODEL_COLUMNS)
    # The summary's own order is kept rather than sorted, so related numbers
    # stay together instead of interleaving alphabetically.
    assert list(frame.columns)[len(_MODEL_COLUMNS) :] == ["samples", "accuracy", "auc"]
    assert frame.loc[0, "epoch"] == 2


@pytest.mark.unit
def test_the_identity_wins_over_a_colliding_summary_key():
    # Defensive: a summary key named like an identity column must not rename the
    # row's own dataset or model, which is what makes a row attributable.
    row = _model_row("GunPoint", "FCN", "ok", summary={"model": "something else"})
    assert row["model"] == "FCN"


@pytest.mark.unit
def test_the_columns_are_a_union_across_formulations():
    # A binary and a multi-label model in one experiment report different
    # metrics, so the file unions them and leaves a cell empty where the
    # formulation does not define one - rather than dropping either row.
    binary = _model_row("A", "FCN", "ok", summary={"samples": 20, "accuracy": "75.00"})
    multi = _model_row(
        "B", "LeNet", "ok", summary={"samples": 30, "subset_accuracy": "71.67"}
    )
    frame = _model_frame([binary, multi])

    assert {"accuracy", "subset_accuracy"} <= set(frame.columns)
    assert pd.isna(frame.loc[0, "subset_accuracy"])
    assert pd.isna(frame.loc[1, "accuracy"])


@pytest.mark.unit
def test_a_row_detail_is_collapsed_to_one_line():
    # A failed model is skipped but still gets a row, and its reason has to
    # fit one cell - an exception message arrives with newlines in it.
    row = _model_row("A", "FCN", "failed", detail="broke on\n  the  test split")
    assert row["status"] == "failed"
    assert row["detail"] == "broke on the test split", "collapsed to one line"


@pytest.mark.unit
def test_a_header_only_frame_still_has_the_identity_columns():
    frame = _model_frame([])
    assert list(frame.columns) == list(_MODEL_COLUMNS)


# ── counts stay counts ────────────────────────────────────────────────────────
# A row that does not define a column leaves a gap, and pandas holds a gap in an
# int64 column by widening it to float64 - so a sample count reaches the CSV as
# "9.0" unless the column is a nullable integer.


@pytest.mark.unit
def test_a_gap_does_not_turn_the_model_counts_into_floats():
    scored = _model_row("A", "FCN", "ok", summary={"samples": 9, "outputs": 3}, epoch=1)
    failed = _model_row("A", "LeNet", "failed", detail="boom")
    frame = _model_frame([scored, failed])

    assert str(frame["samples"].dtype) == "Int64"
    assert str(frame["outputs"].dtype) == "Int64"
    assert str(frame["epoch"].dtype) == "Int64"
    csv = frame.to_csv(index=False)
    assert "9.0" not in csv and "1.0" not in csv
    # The failed row still leaves those cells empty rather than filling them.
    assert csv.splitlines()[2].endswith(",,")


@pytest.mark.unit
def test_an_unscored_metric_row_keeps_the_others_counts_integral():
    rows = [
        _result_row("d", "m", "x", "Complexity", "ok", score=0.5, samples=40, seed=7),
        _result_row("d", "m", "x", "Pixel-Flipping", "error", detail="boom"),
    ]
    frame = _as_integer_columns(pd.DataFrame(rows, columns=_METRICS_COLUMNS))

    assert str(frame["samples"].dtype) == "Int64"
    assert str(frame["seed"].dtype) == "Int64"
    csv = frame.to_csv(index=False)
    assert ",40,7," in csv
    # The score is a measurement, not a count, and keeps its float dtype.
    assert str(frame["score"].dtype) == "float64"


@pytest.mark.unit
def test_a_fractional_value_under_a_count_name_is_left_as_it_is(caplog):
    # Written once at the end of a run, so one odd column must not cost the file.
    frame = _as_integer_columns(pd.DataFrame([{"samples": 1.5}]))

    assert frame["samples"].iloc[0] == 1.5
    assert "not whole numbers" in caplog.text
