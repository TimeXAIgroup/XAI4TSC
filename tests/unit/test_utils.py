"""Unit tests for src/xai4tsc/utils/utils.py."""

import numpy as np
import pytest

from xai4tsc.utils import dict_to_args, merge_dicts, rescale_array

# ── dict_to_args ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_dict_to_args_filters_unknown_keys():
    def fn(a, b):
        pass

    result = dict_to_args({"a": 1, "b": 2, "c": 3}, fn)
    assert result == {"a": 1, "b": 2}
    assert "c" not in result


@pytest.mark.unit
def test_dict_to_args_keeps_all_when_var_kwargs():
    def fn(**kwargs):
        pass

    d = {"x": 1, "y": 2, "z": 3}
    assert dict_to_args(d, fn) == d


@pytest.mark.unit
def test_dict_to_args_returns_none_when_none():
    def fn(a):
        pass

    assert dict_to_args(None, fn) is None


@pytest.mark.unit
def test_dict_to_args_keyword_only():
    def fn(a, *, b, c):
        pass

    result = dict_to_args({"a": 1, "b": 2, "c": 3, "d": 4}, fn)
    assert "a" in result
    assert "b" in result
    assert "c" in result
    assert "d" not in result


# ── merge_dicts ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_merge_dicts_priority_wins():
    result = merge_dicts({"a": 1}, {"a": 99, "b": 2})
    assert result["a"] == 1


@pytest.mark.unit
def test_merge_dicts_base_keys_added():
    result = merge_dicts({"a": 1}, {"b": 2})
    assert result["b"] == 2


@pytest.mark.unit
def test_merge_dicts_nested_recursive():
    prio = {"hp": {"lr": 0.01}}
    base = {"hp": {"lr": 0.001, "epochs": 10}}
    result = merge_dicts(prio, base)
    assert result["hp"]["lr"] == 0.01  # prio wins
    assert result["hp"]["epochs"] == 10  # base fills in


@pytest.mark.unit
def test_merge_dicts_base_not_mutated():
    base = {"a": 1, "b": 2}
    base_copy = dict(base)
    merge_dicts({"c": 3}, base)
    assert base == base_copy


# ── rescale_array ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_rescale_array_min_max():
    arr = np.array([0.0, 5.0, 10.0])
    out = rescale_array(arr, new_min=0.0, new_max=1.0)
    assert pytest.approx(out.min()) == 0.0
    assert pytest.approx(out.max()) == 1.0


@pytest.mark.unit
def test_rescale_array_custom_range():
    arr = np.array([1.0, 2.0, 3.0])
    out = rescale_array(arr, new_min=-1.0, new_max=1.0)
    assert pytest.approx(out.min()) == -1.0
    assert pytest.approx(out.max()) == 1.0


@pytest.mark.unit
def test_rescale_array_flat_returns_new_min():
    arr = np.full(10, 5.0)
    out = rescale_array(arr, new_min=0.0, new_max=1.0)
    assert np.all(out == 0.0)


@pytest.mark.unit
def test_rescale_array_output_dtype_float():
    arr = np.array([1, 2, 3], dtype=np.int32)
    out = rescale_array(arr)
    assert np.issubdtype(out.dtype, np.floating)
