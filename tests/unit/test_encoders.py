"""Unit tests for experiment_runner/encoders.py — the config string to an encoder."""

import numpy as np
import pytest
from experiment_runner.encoders import ENCODERS, build_encoder
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from xai4tsc.data import MultiHotLabelEncoder, PowersetLabelEncoder


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("label", LabelEncoder),
        ("multihot", MultiHotLabelEncoder),
        ("powerset", PowersetLabelEncoder),
    ],
)
def test_build_encoder_returns_the_named_encoder(name, expected):
    assert isinstance(build_encoder(name), expected)


@pytest.mark.unit
def test_build_encoder_defaults_to_label_encoder():
    # An unset `encode` must keep the package default.
    assert isinstance(build_encoder(None), LabelEncoder)


@pytest.mark.unit
@pytest.mark.parametrize("empty", ["", "   "])
def test_an_empty_encode_is_the_default(empty):
    assert isinstance(build_encoder(empty), LabelEncoder)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [{"a": 1}, ["label"], 3, True])
def test_a_non_string_encode_is_a_config_error(bad):
    # A YAML mapping or list used to reach the table lookup and raise `unhashable
    # type`, which escaped the config check (it catches ValueError) and named
    # neither the key nor the fix.
    with pytest.raises(ValueError, match="must be a string naming an encoder"):
        build_encoder(bad)


@pytest.mark.unit
def test_build_encoder_returns_a_fresh_object_each_call():
    # The runner hands the same object to split() and load_saved_splits(), but two
    # datasets must not share a fitted encoder.
    assert build_encoder("label") is not build_encoder("label")


@pytest.mark.unit
def test_build_encoder_resolves_a_dotted_path():
    # The escape hatch: any target encoder, including ones with no table entry.
    assert isinstance(
        build_encoder("sklearn.preprocessing.LabelBinarizer"), LabelBinarizer
    )


@pytest.mark.unit
def test_an_unknown_name_raises_and_names_the_available_keys():
    # It used to fall back to LabelEncoder in silence, so `encode: "multhot"`
    # trained a different problem and reported plausible numbers for it.
    with pytest.raises(ValueError, match="Unknown encode: 'multhot'") as err:
        build_encoder("multhot")
    for key in ENCODERS:
        assert key in str(err.value)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["onehot", "ordinal"])
def test_a_retired_name_raises_and_points_at_label(name):
    # Both named sklearn FEATURE encoders. The message has to say what to use
    # instead, because a config carrying either one stops running.
    with pytest.raises(ValueError, match="no longer available") as err:
        build_encoder(name)
    assert "'label'" in str(err.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["sklearn.preprocessing.NoSuchEncoder", "no.such.module.Encoder"],
)
def test_an_unresolvable_dotted_path_raises(name):
    with pytest.raises(ValueError, match="Unknown encode"):
        build_encoder(name)


@pytest.mark.unit
def test_a_dotted_path_needing_constructor_arguments_raises():
    with pytest.raises(ValueError, match="without arguments"):
        build_encoder("numpy.dtype")


@pytest.mark.unit
def test_the_table_holds_only_target_encoders():
    # sklearn separates target from feature encoders deliberately: a feature
    # encoder takes `X` and exposes `categories_`, and the package rejects one.
    # This is the guard that keeps the table on the right side of that line.
    for name, factory in ENCODERS.items():
        encoder = factory()
        first_param = list(encoder.fit.__code__.co_varnames[:2])[-1]
        assert first_param == "y", f"{name} takes {first_param!r}, not labels"
        assert not hasattr(encoder, "categories_"), name


@pytest.mark.unit
def test_the_shipped_names_round_trip_through_the_package():
    # Each shipped name, exercised end to end on the shape it reads and the
    # target rank it returns: 1-D is class indices, 2-D is one column per output.
    from xai4tsc.data.base import _fit_label_encoder

    scalars = np.array([2, 0, 1, 0])
    _, targets = _fit_label_encoder(scalars, build_encoder("label"))
    assert targets.tolist() == [2, 0, 1, 0]
    assert targets.dtype == np.int64

    rows = np.array([[0, 1], [1, 1], [0, 1], [0, 0]])

    encoder, targets = _fit_label_encoder(rows, build_encoder("powerset"))
    assert targets.tolist() == [1, 2, 1, 0]
    np.testing.assert_array_equal(encoder.inverse_transform(targets), rows)

    encoder, targets = _fit_label_encoder(rows, build_encoder("multihot"))
    np.testing.assert_array_equal(targets, rows)
    assert targets.dtype == np.int64
    assert encoder.classes_.tolist() == [0, 1], "one entry per model output"


@pytest.mark.unit
def test_multihot_takes_column_names_through_encode_params():
    encoder = build_encoder("multihot", {"classes": ["7Hz", "19Hz", "37Hz"]})
    encoder.fit(np.array([[1, 1, 0], [0, 0, 1]]))
    assert encoder.classes_.tolist() == ["7Hz", "19Hz", "37Hz"]


@pytest.mark.unit
def test_column_names_of_the_wrong_length_raise():
    # The columns are positional, so a short list cannot be reordered to fit; it
    # would name an output after a different output.
    encoder = build_encoder("multihot", {"classes": ["7Hz", "19Hz"]})
    with pytest.raises(ValueError, match="class name"):
        encoder.fit(np.array([[1, 1, 0], [0, 0, 1]]))


@pytest.mark.unit
def test_params_without_a_name_are_an_error():
    with pytest.raises(ValueError, match="without an encode"):
        build_encoder(None, {"classes": [0, 1]})


@pytest.mark.unit
def test_multilabelbinarizer_is_reachable_for_set_valued_labels():
    # Label *sets* are MultiLabelBinarizer's native input and need no encoder of
    # ours; the dotted path is the documented way to reach it.
    from sklearn.preprocessing import MultiLabelBinarizer

    encoder = build_encoder("sklearn.preprocessing.MultiLabelBinarizer")
    assert isinstance(encoder, MultiLabelBinarizer)
    assert encoder.fit_transform([("b",), ("a", "c")]).shape == (2, 3)
