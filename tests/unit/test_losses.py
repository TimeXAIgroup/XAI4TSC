"""Unit tests for experiment_runner/lookups.py - the config string to a torch loss."""

import pytest
import torch
from experiment_runner.lookups import LOSSES, as_tensors, build_loss
from torch import nn


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("crossentropy", nn.CrossEntropyLoss),
        ("bcewithlogits", nn.BCEWithLogitsLoss),
        ("nll", nn.NLLLoss),
    ],
)
def test_build_loss_returns_the_named_loss(name, expected):
    assert isinstance(build_loss(name), expected)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["CrossEntropy", "CROSSENTROPY", "crossEntropy"])
def test_the_table_is_matched_case_insensitively(name):
    # Four shipped configs and seven sandbox ones spell it "CrossEntropy". The
    # package used to lower-case it; now the runner does, and this is the test
    # standing between that change and every one of those configs.
    assert isinstance(build_loss(name), nn.CrossEntropyLoss)


@pytest.mark.unit
@pytest.mark.parametrize("empty", [None, "", "  "])
def test_no_name_leaves_the_package_default(empty):
    # None, not CrossEntropyLoss(): the default belongs to train_model's
    # signature, in one place.
    assert build_loss(empty) is None


@pytest.mark.unit
def test_build_loss_resolves_a_dotted_path():
    # A dotted path is case-sensitive - lower-casing it would never import.
    assert isinstance(build_loss("torch.nn.SmoothL1Loss"), nn.SmoothL1Loss)


@pytest.mark.unit
def test_list_params_become_float32_tensors():
    # torch registers `weight` as a buffer and refuses a list outright:
    # "cannot assign 'list' object to buffer 'weight'". YAML has no tensor.
    loss = build_loss("crossentropy", {"weight": [1.0, 3.0, 1.0]})
    assert isinstance(loss.weight, torch.Tensor)
    assert loss.weight.dtype == torch.float32, "float64 raises at the first call"
    assert loss.weight.tolist() == [1.0, 3.0, 1.0]


@pytest.mark.unit
def test_pos_weight_reaches_bcewithlogits():
    loss = build_loss("bcewithlogits", {"pos_weight": [1.0, 2.0, 1.0]})
    assert loss.pos_weight.tolist() == [1.0, 2.0, 1.0]


@pytest.mark.unit
def test_non_list_params_pass_through_untouched():
    loss = build_loss("crossentropy", {"label_smoothing": 0.1, "reduction": "sum"})
    assert loss.label_smoothing == 0.1
    assert loss.reduction == "sum"


@pytest.mark.unit
def test_as_tensors_leaves_scalars_and_strings_alone():
    out = as_tensors({"weight": [1.0, 2.0], "reduction": "mean", "ignore_index": -1})
    assert isinstance(out["weight"], torch.Tensor)
    assert out["reduction"] == "mean"
    assert out["ignore_index"] == -1


@pytest.mark.unit
def test_an_unknown_name_raises_and_names_the_available_keys():
    with pytest.raises(ValueError, match="Unknown loss_func: 'focalloss'") as err:
        build_loss("focalloss")
    for key in LOSSES:
        assert key in str(err.value)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [{"a": 1}, ["crossentropy"], 3])
def test_a_non_string_name_is_a_config_error(bad):
    with pytest.raises(ValueError, match="must be a string naming a loss"):
        build_loss(bad)


@pytest.mark.unit
def test_params_without_a_name_raise():
    # Otherwise the weights would be silently discarded.
    with pytest.raises(ValueError, match="without a loss_func"):
        build_loss(None, {"weight": [1.0, 2.0]})


@pytest.mark.unit
def test_a_bad_constructor_argument_is_a_config_error():
    with pytest.raises(ValueError, match="could not be constructed from"):
        build_loss("crossentropy", {"nonsense": 2})


@pytest.mark.unit
@pytest.mark.parametrize("path", ["os.system", "torch.optim.Adam"])
def test_a_dotted_path_that_is_not_a_module_class_is_rejected(path):
    with pytest.raises(ValueError, match="not a subclass of Module"):
        build_loss(path, {"command": "echo pwned"})
