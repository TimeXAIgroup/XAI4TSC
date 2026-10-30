"""Unit tests for experiment_runner/lookups.py - the string to a torch factory."""

from functools import partial

import pytest
import torch
from experiment_runner.lookups import OPTIMIZERS, build_optimizer
from torch import optim


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("adam", optim.Adam),
        ("adamw", optim.AdamW),
        ("sgd", optim.SGD),
        ("rmsprop", optim.RMSprop),
    ],
)
def test_build_optimizer_returns_the_named_class(name, expected):
    # With no parameters to bind it is the bare class, which keeps logs readable.
    assert build_optimizer(name) is expected


@pytest.mark.unit
def test_the_table_is_matched_case_insensitively():
    assert build_optimizer("AdamW") is optim.AdamW


@pytest.mark.unit
@pytest.mark.parametrize("empty", [None, "", "  "])
def test_no_name_leaves_the_package_default(empty):
    assert build_optimizer(empty) is None


@pytest.mark.unit
def test_params_are_bound_into_a_partial():
    # The optimizer cannot be constructed here: it needs model.parameters().
    built = build_optimizer("sgd", {"momentum": 0.9, "nesterov": True})
    assert isinstance(built, partial)
    assert built.func is optim.SGD
    assert built.keywords == {"momentum": 0.9, "nesterov": True}
    # It is callable exactly the way train_model calls it.
    weights = [torch.nn.Parameter(torch.zeros(2))]
    instance = built(weights, lr=0.01)
    assert isinstance(instance, optim.SGD)
    assert instance.param_groups[0]["momentum"] == 0.9
    assert instance.param_groups[0]["lr"] == 0.01


@pytest.mark.unit
def test_build_optimizer_resolves_a_dotted_path():
    assert build_optimizer("torch.optim.NAdam") is optim.NAdam


@pytest.mark.unit
def test_an_unknown_name_raises_and_names_the_available_keys():
    with pytest.raises(ValueError, match="Unknown optimizer: 'nadam'") as err:
        build_optimizer("nadam")  # the table holds no nadam; the dotted path does
    for key in OPTIMIZERS:
        assert key in str(err.value)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [{"a": 1}, ["sgd"], 3])
def test_a_non_string_name_is_a_config_error(bad):
    with pytest.raises(ValueError, match="must be a string naming an optimizer"):
        build_optimizer(bad)


@pytest.mark.unit
def test_params_without_a_name_raise():
    with pytest.raises(ValueError, match="without an optimizer"):
        build_optimizer(None, {"momentum": 0.9})


@pytest.mark.unit
@pytest.mark.parametrize("path", ["os.system", "torch.nn.MSELoss"])
def test_a_dotted_path_that_is_not_an_optimizer_class_is_rejected(path):
    with pytest.raises(ValueError, match="not a subclass of Optimizer"):
        build_optimizer(path)
