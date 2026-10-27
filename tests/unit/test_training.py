"""Unit tests for the loss and optimizer contract of ``ModelBase.train_model``."""

import numpy as np
import pytest
import torch
from torch import nn, optim

from xai4tsc.models.base import _batch_accuracy_counts, _target_tensor
from xai4tsc.models.models import load_model

_TINY_FCN = {
    "model": "FCN",
    "init_params": {
        "in_channels": 1,
        "num_classes": 2,
        "filters": [4, 8, 4],
        "kernel_sizes": [7, 5, 3],
    },
}
_HP = {"epochs": 2, "batchsize": 8, "patience": 5, "save_best": False}


def _batch(n_classes=2, multi_label=False):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((16, 1, 32)).astype("float32")
    if multi_label:
        labels = rng.integers(0, 2, size=(16, n_classes)).astype(np.int64)
    else:
        labels = np.array(list(range(n_classes)) * (16 // n_classes))
    return data, labels


def _model(tmp_path, n_classes=2):
    config = {
        **_TINY_FCN,
        "init_params": {**_TINY_FCN["init_params"], "num_classes": n_classes},
    }
    return load_model(config, device="cpu", save_path=tmp_path)


# ── the target dtype follows the rank ─────────────────────────────────────────


@pytest.mark.unit
def test_target_tensor_dtype_follows_the_rank():
    # CrossEntropyLoss takes int64 indices and rejects int32 and float32;
    # BCEWithLogitsLoss needs float and rejects int64. The rank decides which.
    assert _target_tensor(np.array([0, 1, 2])).dtype == torch.int64
    assert _target_tensor(np.array([[0, 1], [1, 0]])).dtype == torch.float32


# ── early stopping ───────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("save_best", [True, False])
def test_early_stopping_does_not_depend_on_save_best(tmp_path, caplog, save_best):
    # A zero learning rate keeps the loss flat: epoch 1 improves on the start
    # value, the next `patience` epochs do not. Once, best_loss was only updated
    # when save_best was set, so without it every epoch counted as improving.
    data, labels = _batch()
    with caplog.at_level("INFO", logger="xai4tsc.models.base"):
        _model(tmp_path).train_model(
            data,
            labels,
            epochs=12,
            batchsize=8,
            patience=3,
            learn_rate=0.0,
            save_best=save_best,
        )
    assert "Early stopping at epoch 4" in caplog.text


@pytest.mark.unit
def test_min_delta_decides_what_counts_as_an_improvement(tmp_path, caplog):
    # The best loss starts at 1e6, so a min_delta that large makes even the first
    # epoch no improvement and training stops one epoch before the default does.
    data, labels = _batch()
    with caplog.at_level("INFO", logger="xai4tsc.models.base"):
        _model(tmp_path).train_model(
            data,
            labels,
            epochs=12,
            batchsize=8,
            patience=3,
            learn_rate=0.0,
            min_delta=1e6,
            save_best=False,
        )
    assert "Early stopping at epoch 3 (patience=3, min_delta=1e+06)" in caplog.text


# ── the loss: default, object, and the two compatibility layers ───────────────


@pytest.mark.unit
def test_the_default_loss_trains_exactly_as_the_explicit_one(tmp_path):
    # `None` must be indistinguishable from what the whitelist used to build.
    data, labels = _batch()
    torch.manual_seed(0)
    default = _model(tmp_path / "a").train_model(data, labels, **_HP)
    torch.manual_seed(0)
    explicit = _model(tmp_path / "b").train_model(
        data, labels, loss_func=nn.CrossEntropyLoss(), **_HP
    )
    for left, right in zip(
        default.state_dict().values(), explicit.state_dict().values(), strict=True
    ):
        assert torch.equal(left, right)


@pytest.mark.unit
def test_a_weighted_loss_is_accepted_and_changes_the_training(tmp_path):
    data, labels = _batch()
    weighted = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]))
    torch.manual_seed(0)
    a = _model(tmp_path / "a").train_model(data, labels, **_HP)
    torch.manual_seed(0)
    b = _model(tmp_path / "b").train_model(data, labels, loss_func=weighted, **_HP)
    assert any(
        not torch.equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )


@pytest.mark.unit
def test_an_index_loss_against_per_output_targets_raises(tmp_path):
    # The one case a smoke call cannot catch: CrossEntropyLoss accepts a 2-D
    # float target, reads each row as soft class probabilities over a competitive
    # softmax, and optimises a different objective without complaint.
    data, labels = _batch(n_classes=3, multi_label=True)
    with pytest.raises(ValueError, match="bcewithlogits"):
        _model(tmp_path, n_classes=3).train_model(data, labels, **_HP)


@pytest.mark.unit
def test_a_mismatched_loss_raises_before_the_first_epoch(tmp_path, caplog):
    # Layer 1: any other mismatch becomes a startup error naming both sides.
    data, labels = _batch()
    with (
        caplog.at_level("INFO", logger="xai4tsc.models.base"),
        pytest.raises(ValueError, match="cannot score this model's output"),
    ):
        _model(tmp_path).train_model(
            data, labels, loss_func=nn.BCEWithLogitsLoss(), **_HP
        )
    assert "Epoch" not in caplog.text, "it must fail before any epoch runs"


@pytest.mark.unit
def test_a_weight_of_the_wrong_length_raises_before_the_first_epoch(tmp_path):
    data, labels = _batch()
    wrong = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))  # 2 classes
    with pytest.raises(ValueError, match="cannot score this model's output"):
        _model(tmp_path).train_model(data, labels, loss_func=wrong, **_HP)


@pytest.mark.unit
def test_per_output_targets_train_with_bcewithlogits(tmp_path, caplog):
    # The combination B exists to enable, end to end: a 2-D target, a float cast
    # at the tensor boundary, and an accuracy that is reported per output.
    data, labels = _batch(n_classes=3, multi_label=True)
    with caplog.at_level("INFO", logger="xai4tsc.models.base"):
        model = _model(tmp_path, n_classes=3).train_model(
            data, labels, loss_func=nn.BCEWithLogitsLoss(), **_HP
        )
    assert model.best_epoch is not None
    assert "bit_acc" in caplog.text, "a per-output accuracy must say so"
    assert "nan" not in caplog.text.lower()


# ── the accuracy counts ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_accuracy_counts_are_argmax_for_index_targets():
    outputs = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    assert _batch_accuracy_counts(outputs, targets) == (2, 3)


@pytest.mark.unit
def test_accuracy_counts_are_per_output_for_two_dimensional_targets():
    # Before this, argmax was compared against the rows: it raised when the batch
    # size differed from the column count, and silently reported above 100% when
    # they matched (133% for a 3x3 batch).
    outputs = torch.tensor([[2.0, -2.0, 2.0], [-2.0, 2.0, -2.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    correct, total = _batch_accuracy_counts(outputs, targets)
    assert (correct, total) == (5, 6)
    assert 0 <= 100 * correct / total <= 100


@pytest.mark.unit
def test_a_square_batch_no_longer_reports_above_one_hundred_percent():
    outputs = torch.randn(3, 3)
    targets = torch.randint(0, 2, (3, 3)).float()
    correct, total = _batch_accuracy_counts(outputs, targets)
    assert 100 * correct / total <= 100


# ── the optimizer: class, factory, instance ───────────────────────────────────


@pytest.mark.unit
def test_the_default_optimizer_trains_exactly_as_adam(tmp_path):
    data, labels = _batch()
    torch.manual_seed(0)
    default = _model(tmp_path / "a").train_model(data, labels, **_HP)
    torch.manual_seed(0)
    explicit = _model(tmp_path / "b").train_model(
        data, labels, optimizer=optim.Adam, **_HP
    )
    for left, right in zip(
        default.state_dict().values(), explicit.state_dict().values(), strict=True
    ):
        assert torch.equal(left, right)


@pytest.mark.unit
def test_an_optimizer_class_is_constructed_with_the_learn_rate(tmp_path):
    data, labels = _batch()
    model = _model(tmp_path)
    model.train_model(data, labels, optimizer=optim.SGD, learn_rate=0.05, **_HP)
    # Nothing in the repo exercised a non-Adam optimizer before this.
    assert model.best_epoch is not None


@pytest.mark.unit
def test_a_factory_carries_its_constructor_arguments(tmp_path):
    from functools import partial

    data, labels = _batch()
    model = _model(tmp_path)
    model.train_model(data, labels, optimizer=partial(optim.SGD, momentum=0.9), **_HP)
    assert model.best_epoch is not None


@pytest.mark.unit
def test_an_optimizer_instance_is_used_and_reports_that_learn_rate_is_ignored(
    tmp_path, caplog
):
    data, labels = _batch()
    model = _model(tmp_path)
    instance = optim.SGD(model.parameters(), lr=0.123)
    with caplog.at_level("WARNING", logger="xai4tsc.models.base"):
        model.train_model(data, labels, optimizer=instance, learn_rate=0.001, **_HP)
    assert "does not apply" in caplog.text
    assert "0.123" in caplog.text, "the lr actually in force must be recorded"
