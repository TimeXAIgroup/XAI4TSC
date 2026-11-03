"""Unit tests for the loss and optimizer contract of ``ModelBase.train_model``."""

import numpy as np
import pytest
import torch
from torch import nn, optim

from xai4tsc.models.base import _batch_accuracy_counts, _batches, _target_tensor
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


@pytest.mark.unit
def test_a_negative_min_delta_still_stops_early(tmp_path, caplog):
    # Read as -0.005 the check would count a flat loss as improving forever.
    data, labels = _batch()
    with caplog.at_level("INFO", logger="xai4tsc.models.base"):
        _model(tmp_path).train_model(
            data,
            labels,
            epochs=12,
            batchsize=8,
            patience=3,
            learn_rate=0.0,
            min_delta=-5e-3,
            save_best=False,
        )
    assert "Early stopping at epoch 4" in caplog.text


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
    correct, total = _batch_accuracy_counts(outputs, targets)
    assert (int(correct), total) == (2, 3)


@pytest.mark.unit
def test_accuracy_counts_are_per_output_for_two_dimensional_targets():
    # Before this, argmax was compared against the rows: it raised when the batch
    # size differed from the column count, and silently reported above 100% when
    # they matched (133% for a 3x3 batch).
    outputs = torch.tensor([[2.0, -2.0, 2.0], [-2.0, 2.0, -2.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    correct, total = _batch_accuracy_counts(outputs, targets)
    assert (int(correct), total) == (5, 6)
    assert 0 <= 100 * correct / total <= 100


@pytest.mark.unit
def test_a_square_batch_no_longer_reports_above_one_hundred_percent():
    outputs = torch.randn(3, 3)
    targets = torch.randint(0, 2, (3, 3)).float()
    correct, total = _batch_accuracy_counts(outputs, targets)
    assert 100 * correct / total <= 100


# ── batching and data placement ───────────────────────────────────────────────


@pytest.mark.unit
def test_batches_cover_every_sample_with_a_short_last_batch():
    data = torch.arange(10.0).reshape(10, 1, 1)
    targets = torch.arange(10)
    batches = list(_batches((data, targets), 4, torch.device("cpu")))
    assert [len(x) for x, _ in batches] == [4, 4, 2]
    assert torch.equal(torch.cat([y for _, y in batches]), targets)


@pytest.mark.unit
def test_validation_loss_is_the_mean_per_sample_over_a_short_last_batch(tmp_path):
    # Batches of 4, 4 and 2: an unweighted mean of batch means would count the
    # last two samples twice as much as the others.
    rng = np.random.default_rng(0)
    data = torch.from_numpy(rng.standard_normal((10, 1, 32)).astype("float32"))
    targets = torch.tensor([0, 1] * 5)
    model = _model(tmp_path)
    loss_func = nn.CrossEntropyLoss()
    got = model._validation_loss((data, targets), loss_func, 4, torch.device("cpu"))
    with torch.no_grad():
        expected = loss_func(model(data), targets).item()
    assert got == pytest.approx(expected, rel=1e-6)


def _trained_state(tmp_path, device, data_on_device):
    data, labels = _batch()
    torch.manual_seed(0)
    model = load_model(_TINY_FCN, device=device, save_path=tmp_path / str(device))
    model.train_model(
        data,
        labels,
        data_val=data,
        labels_val=labels,
        data_on_device=data_on_device,
        # 16 samples in batches of 3: the pinned buffers are reused and the last
        # batch is short.
        **{**_HP, "batchsize": 3},
    )
    return {k: v.cpu() for k, v in model.state_dict().items()}


@pytest.mark.unit
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs a CUDA device"
            ),
        ),
    ],
)
def test_data_on_device_does_not_change_the_trained_weights(tmp_path, device):
    on = _trained_state(tmp_path / "on", device, True)
    off = _trained_state(tmp_path / "off", device, False)
    # On a GPU the kernel a library picks can depend on where a batch's buffer
    # lies in memory, which changes float32 rounding; on the CPU they are equal.
    tolerance = {} if device == "cpu" else {"rtol": 1e-3, "atol": 1e-4}
    for key in on:
        torch.testing.assert_close(on[key], off[key], **tolerance)


@pytest.mark.unit
def test_auto_keeps_data_on_the_cpu_when_it_does_not_fit(tmp_path, monkeypatch, caplog):
    data = torch.zeros(10, 1, 100)  # 4000 bytes
    targets = torch.zeros(10, dtype=torch.int64)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1000, 10**9))
    with caplog.at_level("INFO", logger="xai4tsc.models.base"):
        train, val = _model(tmp_path)._place_data(
            (data, targets), None, torch.device("cuda"), "auto", 1.0
        )
    assert train[0] is data and val is None
    assert "kept on the CPU" in caplog.text


@pytest.mark.unit
def test_auto_honours_the_device_memory_fraction(tmp_path, monkeypatch):
    # 4080 bytes of data and 6000 bytes free: it would fit whole, not in half.
    data = torch.zeros(10, 1, 100)
    targets = torch.zeros(10, dtype=torch.int64)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (6000, 10**9))
    train, _ = _model(tmp_path)._place_data(
        (data, targets), None, torch.device("cuda"), "auto", 0.5
    )
    assert train[0] is data


@pytest.mark.unit
@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_device_memory_fraction_outside_zero_one_raises(tmp_path, fraction):
    with pytest.raises(ValueError, match="device_memory_fraction"):
        _model(tmp_path)._place_data(
            (torch.zeros(2, 1, 4), torch.zeros(2)),
            None,
            torch.device("cpu"),
            "auto",
            fraction,
        )


@pytest.mark.unit
def test_auto_on_mps_measures_against_the_recommended_memory(tmp_path, monkeypatch):
    # 4080 bytes of data; Metal recommends 10000 bytes, of which 4000 are in use.
    data = torch.zeros(10, 1, 100)
    targets = torch.zeros(10, dtype=torch.int64)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 10000)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 4000)
    train, _ = _model(tmp_path)._place_data(
        (data, targets), None, torch.device("mps"), "auto", 0.5
    )
    assert train[0] is data, "4080 bytes do not fit in half of 6000"


@pytest.mark.unit
def test_auto_keeps_data_on_the_cpu_on_a_cpu_device(tmp_path):
    data = torch.zeros(4, 1, 8)
    train, _ = _model(tmp_path)._place_data(
        (data, torch.zeros(4)), None, torch.device("cpu"), "auto", 1.0
    )
    assert train[0] is data


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
