"""Integration tests that run the shipped example scripts end to end."""

import importlib.util
from pathlib import Path

import pytest

from xai4tsc.data.datasets import UcrUeaDataset

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "getting_started.py"


def _load_example():
    """Import the example by path, without putting examples/ on sys.path."""
    spec = importlib.util.spec_from_file_location("getting_started", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_getting_started_trains_a_model_that_is_not_degenerate(
    ucr_dataset_path, tmp_path
):
    """
    The flagship example must not ship a one-class predictor.

    It used to: with ``use_predefined_splits=True`` the 80/10 split it asked for
    was discarded, so there was no validation set, early stopping fell back to
    training loss and the shipped result was ``spec=0.0000``, ``TN=0``,
    ``mcc=-0.083`` - worse than chance by MCC - whose explanations the example
    then scored.
    """
    # The fixture has already downloaded GunPoint into <cache>/ucr/GunPoint, and
    # <cache>/ucr is where sktime looks when it is handed cache_dir.
    results = _load_example().main(
        str(tmp_path), cache_dir=str(ucr_dataset_path.parent)
    )

    # The real defect was degeneracy, not a low score: a predictor that answers
    # one class for every input gets TN=0 and specificity 0. That is a far
    # steadier thing to assert than a threshold on 20 test samples.
    assert results["tp"] > 0, f"predicts no positives: {results}"
    assert results["tn"] > 0, f"predicts no negatives: {results}"
    assert float(results["accuracy"]) > 60, f"below majority baseline: {results}"
    assert float(results["mcc"]) > 0, f"no better than chance by MCC: {results}"

    # The example's own artifacts land in out_dir, named after the model.
    assert (tmp_path / "FCN_training_curve.png").exists()
    assert list(tmp_path.glob("FCN_epoch_*.pt"))


@pytest.mark.integration
def test_official_splits_warn_about_the_fractions_they_ignore(ucr_dataset_path, caplog):
    """The warning must fire on the real predefined-split path, not just in unit."""
    ds = UcrUeaDataset("GunPoint", cache_dir=str(ucr_dataset_path.parent))

    with caplog.at_level("WARNING", logger="xai4tsc.data.datasets"):
        splits, _ = ds.split(
            train_split=0.8,
            val_split=0.1,
            random_state=42,
            use_predefined_splits=True,
        )

    assert "val_split=0.1" in caplog.text
    assert "no validation set" in caplog.text
    # Behaviour is unchanged: the archive partition, and no third split.
    assert len(splits) == 2
    assert [len(y) for _, y, _ in splits] == [50, 150]
