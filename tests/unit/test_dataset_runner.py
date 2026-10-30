"""Unit tests for the runner's dataset dispatch - constructs datasets, loads none."""

import pytest
from experiment_runner.config import config_resolve
from experiment_runner.main import MASTER_CONFIG, _build_dataset

from xai4tsc.data import LocalDataset, SynthMultilabelFreqDataset, UcrUeaDataset


def _entry(tmp_path, **dataset):
    """Return one dataset entry resolved against master, as the loop sees it."""
    user = {
        "experiment_name": "dataset_dispatch",
        "results_rel_path": str(tmp_path),
        "data_config": {"datasets": [dataset]},
        "train_config": {"models": []},
        "explanation_config": {"explainers": []},
        "evaluation_config": {"metrics": []},
    }
    return config_resolve(user, MASTER_CONFIG)["data_config"]["datasets"][0]


@pytest.mark.unit
def test_a_catalog_name_builds_the_synthetic_class_from_init_params(tmp_path):
    entry = _entry(
        tmp_path,
        dataset="Synth_Multilabel_Freq",
        init_params={"sample_length": 600},
        max_samples=5,
    )
    ds = _build_dataset(entry, tmp_path)
    assert isinstance(ds, SynthMultilabelFreqDataset)
    assert ds.sample_length == 600
    assert ds._max_samples == 5


@pytest.mark.unit
def test_a_path_builds_a_local_dataset(tmp_path):
    entry = _entry(
        tmp_path,
        dataset="my_ecg",
        path=str(tmp_path),
        max_samples=5,
    )
    ds = _build_dataset(entry, tmp_path)
    assert isinstance(ds, LocalDataset)
    assert ds.name == "my_ecg"
    assert ds._max_samples == 5


@pytest.mark.unit
def test_any_other_name_builds_a_ucr_uea_dataset(tmp_path):
    entry = _entry(tmp_path, dataset="GunPoint", download=False)
    ds = _build_dataset(entry, tmp_path)
    assert isinstance(ds, UcrUeaDataset)
    assert ds.name == "GunPoint"
