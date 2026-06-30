import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xai4tsc.data.datasets import LocalDataset, UcrUeaDataset
from xai4tsc.models.models import load_model

logger = logging.getLogger(__name__)

_CACHE_DEFAULT = Path(__file__).parent / "cache"


def _cache_root() -> Path:
    env = os.environ.get("XAI4TSC_TEST_CACHE")
    return Path(env) if env else _CACHE_DEFAULT


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def ucr_dataset_path():
    """
    Download GunPoint once and return the path to its raw data directory.

    Uses ``XAI4TSC_TEST_CACHE`` env var as root, falling back to ``./tests/cache``.
    Subsequent sessions reuse the cached files without re-downloading.
    """
    cache = _cache_root()
    ucr_dir = cache / "ucr"
    dataset_dir = ucr_dir / "GunPoint"

    if (dataset_dir / "data.npy").exists():
        logger.info("GunPoint already cached at %s — skipping download", dataset_dir)
    else:
        logger.info("Downloading GunPoint into %s …", dataset_dir)
        ucr_dir.mkdir(parents=True, exist_ok=True)
        ds = UcrUeaDataset("GunPoint", cache_dir=ucr_dir, download=True)
        data, labels, _ = ds.load()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.save(dataset_dir / "data.npy", data)
        pd.DataFrame({"label": labels}).to_json(
            dataset_dir / "label.json", orient="records"
        )

    return dataset_dir


@pytest.fixture(scope="session")
def split_dataset(ucr_dataset_path):
    """
    Return ``(splits, encoder)`` for GunPoint with fixed parameters.

    Caches split files to ``cache_root/splits/GunPoint_tr0.8_v0.1_s42_label/``
    so the expensive split is done at most once per test cache.
    """
    cache = _cache_root()
    split_cache = cache / "splits" / "GunPoint_tr0.8_v0.1_s42_label"

    ds = LocalDataset(ucr_dataset_path, name="GunPoint")

    split_dir = split_cache / "splits"
    if split_dir.exists():
        logger.info("Loading cached splits from %s", split_dir)
        splits, encoder = ds.load_saved_splits(split_dir, encode="label")
    else:
        logger.info("Splitting GunPoint and caching to %s", split_cache)
        splits, encoder = ds.split(
            train_split=0.8,
            val_split=0.1,
            random_state=42,
            encode="label",
        )
        ds.save_splits(split_cache)

    return splits, encoder


@pytest.fixture(scope="session")
def trained_model(split_dataset, tmp_path_factory):
    """
    Return an ``FCN`` model trained for 2 epochs on GunPoint train split.

    Session-scoped so all integration tests share one trained model instance
    and training runs only once.
    """
    splits, _ = split_dataset
    train_data, train_labels = splits[0][0], splits[0][1]

    save_path = tmp_path_factory.mktemp("trained_model")
    model_config = {
        "model": "FCN",
        "init_params": {"in_channels": 1, "num_classes": 2},
    }
    model = load_model(model_config, device="cpu", save_path=save_path)
    model.train_model(
        train_data,
        train_labels,
        hyperparams={
            "epochs": 2,
            "batchsize": 8,
            "loss_func": "CrossEntropy",
            "optimizer": "adam",
            "learn_rate": 0.001,
            "patience": 2,
            "save_best": True,
        },
    )
    return model
