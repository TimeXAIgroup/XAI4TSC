"""
Unit tests for the data domain: the domain a dataset's arrays are expressed in.

Covers the config spelling resolver, where each dataset kind gets its domain from,
how it reaches the Explanation, how it decides a plain attributor's explanation
domain, and the refusal of a domain-lifting explainer on data it cannot transform.
"""

import numpy as np
import pytest
import torch
from torch import nn

from xai4tsc.data import load_dataset
from xai4tsc.data.base import DatasetBase
from xai4tsc.data.datasets import FreqShapesDataset, LocalDataset
from xai4tsc.xai import Domain
from xai4tsc.xai._types import DATA_DOMAINS, resolve_data_domain
from xai4tsc.xai.explain import _get_explanation

_STFT = {"name": "stft", "params": {"n_fft": 16, "win_length": 16, "hop_length": 8}}


class _TinyNet(nn.Module):
    def __init__(self, channels: int, timesteps: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(channels * timesteps, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))

    def predict(self, x):
        with torch.no_grad():
            p = torch.softmax(
                self(torch.as_tensor(np.asarray(x), dtype=torch.float32)), 1
            ).numpy()
        return p.argmax(1), p


def _explain(method, params, data_domain):
    data = np.random.default_rng(0).standard_normal((3, 1, 33)).astype(np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    return _get_explanation(
        method=method,
        model=_TinyNet(1, 33, 2).eval(),
        data=data,
        labels=labels,
        encoder=None,
        params=params,
        targets="label",
        device="cpu",
        data_domain=data_domain,
    )


# ── resolving the config spelling ─────────────────────────────────────────────


@pytest.mark.unit
def test_resolve_data_domain_accepts_names_and_members():
    assert resolve_data_domain("time") is Domain.TIME
    assert resolve_data_domain("frequency") is Domain.FREQUENCY
    assert resolve_data_domain("time_frequency") is Domain.TIME_FREQUENCY
    assert resolve_data_domain(Domain.FREQUENCY) is Domain.FREQUENCY
    assert set(DATA_DOMAINS) == {"time", "frequency", "time_frequency"}


@pytest.mark.unit
def test_resolve_data_domain_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown data_domain 'freq'"):
        resolve_data_domain("freq")


# ── where each dataset kind gets its domain ───────────────────────────────────


@pytest.mark.unit
def test_time_domain_is_the_default():
    assert DatasetBase.data_domain is Domain.TIME


@pytest.mark.unit
def test_local_dataset_takes_its_domain_from_the_caller():
    # Only the user knows what is in the file, so this one is a parameter.
    assert LocalDataset("/tmp/x", name="x").data_domain is Domain.TIME
    assert (
        LocalDataset("/tmp/x", name="x", data_domain="frequency").data_domain
        is Domain.FREQUENCY
    )


@pytest.mark.unit
def test_freq_shapes_is_time_domain_despite_its_name():
    # The *discriminative features* are frequencies localized in time; the arrays
    # themselves are raw series, so a frequency explainer still applies to it.
    assert FreqShapesDataset.data_domain is Domain.TIME


@pytest.mark.unit
def test_load_dataset_ignores_the_setting_for_a_synthetic_dataset(caplog):
    with caplog.at_level("WARNING", logger="xai4tsc.data.datasets"):
        ds = load_dataset("freq_shapes", data_domain="frequency")
    assert ds.data_domain is Domain.TIME
    assert "is ignored" in caplog.text


@pytest.mark.unit
def test_load_dataset_forwards_the_setting_to_a_local_dataset():
    ds = load_dataset("local", path="/tmp/x", data_domain="time_frequency")
    assert ds.data_domain is Domain.TIME_FREQUENCY


# ── what it decides downstream ────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "data_domain", [Domain.TIME, Domain.FREQUENCY, Domain.TIME_FREQUENCY]
)
def test_a_plain_attributor_explains_in_the_data_domain(data_domain):
    # Relevance is assigned to the features the model was given, so the attribution
    # lives wherever its input does: the model-native domain.
    exp = _explain("integrated_gradients", {}, data_domain)
    assert exp.data_domain is data_domain
    assert exp.explanation_domain is data_domain


@pytest.mark.unit
def test_a_lifting_explainer_still_names_its_own_output_domain():
    exp = _explain("frequency", {"base": "integrated_gradients"}, Domain.TIME)
    assert exp.data_domain is Domain.TIME
    assert exp.explanation_domain is Domain.FREQUENCY


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("frequency", {"base": "integrated_gradients"}),
        ("timefrequency", {"base": "integrated_gradients", "transform": _STFT}),
        ("freqrise", {"domain": "fft"}),
        ("random_frequency", {}),
    ],
)
def test_a_lifting_explainer_is_refused_on_data_it_cannot_transform(method, params):
    # Lifting frequency data into the frequency domain would transform it twice.
    with pytest.raises(ValueError, match="consumes"):
        _explain(method, params, Domain.FREQUENCY)


# ── config validation ─────────────────────────────────────────────────────────


def _cfg(explainers, metrics, data_domain):
    return {
        "general": {},
        "data_config": {
            "datasets": [
                {"dataset": "spectra", "path": "/tmp/x", "data_domain": data_domain}
            ]
        },
        "train_config": {"models": [{"model": "FCN"}]},
        "explanation_config": {"explainers": explainers},
        "evaluation_config": {"metrics": metrics},
    }


@pytest.mark.unit
def test_config_accepts_an_input_perturbing_metric_on_frequency_data():
    # The model-native rule in practice: Pixel-Flipping is not "time only", it
    # follows the model, and on spectra a plain attributor satisfies it.
    from experiment_runner.config import config_sanity_check

    assert config_sanity_check(
        _cfg(
            [{"method": "Integrated_Gradients"}],
            [{"metric": "Pixel-Flipping"}],
            "frequency",
        )
    )


@pytest.mark.unit
def test_config_accepts_a_native_frequency_metric_on_frequency_data():
    from experiment_runner.config import config_sanity_check

    assert config_sanity_check(
        _cfg(
            [{"method": "Integrated_Gradients"}],
            [{"metric": "Frequency Perturbation"}],
            "frequency",
        )
    )


@pytest.mark.unit
def test_config_rejects_a_lifting_explainer_on_frequency_data():
    from experiment_runner.config import config_sanity_check

    with pytest.raises(ValueError, match="consumes"):
        config_sanity_check(
            _cfg([{"method": "FreqRISE"}], [{"metric": "Complexity"}], "frequency")
        )


@pytest.mark.unit
def test_configured_data_domains_reads_each_dataset_kind():
    from experiment_runner.config import _configured_data_domains

    config = {
        "data_config": {
            "datasets": [
                {"dataset": "ECG200"},  # UCR: always raw series
                {"dataset": "freq_shapes"},  # synthetic: declares its own
                {"dataset": "spectra", "path": "/tmp/x", "data_domain": "frequency"},
            ]
        }
    }
    assert _configured_data_domains(config) == {Domain.TIME, Domain.FREQUENCY}
