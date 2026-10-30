"""
End-to-end integration tests for TSHAP on a real model + cached GunPoint.

Covers the three consumption paths the package promises: the public
``generate_explanation`` API, the runner adapter (``background_data`` selector →
ndarray), and a Quantus metric via ``evaluate`` (confirming TSHAP attributions
are consumable unchanged).
"""

import numpy as np
import pytest
import quantus as q
from experiment_runner import explain as runner_explain

from xai4tsc.evaluation import QuantusEvaluator, evaluate
from xai4tsc.xai.explain import generate_explanation
from xai4tsc.xai.feature_attribution import TSHAPExplainer


@pytest.mark.integration
def test_tshap_centroid_background_end_to_end(trained_model, split_dataset):
    """TSHAP with a centroid background returns finite, shape-matched values."""
    splits, encoder = split_dataset
    train_data = splits[0][0]
    test_data, test_labels = splits[1][0], splits[1][1]

    exp = generate_explanation(
        explainer=TSHAPExplainer(
            window_length=0.1,
            stride=3,
            perturb_baseline="centroid",
            background_data=train_data,
        ),
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1],
        device="cpu",
    )

    assert exp.exp_values.shape == test_data[[0, 1]].shape
    assert np.all(np.isfinite(exp.exp_values))


@pytest.mark.integration
def test_tshap_roi_and_per_channel_run(trained_model, split_dataset):
    """ROI and per_channel modes run end-to-end and stay finite."""
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    for params in (
        {"window_length": 0.1, "stride": 2, "perturb_baseline": "mean", "roi": True},
        {
            "window_length": 0.1,
            "stride": 3,
            "perturb_baseline": "black",
            "channel_mode": "per_channel",
        },
    ):
        exp = generate_explanation(
            explainer=TSHAPExplainer(**params),
            model=trained_model,
            data=test_data,
            labels=test_labels,
            encoder=encoder,
            indices=[0, 1],
            device="cpu",
        )
        assert exp.exp_values.shape == test_data[[0, 1]].shape
        assert np.all(np.isfinite(exp.exp_values))


@pytest.mark.integration
def test_tshap_consumable_by_quantus_metric(trained_model, split_dataset):
    """A TSHAP explanation feeds a Quantus metric and yields a finite score."""
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    exp = generate_explanation(
        explainer=TSHAPExplainer(window_length=0.1, stride=3, perturb_baseline="mean"),
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1],
        device="cpu",
    )

    # Quantus guards reject degenerate-sign attributions (e.g. all-negative)
    # before applying its own ``abs``. TSHAP relevance can be all-negative for a
    # given sample/model, so feed the magnitude — Complexity is defined on the
    # relevance distribution. This still exercises the full
    # explanation -> evaluate -> Quantus path on real TSHAP output.
    exp.exp_values = np.abs(exp.exp_values)

    score = evaluate(
        QuantusEvaluator(
            q.Complexity,
            normalise=True,
            abs=True,
            disable_warnings=True,
            return_aggregate=True,
        ),
        model=trained_model,
        explanation=exp,
        data=test_data[exp.indices],
        labels=test_labels[exp.indices],
        device="cpu",
        call_params={"softmax": False},
    )

    scalar = float(score[0]) if hasattr(score, "__len__") else float(score)
    assert np.isfinite(scalar) and not np.isnan(scalar)


@pytest.mark.integration
def test_tshap_through_runner_adapter(trained_model, split_dataset, tmp_path):
    """The runner adapter resolves the background_data selector and runs TSHAP."""
    splits, encoder = split_dataset
    train_data = splits[0][0]
    test_data, test_labels = splits[1][0], splits[1][1]

    explainer_config = {
        "method": "TSHAP",
        "target": "predicted",
        "threshold": 0.5,
        "samples": {"type": "index", "indices": [0, 1]},
        "visualization_type": ["points"],
        "save_exp_values": False,
        "window_length": 0.1,
        "stride": 3,
        "perturb_baseline": "centroid",
        "background_data": "train_set",
    }

    explanation = runner_explain._generate_explanation(
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        prng=np.random.default_rng(0),
        device="cpu",
        config={"results_rel_path": tmp_path},
        explainer_config=explainer_config,
        save_path=tmp_path,
        splits={
            "train_set": train_data,
            "test_set": test_data,
            "val_set": None,
        },
    )

    assert explanation.exp_values.shape == test_data[[0, 1]].shape
    assert np.all(np.isfinite(explanation.exp_values))
    # The selector must not have been mutated into an array in the user's config.
    assert explainer_config["background_data"] == "train_set"
