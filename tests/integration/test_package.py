"""Integration tests for the xai4tsc package API — no experiment runner involved."""

import numpy as np
import pytest
import torch

import xai4tsc
from xai4tsc.data.datasets import LocalDataset
from xai4tsc.evaluation.evaluate import evaluate
from xai4tsc.models.models import load_model
from xai4tsc.xai.explain import generate_explanation


@pytest.mark.integration
def test_dataset_split_and_reload(ucr_dataset_path, tmp_path):
    """Split → save → reload produces numerically identical arrays."""
    ds = LocalDataset(ucr_dataset_path, name="GunPoint")
    splits_orig, _ = ds.split(
        train_split=0.8, val_split=0.1, random_state=42, encode="label"
    )
    ds.save_splits(tmp_path)

    ds2 = LocalDataset(ucr_dataset_path, name="GunPoint")
    splits_loaded, _ = ds2.load_saved_splits(tmp_path / "splits", encode="label")

    np.testing.assert_array_equal(splits_orig[0][0], splits_loaded[0][0])
    np.testing.assert_array_equal(splits_orig[1][0], splits_loaded[1][0])
    np.testing.assert_array_equal(splits_orig[0][1], splits_loaded[0][1])


@pytest.mark.integration
def test_model_trains_and_predicts(split_dataset, tmp_path):
    """FCN trains without error and predict() returns correctly shaped output."""
    splits, _ = split_dataset
    train_data, train_labels = splits[0][0], splits[0][1]
    test_data, test_labels = splits[1][0], splits[1][1]

    model = load_model(
        {"model": "FCN", "init_params": {"in_channels": 1, "num_classes": 2}},
        device="cpu",
        save_path=tmp_path,
    )
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
            "save_best": False,
        },
    )
    out_classes, _ = model.predict(test_data, test_labels)

    assert out_classes.shape[0] == len(test_data)
    assert np.all((out_classes >= 0) & (out_classes < 2))


@pytest.mark.integration
def test_lstm_trains_predicts_and_is_attributable(split_dataset, tmp_path):
    """LSTM trains, predicts, and yields finite gradient attributions (CPU)."""
    splits, encoder = split_dataset
    train_data, train_labels = splits[0][0], splits[0][1]
    test_data, test_labels = splits[1][0], splits[1][1]

    model = load_model(
        {
            "model": "LSTM",
            "init_params": {
                "in_channels": 1,
                "num_classes": 2,
                "hidden_size": 16,
                "num_layers": 1,
            },
        },
        device="cpu",
        save_path=tmp_path,
    )
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
            "save_best": False,
        },
    )
    out_classes, _ = model.predict(test_data, test_labels)
    assert out_classes.shape[0] == len(test_data)

    exp = generate_explanation(
        method="integrated_gradients",
        model=model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1],
        device="cpu",
    )
    assert exp.exp_values.shape == test_data[[0, 1]].shape
    assert np.all(np.isfinite(exp.exp_values))


@pytest.mark.integration
def test_patchtst_trains_predicts_and_is_attributable(split_dataset, tmp_path):
    """PatchTST trains, predicts, and yields finite gradient attributions (CPU)."""
    splits, encoder = split_dataset
    train_data, train_labels = splits[0][0], splits[0][1]
    test_data, test_labels = splits[1][0], splits[1][1]

    model = load_model(
        {
            "model": "PatchTST",
            "init_params": {
                "in_channels": 1,
                "num_classes": 2,
                "d_model": 16,
                "patch_len": 8,
                "stride": 4,
                "num_layers": 1,
                "nhead": 2,
            },
        },
        device="cpu",
        save_path=tmp_path,
    )
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
            "save_best": False,
        },
    )
    out_classes, _ = model.predict(test_data, test_labels)
    assert out_classes.shape[0] == len(test_data)

    exp = generate_explanation(
        method="integrated_gradients",
        model=model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1],
        device="cpu",
    )
    assert exp.exp_values.shape == test_data[[0, 1]].shape
    assert np.all(np.isfinite(exp.exp_values))


@pytest.mark.integration
def test_xlstm_trains_predicts_and_is_attributable(split_dataset, tmp_path):
    """XLSTM trains, predicts, and yields finite gradient attributions (CPU)."""
    splits, encoder = split_dataset
    train_data, train_labels = splits[0][0], splits[0][1]
    test_data, test_labels = splits[1][0], splits[1][1]

    model = load_model(
        {
            "model": "XLSTM",
            "init_params": {
                "in_channels": 1,
                "num_classes": 2,
                "embed_dim": 16,
                "num_blocks": 1,
                "num_heads": 2,
            },
        },
        device="cpu",
        save_path=tmp_path,
    )
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
            "save_best": False,
        },
    )
    out_classes, _ = model.predict(test_data, test_labels)
    assert out_classes.shape[0] == len(test_data)

    exp = generate_explanation(
        method="integrated_gradients",
        model=model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1],
        device="cpu",
    )
    assert exp.exp_values.shape == test_data[[0, 1]].shape
    assert np.all(np.isfinite(exp.exp_values))


@pytest.mark.integration
def test_explanation_shape_and_validity(trained_model, split_dataset):
    """Integrated Gradients explanation has correct shape and no NaN values."""
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    exp = generate_explanation(
        method="integrated_gradients",
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1],
        device="cpu",
    )

    assert exp.exp_values.shape == test_data[[0, 1]].shape
    assert not np.any(np.isnan(exp.exp_values))
    np.testing.assert_array_equal(exp.indices, [0, 1])


@pytest.mark.integration
def test_evaluation_returns_score(trained_model, split_dataset):
    """evaluate() with Complexity returns a finite non-NaN scalar."""
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    exp = generate_explanation(
        method="integrated_gradients",
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1],
        device="cpu",
    )

    score = evaluate(
        model=trained_model,
        metric="Complexity",
        explanation=exp,
        data=test_data[exp.indices],
        labels=test_labels[exp.indices],
        metric_class_params={
            "normalise": True,
            "abs": True,
            "disable_warnings": True,
            "return_aggregate": True,
        },
        metric_call_params={"softmax": False},
        device="cpu",
    )

    assert score is not None
    scalar = float(score[0]) if hasattr(score, "__len__") else float(score)
    assert np.isfinite(scalar)
    assert not np.isnan(scalar)


@pytest.mark.integration
@pytest.mark.parametrize("targets", ["predicted", "label", "all"])
def test_explanation_target_modes(trained_model, split_dataset, targets):
    """Each target mode yields a correctly shaped, finite attribution array."""
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    exp = generate_explanation(
        method="integrated_gradients",
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        targets=targets,
        indices=[0, 1],
        device="cpu",
    )

    if targets == "all":
        # One attribution per class: (n_classes, n_samples, C, T).
        n_classes = len(encoder.classes_)
        assert exp.exp_values.ndim == 4
        assert exp.exp_values.shape == (n_classes, *test_data[[0, 1]].shape)
    else:
        assert exp.exp_values.shape == test_data[[0, 1]].shape
    assert np.all(np.isfinite(exp.exp_values))


@pytest.mark.integration
@pytest.mark.parametrize("targets", ["predicted", "all"])
def test_plot_exp_saves_pngs(trained_model, split_dataset, tmp_path, targets):
    """Saving an explanation writes one PNG per sample (and per class for 'all')."""
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    generate_explanation(
        method="integrated_gradients",
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        targets=targets,
        indices=[0, 1],
        device="cpu",
        save_path=tmp_path,
        visualization_type=["bubbles"],
    )

    for idx in (0, 1):
        sample_dir = tmp_path / f"index_{idx}"
        assert sample_dir.is_dir()
        pngs = list(sample_dir.glob("relevance_class_*_bubbles*.png"))
        if targets == "all":
            # One plot per class.
            assert len(pngs) == len(encoder.classes_)
        else:
            assert len(pngs) >= 1


@pytest.mark.integration
@pytest.mark.parametrize(
    "method",
    [
        "integrated_gradients",
        "deeplift",
        "deconvolution",
        "guided_backpropagation",
        "occlusion",
        "tshap",
    ],
)
def test_all_explainers_produce_valid_attributions(
    trained_model, split_dataset, method
):
    """Every built-in explainer returns shape-matched, finite attributions."""
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    exp = generate_explanation(
        method=method,
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1],
        device="cpu",
    )

    assert exp.exp_values.shape == test_data[[0, 1]].shape
    assert np.all(np.isfinite(exp.exp_values))


_STFT = {"name": "stft", "params": {"n_fft": 16, "win_length": 16, "hop_length": 8}}
_FFT = {"name": "fft", "params": {}}


_FREQRISE_FFT = {"domain": "fft", "num_batches": 5, "batch_size": 4, "seed": 0}
# Small STFT grid (n_fft=16 → 9 freq bins), so num_cells must stay small.
_FREQRISE_STFT = {**_FREQRISE_FFT, "domain": "stft", "transform": _STFT, "num_cells": 4}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("method", "params", "expected_domain", "expected_ndim"),
    [
        ("freqrise", _FREQRISE_FFT, "Frequency", 3),
        ("freqrise", _FREQRISE_STFT, "Time-Frequency", 4),
        (
            "frequency",
            {"base": "integrated_gradients", "transform": _FFT},
            "Frequency",
            3,
        ),
        (
            "timefrequency",
            {"base": "guided_backpropagation", "transform": _STFT},
            "Time-Frequency",
            4,
        ),
        ("random_frequency", {"transform": _FFT, "seed": 0}, "Frequency", 3),
        ("random_timefrequency", {"transform": _STFT, "seed": 0}, "Time-Frequency", 4),
    ],
)
def test_freq_explainers_produce_valid_attributions(
    trained_model, split_dataset, method, params, expected_domain, expected_ndim
):
    """Each frequency/TF explainer runs end-to-end and returns finite attributions."""
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    exp = generate_explanation(
        method=method,
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        params=params,
        indices=[0, 1],
        device="cpu",
    )

    assert exp.explanation_domain.value == expected_domain
    assert exp.exp_values.shape[0] == 2  # two samples explained
    assert exp.exp_values.ndim == expected_ndim
    assert np.all(np.isfinite(exp.exp_values))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("method", "params", "metric"),
    [
        # FFT-mode FreqRISE (real relevance, RFFTransform) → frequency metric.
        (
            "freqrise",
            {
                "domain": "fft",
                "num_batches": 4,
                "batch_size": 4,
                "num_cells": 8,
                "seed": 0,
            },
            "Frequency Perturbation",
        ),
        # Explanation-space wrapper (complex relevance, reduced at the boundary).
        (
            "frequency",
            {"base": "integrated_gradients", "transform": _FFT},
            "Frequency Perturbation",
        ),
        # Time-frequency wrapper → TF metric.
        (
            "timefrequency",
            {"base": "guided_backpropagation", "transform": _STFT},
            "Time-Frequency Perturbation",
        ),
    ],
)
def test_freq_metric_end_to_end(trained_model, split_dataset, method, params, metric):
    """A frequency/TF explainer feeds its matching perturbation metric end-to-end."""
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    exp = generate_explanation(
        method=method,
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        params=params,
        indices=[0, 1],
        device="cpu",
    )
    result = np.asarray(
        evaluate(
            model=trained_model,
            metric=metric,
            explanation=exp,
            data=test_data[exp.indices],
            labels=test_labels[exp.indices],
            metric_class_params={"features_in_step": 5},
        )
    )
    assert result.shape[0] == 2
    assert np.all(np.isfinite(result))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("method", "params", "pattern"),
    [
        (
            "frequency",
            {"base": "integrated_gradients", "transform": _FFT},
            "*frequency*.png",
        ),
        (
            "timefrequency",
            {"base": "guided_backpropagation", "transform": _STFT},
            "*timefrequency*.png",
        ),
    ],
)
def test_freq_explanation_saves_plots(
    trained_model, split_dataset, tmp_path, method, params, pattern
):
    """Generating a freq/TF explanation with a save_path writes the domain plot."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    generate_explanation(
        method=method,
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        params=params,
        indices=[0, 1],
        device="cpu",
        save_path=tmp_path,
    )
    assert len(list(tmp_path.rglob(pattern))) == 2


@pytest.mark.integration
def test_integrated_gradients_param_branches(trained_model, split_dataset):
    """Non-default IG params (incl. internal_batch_size) run end-to-end."""
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    exp = generate_explanation(
        method="integrated_gradients",
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1],
        params={
            "n_steps": 10,
            "integration_method": "riemann_trapezoid",
            "internal_batch_size": 4,
            "multiply_by_inputs": False,
        },
        device="cpu",
    )

    assert exp.exp_values.shape == test_data[[0, 1]].shape
    assert np.all(np.isfinite(exp.exp_values))


@pytest.mark.integration
def test_occlusion_window_and_strides(trained_model, split_dataset):
    """Occlusion with explicit window_shape/strides/baseline runs end-to-end."""
    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    exp = generate_explanation(
        method="occlusion",
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1],
        params={"window_shape": [1, 3], "strides": 2, "baseline": 0},
        device="cpu",
    )

    assert exp.exp_values.shape == test_data[[0, 1]].shape
    assert np.all(np.isfinite(exp.exp_values))


@pytest.mark.integration
def test_custom_explainer_registration(trained_model, split_dataset):
    """A custom GradientExplainer subclass can be registered and used."""
    from captum.attr import Saliency

    from xai4tsc import GradientExplainer, register_explainer

    class SaliencyExplainer(GradientExplainer):
        def _get_captum_attribution(self, model):
            return Saliency(model)

    register_explainer("saliency", SaliencyExplainer)
    assert "saliency" in xai4tsc.EXPLAINERS

    splits, encoder = split_dataset
    test_data, test_labels = splits[1][0], splits[1][1]

    exp = generate_explanation(
        method="saliency",
        model=trained_model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1],
        device="cpu",
    )

    assert exp.exp_values.shape == test_data[[0, 1]].shape
    assert not np.any(np.isnan(exp.exp_values))


# ── Learnability: every model fits a trivially separable synthetic task ───────


def _make_separable_task(n_per_class=32, length=64, seed=0):
    """Two classes separable by global energy: class 1 carries an added bump."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.3, size=(2 * n_per_class, 1, length)).astype(np.float32)
    labels = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    lo, hi = length // 3, 2 * length // 3
    data[n_per_class:, 0, lo:hi] += 2.0  # class 1 has a bump -> higher energy
    perm = rng.permutation(2 * n_per_class)
    return data[perm], labels[perm]


# Same tiny configs as the unit gradient-flow test, one per built-in model.
_LEARN_CONFIGS = [
    {"model": "FCN", "init_params": {"filters": [8, 16, 8], "kernel_sizes": [7, 5, 3]}},
    {"model": "LeNet", "init_params": {"head_hidden": 8}},
    {"model": "ResNet", "init_params": {"n_feature_maps": 8}},
    {"model": "LSTM", "init_params": {"hidden_size": 16, "num_layers": 1}},
    {
        "model": "PatchTST",
        "init_params": {
            "d_model": 16,
            "patch_len": 8,
            "stride": 4,
            "num_layers": 1,
            "nhead": 2,
        },
    },
    {
        "model": "XLSTM",
        "init_params": {"embed_dim": 16, "num_blocks": 1, "num_heads": 2},
    },
]


@pytest.mark.integration
@pytest.mark.parametrize(
    "config", _LEARN_CONFIGS, ids=[c["model"] for c in _LEARN_CONFIGS]
)
def test_model_learns_separable_task(config, tmp_path):
    """Each model fits a trivially separable synthetic task (it actually learns)."""
    torch.manual_seed(0)
    data, labels = _make_separable_task()
    model = load_model(
        {
            "model": config["model"],
            "init_params": {
                "in_channels": 1,
                "num_classes": 2,
                **config["init_params"],
            },
        },
        device="cpu",
        save_path=tmp_path,
    )
    model.train_model(
        data,
        labels,
        hyperparams={
            "epochs": 60,
            "batchsize": 16,
            "loss_func": "CrossEntropy",
            "optimizer": "adam",
            "learn_rate": 0.01,
            "patience": 60,
            "save_best": False,
        },
    )
    preds, _ = model.predict(data, labels)
    acc = float((preds == labels).mean())
    assert acc > 0.85, f"{config['model']} only reached train accuracy {acc:.2f}"
