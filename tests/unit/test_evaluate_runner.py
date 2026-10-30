"""Unit tests for the runner's metric adapter in experiment_runner/evaluate.py."""

import pytest
from experiment_runner.evaluate import build_evaluator
from experiment_runner.lookups import build_perturb_func
from quantus.functions.perturb_func import baseline_replacement_by_indices


@pytest.mark.unit
def test_build_perturb_func_names_a_quantus_function():
    func = build_perturb_func("baseline_replacement_by_indices")
    assert func is baseline_replacement_by_indices


@pytest.mark.unit
@pytest.mark.parametrize("name", ["math.sqrt", "garbage", "random"])
def test_build_perturb_func_rejects_anything_else(name):
    with pytest.raises(ValueError, match="perturb_func"):
        build_perturb_func(name)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["black", "white", "mean", "centroid", "random"])
def test_a_baseline_name_on_a_quantus_metric_is_rejected(name):
    # A baseline name used to reach Quantus as TSHAP's (C, T) function; the message
    # says what to write instead. Raised before the metric runs.
    conf = {
        "metric": "Region Segmentation",
        "metric_class_params": {"perturb_func": name},
        "metric_call_params": {},
    }
    with pytest.raises(
        ValueError, match=f"Region Segmentation: .*perturb_baseline: {name}"
    ):
        build_evaluator(conf)


@pytest.mark.unit
def test_build_evaluator_maps_a_quantus_name_to_its_class():
    import quantus

    from xai4tsc.evaluation import QuantusEvaluator

    evaluator = build_evaluator(
        {"metric": "Region Segmentation", "metric_class_params": {"patch_size": 4}}
    )
    assert isinstance(evaluator, QuantusEvaluator)
    assert evaluator.quantus_class is quantus.RegionPerturbation


@pytest.mark.unit
def test_build_evaluator_filters_params_a_native_metric_does_not_take():
    from xai4tsc.evaluation.frequency_evaluate import FrequencyEvaluator

    evaluator = build_evaluator(
        {
            "metric": "Frequency Perturbation",
            "metric_class_params": {"normalise": True, "features_in_step": 3},
        }
    )
    assert isinstance(evaluator, FrequencyEvaluator)
    assert evaluator.features_in_step == 3
