"""
Entry point for the xai4tsc experiment runner.

Run from the repository root as a module::

    python -m experiment_runner.main --conf path/to/config.yaml

Orchestrates the full evaluation pipeline as a nested loop over datasets,
models, explainers, and metrics. Results are saved as ``metrics.csv`` files
at three granularity levels (global, per-dataset, per-model) under
``results_rel_path``.
"""

import argparse
import copy
import dataclasses
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from xai4tsc.data import (
    SYNTHETIC_DATASETS,
    LocalDataset,
    SyntheticDataset,
    load_dataset,
)
from xai4tsc.evaluation.evaluate import METRICS
from xai4tsc.logging_config import add_file_log, remove_file_log
from xai4tsc.models.models import load_model
from xai4tsc.utils.utils import merge_dicts
from xai4tsc.xai import is_multi_target

from .cache import get_dataset_cache_dir, get_split_cache_path
from .config import config_resolve, config_sanity_check, disambiguate
from .download_datasets import ensure_datasets_cached
from .evaluate import _animate, _evaluate
from .explain import _explainer_display_name, _generate_explanation
from .log_setup import _add_file_logging, _setup_console_logging
from .utils import (
    _METRICS_COLUMNS,
    _aggregate_metric,
    _format_label_list,
    _result_row,
    expand_datasets,
)

logger = logging.getLogger("xai4tsc.runner")

_HERE = Path(__file__).parent
MASTER_CONFIG = _HERE / "configs" / "master.yaml"


def main(config_path: str, debug: bool = False) -> int:
    """
    Run a complete XAI evaluation experiment from a YAML configuration file.

    Orchestrates the full pipeline as a nested loop over datasets → models →
    explainers → metrics. Results are saved as ``metrics.csv`` at three
    granularity levels (global, per-dataset, per-model) under
    ``results_rel_path``.

    Failures are isolated at the level they occur: a raising metric costs only
    that metric, a raising explainer only that explainer, and a raising model
    only that model, so the rest of the dataset's results are still collected
    and written.

    Returns
    -------
    int
        Total number of failures across all four levels (datasets, models,
        explainers and metric evaluations). ``0`` means everything configured
        either completed or was explicitly skipped, so callers can treat a
        non-zero value as "this run needs attention" (the CLI turns it into a
        non-zero exit status).
    """
    config, run_log = initial_setup(config_path, debug=debug)
    try:
        return _run(config)
    finally:
        # Detach the run-level log, so a second run in this process does not
        # also write into this one's experiment.log.
        remove_file_log(run_log)


def _run(config: dict) -> int:
    """
    Execute the pipeline for an already-resolved *config*.

    Split from :func:`main` so the run-level log handler attached during setup is
    always detached, whatever the run does.

    Parameters
    ----------
    config : dict
        Configuration as returned by :func:`initial_setup`.

    Returns
    -------
    int
        Total number of failures, as described on :func:`main`.
    """
    config = config_resolve(config, master_path=MASTER_CONFIG)
    config_sanity_check(config)
    log_config = {**config, "results_rel_path": str(config["results_rel_path"])}
    logger.info(
        "Resolved configuration:\n%s",
        yaml.dump(log_config, default_flow_style=False, sort_keys=False),
    )

    g_con = config["general"]
    d_con = config["data_config"]
    t_con = config["train_config"]
    e_con = config["explanation_config"]
    eval_con = config["evaluation_config"]

    # Detect wildcard usage before expansion (needed for bulk download decision).
    _uses_ucr = any(e.get("dataset") == "UCR" for e in d_con["datasets"])
    _uses_uea = any(e.get("dataset") == "UEA" for e in d_con["datasets"])

    # Expand "UCR" / "UEA" keywords and resolve per-member overrides.
    d_con["datasets"] = expand_datasets(d_con["datasets"])

    # Construct identifiers for each dataset/model/explainer/metric, allows
    # running same instances multiple times
    dataset_ids = disambiguate([d["dataset"] for d in d_con["datasets"]])
    model_ids = disambiguate([m["model"] for m in t_con["models"]])
    explainer_ids = disambiguate(
        [_explainer_display_name(e) for e in e_con["explainers"]]
    )
    metric_ids = disambiguate([m["metric"] for m in eval_con["metrics"]])
    logger.info("Running experiment on %d dataset(s).", len(d_con["datasets"]))

    # Make results reproducible
    seed = config["general"]["seed"]
    reproducible = config["general"]["reproducible"]
    if reproducible:
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        logger.info("Setting seed to: %s", seed)
    else:
        logger.info(
            "reproducible: false - splits are drawn unseeded and are neither "
            "read from nor written to the split cache."
        )
    prng = np.random.default_rng(seed if reproducible else None)
    legacy_prng = int(prng.integers(2**31))

    # check for computational devices
    if g_con["device"] == "use_available":
        if torch.cuda.is_available():
            # Nvidia cuda backend
            device = "cuda"
        elif torch.backends.mps.is_available():
            # Mac OS backend
            device = "mps"
        else:
            # Fallback: CPU
            device = "cpu"
    else:
        device = g_con["device"]
    logger.info("Using device %s for computation", device)
    device = torch.device(device)

    # Start
    logger.info("--- Starting Experiment: %s ---", config["experiment_name"])
    metric_rows = []

    cache_path = Path(config["cache_path"]) if config.get("cache_path") else None
    cache_dir = get_dataset_cache_dir(cache_path)

    if cache_dir is not None:
        ensure_datasets_cached(
            cache_dir,
            [d["dataset"] for d in d_con["datasets"]],
            uses_ucr_wildcard=_uses_ucr,
            uses_uea_wildcard=_uses_uea,
        )

    # Capture per-dataset outcomes, to report in the summary.
    completed_datasets: list[str] = []
    skipped_datasets: list[str] = []
    failed_datasets: list[str] = []
    # Failures inside a dataset are isolated at their own level so one bad
    # component does not discard the rest of the dataset's results. Each list
    # holds "dataset/model/explainer/metric" style labels, trimmed to the level
    # that failed.
    failed_models: list[str] = []
    failed_explainers: list[str] = []
    failed_metrics: list[str] = []

    # Step 1: Prepare the data
    for d_ind, dataset in enumerate(d_con["datasets"]):
        ds = splits = encoder = None
        train_data = train_labels = test_data = test_labels = None
        dataset_id = dataset_ids[d_ind]
        try:
            # Log dataset preparation
            logger.info(
                "%s/%s: Preparing data from dataset %s",
                d_ind + 1,
                len(d_con["datasets"]),
                dataset["dataset"],
            )

            # Fill remaining gaps from default_settings (per-member overrides
            # were already applied during dataset expansion).
            dataset = merge_dicts(dataset, d_con["default_settings"])

            if dataset.get("skip_dataset"):
                logger.info(
                    "Skipping dataset '%s' (skip_dataset: true).", dataset["dataset"]
                )
                skipped_datasets.append(dataset_id)
                continue

            dataset_path = Path(config["results_rel_path"]) / dataset_id
            dataset_path.mkdir(parents=True, exist_ok=True)

            if dataset.get("is_presplit"):
                # Pre-split local data: load directly, skip split() and cache entirely
                ds = LocalDataset(dataset["path"], name=dataset["dataset"])
                splits, encoder = ds.load_saved_splits(
                    Path(dataset["path"]), encode=dataset["encode"]
                )
            else:
                use_predefined_splits = dataset.get("use_predefined_splits", False)
                is_synthetic = dataset["dataset"] in SYNTHETIC_DATASETS
                if is_synthetic:
                    # Synthetic datasets are generated from params (forwarded via
                    # `init_params`); they take no path/download/format.
                    ds = load_dataset(
                        name=dataset["dataset"],
                        cache_dir=cache_dir,
                        max_samples=dataset.get("max_samples"),
                        sample_strategy=dataset.get("sample_strategy", "random"),
                        max_series_length=dataset.get("max_series_length"),
                        series_position=dataset.get("series_position", "first"),
                        **dataset.get("init_params", {}),
                    )
                else:
                    ds = load_dataset(
                        name=dataset["dataset"],
                        path=dataset.get("path") or None,
                        cache_dir=cache_dir,
                        download=dataset.get("download", True),
                        data_format=dataset.get("format", "numpy"),
                        pad_series=dataset.get("allow_padding", False),
                        max_samples=dataset.get("max_samples"),
                        sample_strategy=dataset.get("sample_strategy", "random"),
                        max_series_length=dataset.get("max_series_length"),
                        series_position=dataset.get("series_position", "first"),
                        use_predefined_splits=use_predefined_splits,
                        layout=dataset.get("layout", "auto"),
                    )

                # Synthetic auto-detect: a pre-split layout already in the
                # synthetic cache dir is loaded directly (skip generation + split).
                synth_dir = (
                    ds.cache_subdir(ds.cache_dir)
                    if is_synthetic and ds.cache_dir is not None
                    else None
                )
                if synth_dir is not None and SyntheticDataset.is_split_layout(
                    synth_dir
                ):
                    logger.info(
                        "Loading pre-split synthetic dataset from %s", synth_dir
                    )
                    splits, encoder = ds.load_saved_splits(
                        synth_dir, encode=dataset["encode"]
                    )
                else:
                    split_cache = get_split_cache_path(
                        cache_path if reproducible else None,
                        dataset["dataset"],
                        dataset["train_split"],
                        dataset["val_split"],
                        legacy_prng,
                        dataset["encode"],
                        allow_padding=dataset.get("allow_padding", False),
                        allow_imputation=dataset.get("allow_imputation", False),
                        max_samples=dataset.get("max_samples"),
                        sample_strategy=dataset.get("sample_strategy", "random"),
                        max_series_length=dataset.get("max_series_length"),
                        series_position=dataset.get("series_position", "first"),
                        stratify=dataset.get("stratify", True),
                        use_predefined_splits=use_predefined_splits,
                        data_format=dataset.get("format", "numpy"),
                        path=dataset.get("path") or None,
                    )
                    split_dir = split_cache / "splits" if split_cache else None
                    if split_dir and split_dir.exists():
                        logger.info("Loading cached splits from %s", split_dir)
                        splits, encoder = ds.load_saved_splits(
                            split_dir, encode=dataset["encode"]
                        )
                    else:
                        splits, encoder = ds.split(
                            train_split=dataset["train_split"],
                            val_split=dataset["val_split"],
                            encode=dataset["encode"],
                            impute_missing=dataset.get("allow_imputation", False),
                            rng=prng,
                            stratify=dataset.get("stratify", True),
                            class_order=dataset.get("classes") or None,
                        )
                        if split_cache:
                            logger.info("Saving splits to cache: %s", split_cache)
                            ds.save_splits(split_cache)

            train_data, train_labels = splits[0][0], splits[0][1]
            test_data, test_labels = splits[1][0], splits[1][1]
            val_data, val_labels = (
                (splits[2][0], splits[2][1]) if len(splits) > 2 else (None, None)
            )
            # Per-sample test metadata (ground-truth localization, when present)
            # rides through to explanations for ground-truth metrics. Normalise to
            # a position-indexable list of records (split() yields a list of dicts;
            # load_saved_splits yields a DataFrame).
            test_metadata = splits[1][2] if len(splits[1]) > 2 else None
            if hasattr(test_metadata, "to_dict"):
                test_metadata = test_metadata.to_dict("records")

            if dataset.get("skip_models"):
                logger.info(
                    "Skipping model/explainer/metrics for '%s' (skip_models: true).",
                    dataset["dataset"],
                )
                continue

            # Step 2: Train the models
            for m_ind, model_config in enumerate(t_con["models"]):
                model_wrapper = None
                model_log = None
                # Deep-copy so auto-detected values don't persist into the next
                # dataset iteration via the shared YAML config object.
                model_config = copy.deepcopy(model_config)
                try:
                    # Update config
                    entry_sets_train = "train" in model_config
                    model_config = merge_dicts(model_config, t_con["default_settings"])
                    if "model_ckpt_path" in model_config and not entry_sets_train:
                        model_config["train"] = None

                    # Auto-detect num_classes and in_channels when not explicitly set.
                    init_params = model_config.setdefault("init_params", {})
                    if init_params.get("num_classes") is None:
                        if hasattr(encoder, "classes_"):
                            init_params["num_classes"] = len(encoder.classes_)
                        else:
                            init_params["num_classes"] = int(
                                np.unique(train_labels).size
                            )
                        logger.info(
                            "Auto-detected num_classes=%d for dataset '%s'",
                            init_params["num_classes"],
                            dataset["dataset"],
                        )
                    if init_params.get("in_channels") is None:
                        init_params["in_channels"] = int(train_data.shape[1])
                        logger.info(
                            "Auto-detected in_channels=%d for dataset '%s'",
                            init_params["in_channels"],
                            dataset["dataset"],
                        )

                    model_id = model_ids[m_ind]
                    model_path = dataset_path / model_id
                    model_path.mkdir(parents=True, exist_ok=True)

                    # Attach a per-model log file capturing this model's
                    # training, explanation, and evaluation. Detached in the
                    # finally below so it never leaks into the next model.
                    model_log = add_file_log(
                        model_path / f"{dataset_id}_{model_id}.log"
                    )

                    # Load the model
                    model_wrapper = load_model(
                        model_config, device, save_path=model_path
                    )

                    # Should the model be trained
                    if not model_config.get("train"):
                        logger.info(
                            "%s/%s: Not training %s ('train' is not set) - %s.",
                            m_ind + 1,
                            len(t_con["models"]),
                            model_config["model"],
                            "explaining the weights loaded from the checkpoint"
                            if "model_ckpt_path" in model_config
                            else "the model is freshly initialised",
                        )
                    else:
                        # Log training start
                        logger.info(
                            "%s/%s: Training %s model...",
                            m_ind + 1,
                            len(t_con["models"]),
                            model_config["model"],
                        )
                        # Train the model (validation drives early stopping
                        # when a val split is present)
                        model_wrapper.train_model(
                            train_data,
                            train_labels,
                            model_config["hyperparams"],
                            data_val=val_data,
                            labels_val=val_labels,
                            save_path=model_path,
                        )
                        # Evaluate on test data
                        model_wrapper.evaluate_model(
                            test_data,
                            test_labels,
                            model_config["hyperparams"],
                            save_path=model_path,
                        )

                    if dataset.get("skip_explainers"):
                        logger.info(
                            "Skipping explainer/metrics for '%s' "
                            "(skip_explainers: true).",
                            dataset["dataset"],
                        )
                        continue

                    # Step 3: Generate explanations
                    for e_ind, explainer_config in enumerate(e_con["explainers"]):
                        explanation = None
                        try:
                            # Log explanation generation. For wrapper methods
                            # show "WRAPPER-BASE" (e.g. SIGN-Integrated_Gradients).
                            explainer = explainer_ids[e_ind]
                            explanation_dir = (
                                model_path
                                / "explanations"
                                / f"explanations_{explainer}"
                            )
                            logger.info(
                                "%s/%s: Generating %s explanations...",
                                e_ind + 1,
                                len(e_con["explainers"]),
                                explainer,
                            )

                            # Update settings
                            explainer_config = merge_dicts(
                                explainer_config, e_con["default_settings"]
                            )

                            explanation = _generate_explanation(
                                model=model_wrapper,
                                data=test_data,
                                labels=test_labels,
                                encoder=encoder,
                                prng=prng,
                                device=device,
                                config=config,
                                explainer_config=explainer_config,
                                data_config=dataset,
                                save_path=model_path,
                                splits={
                                    "train_set": train_data,
                                    "test_set": test_data,
                                    "val_set": val_data,
                                },
                                metadata=test_metadata,
                                explanation_dir=explanation_dir,
                            )

                            logger.info(
                                "Finished generating %s explanations!", explainer
                            )

                            if dataset.get("skip_metrics"):
                                logger.info(
                                    "Skipping metrics for '%s' (skip_metrics: true).",
                                    dataset["dataset"],
                                )
                                continue

                            # Step 4: Evaluate explanations
                            logger.info("Starting to evaluate method %s", explainer)
                            for metric_ind, metric_config in enumerate(
                                eval_con["metrics"]
                            ):
                                logger.info(
                                    "%s/%s: Evaluating %s...",
                                    metric_ind + 1,
                                    len(eval_con["metrics"]),
                                    metric_config["metric"],
                                )
                                metric_name = metric_config["metric"]
                                metric_id = metric_ids[metric_ind]
                                metric_label = (
                                    f"{dataset_id}/{model_id}/{explainer}/{metric_id}"
                                )

                                # Skip domain-restricted metrics on incompatible
                                # explanations (e.g. a frequency metric on a
                                # time-domain attribution)
                                required_domains = getattr(
                                    METRICS.get(metric_config["metric"]),
                                    "required_domains",
                                    None,
                                )
                                if (
                                    required_domains
                                    and explanation.explanation_domain
                                    not in required_domains
                                ):
                                    domains = {d.value for d in required_domains}
                                    logger.info(
                                        "Skipping %s: requires domain %s but "
                                        "explanation is %s.",
                                        metric_config["metric"],
                                        domains,
                                        explanation.explanation_domain.value,
                                    )
                                    metric_rows.append(
                                        _result_row(
                                            dataset_id,
                                            model_id,
                                            explainer,
                                            metric_ids[metric_ind],
                                            "skipped-domain",
                                            detail=(
                                                f"requires {sorted(domains)}, "
                                                "explanation is "
                                                f"{explanation.explanation_domain.value}"
                                            ),
                                        )
                                    )
                                    continue
                                try:
                                    # Update settings
                                    metric_config = merge_dicts(
                                        metric_config, eval_con["default_settings"]
                                    )

                                    exp_data = test_data[explanation.indices]

                                    # Multi-target ("all") stacks one attribution per
                                    # class on a leading axis. Detection is domain-aware
                                    # (via is_multi_target), so a single-target
                                    # time-frequency explanation — also 4-D — is not
                                    # mistaken for a multi-target one, and freq/TF
                                    # multi-target stacks are split just like time ones.
                                    multi_target = is_multi_target(explanation)
                                    if multi_target:
                                        # target="all": one attribution per class.
                                        n_classes = explanation.exp_values.shape[0]
                                        class_scores = []
                                        for class_idx in range(n_classes):
                                            class_exp = dataclasses.replace(
                                                explanation,
                                                exp_values=explanation.exp_values[
                                                    class_idx
                                                ],
                                            )
                                            class_labels = np.full(
                                                len(explanation.indices), class_idx
                                            )
                                            result = _evaluate(
                                                model=model_wrapper,
                                                metric_conf=metric_config,
                                                explanation=class_exp,
                                                data=exp_data,
                                                labels=class_labels,
                                                device=device,
                                            )
                                            agg = _aggregate_metric(result)
                                            if agg is not None:
                                                class_scores.append(agg)
                                        score = (
                                            float(np.mean(class_scores))
                                            if class_scores
                                            else None
                                        )
                                    else:
                                        evaluation = _evaluate(
                                            model=model_wrapper,
                                            metric_conf=metric_config,
                                            explanation=explanation,
                                            data=exp_data,
                                            labels=test_labels[explanation.indices],
                                            device=device,
                                        )
                                        score = _aggregate_metric(evaluation)

                                    if score is None:
                                        metric_rows.append(
                                            _result_row(
                                                dataset_id,
                                                model_id,
                                                explainer,
                                                metric_id,
                                                "nan",
                                                detail=(
                                                    "metric returned an empty or "
                                                    "all-NaN result"
                                                ),
                                            )
                                        )
                                    else:
                                        metric_rows.append(
                                            _result_row(
                                                dataset_id,
                                                model_id,
                                                explainer,
                                                metric_id,
                                                "ok",
                                                score=score,
                                            )
                                        )

                                    # Optional: render a perturbation-process GIF
                                    if metric_config.get("animate"):
                                        anim_dir = explanation_dir
                                        anim_dir.mkdir(parents=True, exist_ok=True)
                                        metric_slug = metric_id.replace(" ", "_")
                                        _animate(
                                            model=model_wrapper,
                                            metric_conf=metric_config,
                                            explanation=explanation,
                                            data=exp_data,
                                            labels=test_labels[explanation.indices],
                                            save_path=anim_dir
                                            / f"animation_{metric_slug}.gif",
                                        )
                                except Exception as exc:
                                    failed_metrics.append(metric_label)
                                    metric_rows.append(
                                        _result_row(
                                            dataset_id,
                                            model_id,
                                            explainer,
                                            metric_id,
                                            "error",
                                            detail=f"{type(exc).__name__}: {exc}",
                                        )
                                    )
                                    logger.exception(
                                        "Metric '%s' failed for %s, skipping to "
                                        "the next metric.",
                                        metric_name,
                                        metric_label,
                                    )

                        except Exception:
                            failed_explainers.append(
                                f"{dataset_id}/{model_id}/{explainer}"
                            )
                            logger.exception(
                                "Explainer '%s' failed on %s/%s, skipping to the "
                                "next explainer.",
                                explainer,
                                dataset["dataset"],
                                model_config["model"],
                            )
                        finally:
                            del explanation
                            gc.collect()
                except Exception:
                    failed_models.append(f"{dataset_id}/{model_id}")
                    logger.exception(
                        "Model '%s' failed on dataset '%s', skipping to the next "
                        "model.",
                        model_config["model"],
                        dataset["dataset"],
                    )
                finally:
                    remove_file_log(model_log)
                    del model_wrapper
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif device.type == "mps":
                        torch.mps.empty_cache()
                    gc.collect()

            completed_datasets.append(dataset_id)
        except Exception:
            failed_datasets.append(dataset_id)
            logger.exception("Dataset '%s' failed, skipping.", dataset["dataset"])
        finally:
            del ds, splits, encoder, train_data, train_labels, test_data, test_labels
            gc.collect()

    df = pd.DataFrame(metric_rows, columns=_METRICS_COLUMNS)

    # Global — all results
    global_path = Path(config["results_rel_path"]) / "metrics.csv"
    df.to_csv(global_path, index=False)
    if metric_rows:
        logger.info("Global metrics saved to %s", global_path)
    else:
        logger.warning(
            "No metric results were produced; wrote a header-only %s. Check the "
            "failures reported below and that the configured metrics are "
            "compatible with the configured explainers.",
            global_path,
        )

    if metric_rows:
        # Per dataset — derive path from "dataset" column
        for dataset_name, group in df.groupby("dataset"):
            path = (
                Path(config["results_rel_path"])
                / dataset_name
                / f"{dataset_name}_metrics.csv"
            )
            group.to_csv(path, index=False)
            logger.info("Dataset metrics saved to %s", path)

        # Per model — derive path from "dataset" + "model" columns
        for (dataset_name, model_name), group in df.groupby(["dataset", "model"]):
            path = (
                Path(config["results_rel_path"])
                / dataset_name
                / model_name
                / f"{dataset_name}_{model_name}_metrics.csv"
            )
            group.to_csv(path, index=False)
            logger.info("Model metrics saved to %s", path)

    # Print dataset summary
    logger.info(
        "\n--- Experiment Finished: %d/%d dataset(s) completed, %d skipped, "
        "%d failed ---",
        len(completed_datasets),
        len(d_con["datasets"]),
        len(skipped_datasets),
        len(failed_datasets),
    )
    if skipped_datasets:
        logger.info("Skipped dataset(s): %s", _format_label_list(skipped_datasets))
    if failed_datasets:
        logger.error(
            "Failed dataset(s): %s - see the tracebacks above.",
            _format_label_list(failed_datasets),
        )

    # Failures inside a dataset are reported separately
    for label, failures in (
        ("model(s)", failed_models),
        ("explainer(s)", failed_explainers),
        ("metric evaluation(s)", failed_metrics),
    ):
        if failures:
            logger.error(
                "Failed %s (%d): %s - see the tracebacks above.",
                label,
                len(failures),
                _format_label_list(failures),
            )

    return (
        len(failed_datasets)
        + len(failed_models)
        + len(failed_explainers)
        + len(failed_metrics)
    )


def initial_setup(
    config_path: str, debug: bool = False
) -> tuple[dict, logging.Handler]:
    """
    Set up experiment as described in the given configuration file.

    Parameters
    ----------
    config_path : str
        Path to the experiment configuration.
    debug : bool
        Enable debug-level logging.

    Returns
    -------
    tuple[dict, logging.Handler]
        The extracted configuration and the run-level log handler, which the
        caller detaches when the run ends.
    """
    _setup_console_logging(debug)

    # Read the experiment configuration
    path = Path(config_path).absolute()
    with open(path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            msg = f"Error loading configuration file from:\n{path}"
            e.add_note(msg)
            raise

    logger.info("--- Starting setup ---")
    # Generate a folder for the experiment:
    if "results_rel_path" in config and config["results_rel_path"] is not None:
        path = Path(config["results_rel_path"]) / config["experiment_name"]
        if Path(path).resolve().exists():
            timestamp = datetime.now(datetime.now().astimezone().tzinfo)
            timestamp = timestamp.strftime("%Y%m%d%H%M%S")
            if Path(path / timestamp).exists():
                logger.error("Could not create results directory!")
                sys.exit(-1)
            logger.warning("Specified directory already exists")
            logger.warning("Creating directory with timestamp")
            path = path.with_name(path.name + f"_{timestamp}" + path.suffix)
        try:
            logger.info("Creating results directory: %s", path)
            path.mkdir(parents=True)
            try:
                config["results_rel_path"] = path.relative_to(Path.cwd())
            except ValueError:
                config["results_rel_path"] = path  # absolute path — keep as-is
        except Exception as error:
            logger.exception("Could not create results directory: %s", error)
            sys.exit(1)

    # Add file logging now that the results directory exists
    run_log = _add_file_logging(debug, log_path=path / "experiment.log")

    return config, run_log


if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser(
        prog="xai4tsc", description="Run an experiment for time series classification."
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=str,
        default=str(_HERE / "configs" / "example.yaml"),
        help="Configuration for the experiment run",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    # Exit non-zero if any dataset failed
    sys.exit(1 if main(args.conf, debug=args.debug) else 0)
