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
from sktime.datasets.tsc_dataset_names import (
    multivariate as _UEA_NAMES,  # noqa: N812 — uppercase denotes a constant
)
from sktime.datasets.tsc_dataset_names import (
    univariate as _UCR_NAMES,  # noqa: N812 — uppercase denotes a constant
)

from xai4tsc.data import (
    SYNTHETIC_DATASETS,
    LocalDataset,
    SyntheticDataset,
    load_dataset,
)
from xai4tsc.evaluation.evaluate import METRICS
from xai4tsc.models.models import load_model
from xai4tsc.utils.utils import merge_dicts
from xai4tsc.xai import Domain

from .cache import get_dataset_cache_dir, get_split_cache_path
from .config import config_resolve, config_sanity_check
from .download_datasets import ensure_datasets_cached
from .evaluate import _animate, _evaluate
from .explain import _explainer_display_name, _generate_explanation
from .log_setup import _add_file_logging, _setup_console_logging

logger = logging.getLogger("xai4tsc.runner")

_HERE = Path(__file__).parent
MASTER_CONFIG = _HERE / "configs" / "master.yaml"

# Reserved keywords that expand to all datasets in the corresponding archive.
_ARCHIVE_KEYWORDS = {"UCR": _UCR_NAMES, "UEA": _UEA_NAMES}


def expand_datasets(datasets: list[dict]) -> list[dict]:
    """
    Expand archive wildcards and resolve per-member ``overrides``.

    Each entry is either a concrete dataset or an archive keyword
    (``UCR`` / ``UEA``) that expands to all its member datasets. An entry's
    inline settings apply to every member it produces; an optional
    ``overrides`` map (``{dataset_name: {setting: value}}``) sets per-member
    exceptions that take precedence over the entry's inline settings.

    Parameters
    ----------
    datasets : list[dict]
        Raw ``data_config["datasets"]`` entries.

    Returns
    -------
    list[dict]
        Concrete dataset entries with ``overrides`` resolved and stripped.
    """
    expanded: list[dict] = []
    for entry in datasets:
        overrides = entry.get("overrides") or {}
        base = {k: v for k, v in entry.items() if k != "overrides"}
        archive = _ARCHIVE_KEYWORDS.get(base.get("dataset"))
        if archive is not None:
            for name in set(overrides) - set(archive):
                logger.warning(
                    "Ignoring override for '%s': not a member of archive '%s'.",
                    name,
                    base["dataset"],
                )
            for name in archive:
                member = {**base, "dataset": name}
                if name in overrides:
                    # overrides[name] wins over the entry's inline settings.
                    member = merge_dicts(copy.deepcopy(overrides[name]), member)
                expanded.append(member)
        else:
            if overrides:
                logger.warning(
                    "Ignoring 'overrides' on non-archive dataset '%s'; "
                    "set per-dataset values inline instead.",
                    base.get("dataset"),
                )
            expanded.append(base)
    return expanded


def _aggregate_metric(evaluation: object) -> float | None:
    """
    Reduce a Quantus metric result to a single float.

    Most Quantus metrics return one score per explained instance; average them
    so the recorded value reflects the whole sample, not just the first element.
    Scalars pass through unchanged. Returns ``None`` for empty/all-NaN results.
    """
    if evaluation is None:
        return None
    arr = np.asarray(evaluation, dtype=float).ravel()
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


def main(config_path: str, debug: bool = False) -> None:
    """
    Run a complete XAI evaluation experiment from a YAML configuration file.

    Orchestrates the full pipeline as a nested loop over datasets → models →
    explainers → metrics. Results are saved as ``metrics.csv`` at three
    granularity levels (global, per-dataset, per-model) under
    ``results_rel_path``.
    """
    config = initial_setup(config_path, debug=debug)
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
    logger.info("Running experiment on %d dataset(s).", len(d_con["datasets"]))

    # Make results reproducible
    if config["general"]["reproducible"]:
        seed = config["general"]["seed"]
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        logger.info("Setting seed to: %s", seed)
        prng = np.random.default_rng(seed)
        legacy_prng = int(prng.integers(2**31))  # used only for split cache key

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

    # Step 1: Prepare the data
    for d_ind, dataset in enumerate(d_con["datasets"]):
        ds = splits = encoder = None
        train_data = train_labels = test_data = test_labels = None
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
                continue

            dataset_path = Path(config["results_rel_path"]) / dataset["dataset"]
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
                        cache_path,
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
                # Deep-copy so auto-detected values don't persist into the next
                # dataset iteration via the shared YAML config object.
                model_config = copy.deepcopy(model_config)
                try:
                    # Update config
                    model_config = merge_dicts(model_config, t_con["default_settings"])

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

                    model_path = dataset_path / model_config["model"]
                    model_path.mkdir(parents=True, exist_ok=True)

                    # Load the model
                    model_wrapper = load_model(
                        model_config, device, save_path=model_path
                    )

                    # Should the model be trained
                    if "train" in model_config and model_config["train"] is not None:
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
                            explainer = _explainer_display_name(explainer_config)
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

                                # Skip domain-restricted metrics on incompatible
                                # explanations (e.g. a frequency metric on a
                                # time-domain attribution), so mixed configs don't
                                # abort. config_sanity_check already guarantees each
                                # such metric has at least one compatible explainer.
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
                                    logger.info(
                                        "Skipping %s: requires domain %s but "
                                        "explanation is %s.",
                                        metric_config["metric"],
                                        {d.value for d in required_domains},
                                        explanation.explanation_domain.value,
                                    )
                                    continue

                                # Update settings
                                metric_config = merge_dicts(
                                    metric_config, eval_con["default_settings"]
                                )

                                exp_data = test_data[explanation.indices]

                                # Multi-target ("all") is a time-domain 4-D stack
                                # (n_classes, n, C, T). A time-frequency explanation
                                # is ALSO 4-D (n, C, n_freq, n_time) but single
                                # target, so gate on the domain to avoid mistaking
                                # it for multi-target.
                                multi_target = (
                                    explanation.explanation_domain == Domain.TIME
                                    and explanation.exp_values.ndim == 4
                                )
                                if multi_target:
                                    # target="all": one attribution per class.
                                    # Evaluate each class separately (Quantus
                                    # needs a 3-D a_batch) and average. y_batch is
                                    # the explained class so the metric measures
                                    # faithfulness w.r.t. that class's output.
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

                                if score is not None:
                                    metric_rows.append(
                                        {
                                            "dataset": dataset["dataset"],
                                            "model": model_config["model"],
                                            "explainer": explainer,
                                            "metric": metric_config["metric"],
                                            "score": score,
                                        }
                                    )

                                # Optional: render a perturbation-process GIF
                                # (only reached for compatible freq/TF pairs, since
                                # incompatible ones were skipped above).
                                if metric_config.get("animate"):
                                    anim_dir = (
                                        model_path
                                        / "explanations"
                                        / f"explanations_{explainer}"
                                    )
                                    anim_dir.mkdir(parents=True, exist_ok=True)
                                    metric_slug = metric_config["metric"].replace(
                                        " ", "_"
                                    )
                                    _animate(
                                        model=model_wrapper,
                                        metric_conf=metric_config,
                                        explanation=explanation,
                                        data=exp_data,
                                        labels=test_labels[explanation.indices],
                                        save_path=anim_dir
                                        / f"animation_{metric_slug}.gif",
                                    )
                        finally:
                            del explanation
                            gc.collect()
                finally:
                    del model_wrapper
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif device.type == "mps":
                        torch.mps.empty_cache()
                    gc.collect()
        except Exception:
            logger.exception(
                "Dataset '%s' failed, skipping.", dataset.get("dataset", "unknown")
            )
        finally:
            del ds, splits, encoder, train_data, train_labels, test_data, test_labels
            gc.collect()

    # Save results
    if metric_rows:
        df = pd.DataFrame(metric_rows)

        # Global — all results
        global_path = Path(config["results_rel_path"]) / "metrics.csv"
        df.to_csv(global_path, index=False)
        logger.info("Global metrics saved to %s", global_path)

        # Per dataset — derive path from "dataset" column
        for dataset_name, group in df.groupby("dataset"):
            path = Path(config["results_rel_path"]) / dataset_name / "metrics.csv"
            group.to_csv(path, index=False)
            logger.info("Dataset metrics saved to %s", path)

        # Per model — derive path from "dataset" + "model" columns
        for (dataset_name, model_name), group in df.groupby(["dataset", "model"]):
            path = (
                Path(config["results_rel_path"])
                / dataset_name
                / model_name
                / "metrics.csv"
            )
            group.to_csv(path, index=False)
            logger.info("Model metrics saved to %s", path)

    logger.info("\n--- Experiment Finished ---")


def initial_setup(config_path: str, debug: bool = False) -> dict:
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
    dict
        The extracted configuration.
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
    _add_file_logging(debug, log_path=path / "experiment.log")

    return config


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
    main(args.conf, debug=args.debug)
