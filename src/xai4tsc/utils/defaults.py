"""Default hyperparameters and metric-constructor defaults used across the framework."""

t_con_defaults = {
    "epochs": 20,
    "batchsize": 8,
    "learn_rate": 1e-3,
    "patience": 3,
    "save_best": True,
    "loss_func": "crossentropy",
    "optimizer": "adam",
}

# Params whose values (float in 0.0-1.0) represent a fraction of series length T
# and are converted to integers at runtime: max(1, int(fraction * T)).
# Keys = Quantus class name; values = set of param names that accept T-fractions.
# Named _T to distinguish from future _fraction_params_C (channel-relative params).
_fraction_params_T: dict[str, set[str]] = {  # noqa: N816 — _T denotes time axis
    "FaithfulnessCorrelation": {"subset_size"},
    "TopKIntersection": {"k"},
}

# Per-metric constructor defaults for QuantusEvaluator.
# Keys are Quantus class names; values map param names to scalars or callables.
# Callables receive the x_batch array (shape B, C, T) and return a scalar.
# Float values for params listed in _fraction_params_T are treated as fractions
# of series length T and converted to integers at runtime.
eval_metric_defaults = {
    "FaithfulnessCorrelation": {
        # 5 % of series length; Quantus default (224) is ImageNet-sized
        "subset_size": 0.05,
    },
    "TopKIntersection": {
        # 10 % of series length; Quantus default (1000) exceeds most TSC series
        "k": 0.1,
    },
}
