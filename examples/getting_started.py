import xai4tsc
from xai4tsc.data import load_dataset
from xai4tsc.evaluation.evaluate import evaluate
from xai4tsc.models.models import load_model
from xai4tsc.xai.explain import generate_explanation


def main(out_dir: str) -> None:
    """
    Run an example to showcase package capabilities.

    Parameters
    ----------
    out_dir : str
        Directory to safe the output to.
    """
    # Enable package logging to the console and to <out_dir>/xai4tsc.log
    xai4tsc.enable_logging(out_dir)

    # Load data (UCR download or local numpy files)
    ds = load_dataset("GunPoint", use_predefined_splits=True)
    splits, encoder = ds.split(train_split=0.8, val_split=0.1, random_state=42)
    ds.save_splits(out_dir)
    train_data, train_labels, _ = splits[0]
    test_data, test_labels, _ = splits[1]

    # Load a model
    model = load_model(
        {"model": "FCN", "init_params": {"in_channels": 1, "num_classes": 2}},
        device="cpu",
    )

    # Set up hyperparameters
    hyperparams = {
        "epochs": 10,
        "batchsize": 32,
        "loss_func": "CrossEntropy",
        "optimizer": "adam",
        "learn_rate": 0.001,
        "patience": 3,
    }

    # Train the model
    model.train_model(
        train_data,
        train_labels,
        hyperparams,
        save_path=out_dir,  # best checkpoint + training plots land here
    )

    # Evaluate on test data
    model.evaluate_model(
        test_data,
        test_labels,
        hyperparams,
        save_path=out_dir,  # save model performance data
    )

    # Generate explanations
    exp = generate_explanation(
        method="integrated_gradients",
        model=model,
        data=test_data,
        labels=test_labels,
        encoder=encoder,
        indices=[0, 1, 2],
        device="cpu",
    )

    # Evaluate the explanations
    _ = evaluate(
        model=model,
        metric="Complexity",
        explanation=exp,
        data=test_data[exp.indices],
        labels=test_labels[exp.indices],
        metric_class_params={"normalise": True, "abs": True, "disable_warnings": True},
        device="cpu",
    )


if __name__ == "__main__":
    main("experiments/results/getting_started")
