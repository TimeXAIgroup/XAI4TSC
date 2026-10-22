import torch

import xai4tsc
from xai4tsc.data import class_names, load_dataset
from xai4tsc.evaluation.evaluate import evaluate
from xai4tsc.xai.explain import generate_explanation


def main(out_dir: str, cache_dir: str | None = None) -> dict:
    """
    Run an example to showcase package capabilities.

    Parameters
    ----------
    out_dir : str
        Directory to save the output to.
    cache_dir : str, optional
        Where to keep the downloaded UCR archive, so repeated runs do not
        re-download it.  ``None`` leaves the choice to sktime.

    Returns
    -------
    dict
        Test-set metrics from :meth:`~xai4tsc.ModelBase.evaluate_model`.
    """
    # Enable package logging to the console and to <out_dir>/xai4tsc.log
    xai4tsc.enable_logging(out_dir)

    # Set the global torch seed
    seed = 0xDEADBEEF
    torch.manual_seed(seed)

    # Load data (UCR download or local numpy files) and split it 80/10/10.
    # The returned list is ordered [train, test, val] - validation comes last.
    ds = load_dataset("GunPoint", cache_dir=cache_dir)
    splits, encoder = ds.split(train_split=0.8, val_split=0.1, random_state=seed)
    ds.save_splits(out_dir)
    train_data, train_labels, _ = splits[0]
    test_data, test_labels, _ = splits[1]
    val_data, val_labels, _ = splits[2]

    # Build a model using its constructor and configure() to set where it runs
    # and saves results. Available models are listed in xai4tsc.models.MODELS.
    model = xai4tsc.FCN(in_channels=1, num_classes=2).configure(
        device="cpu",
        save_path=out_dir,  # best checkpoint + training plots land here
    )

    # Train the model. Passing the validation split makes early stopping and
    # best-checkpoint selection monitor validation loss instead of training loss.
    model.train_model(
        train_data,
        train_labels,
        epochs=50,
        batchsize=32,
        learn_rate=0.001,
        patience=3,
        data_val=val_data,
        labels_val=val_labels,
    )

    # Evaluate on test data. Passing the class names labels the plots and the
    # per-class results file by class rather than by index.
    results = model.evaluate_model(
        test_data, test_labels, batchsize=32, class_names=class_names(encoder)
    )

    # Generate explanations, available explainers are listed in xai4tsc.xai.EXPLAINERS
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
    complexity = evaluate(
        model=model,
        metric="Complexity",
        explanation=exp,
        data=test_data[exp.indices],
        labels=test_labels[exp.indices],
        metric_class_params={"normalise": True, "abs": True, "disable_warnings": True},
        device="cpu",
    )
    print(f"Test accuracy: {results['accuracy']}")
    print(f"Complexity of the explanations: {complexity}")

    return results


if __name__ == "__main__":
    main("experiments/results/getting_started")
