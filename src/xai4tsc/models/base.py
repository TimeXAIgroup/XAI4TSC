"""``ModelBase`` ABC: training, prediction, evaluation, checkpoint persistence."""

import json
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812 — PyTorch ecosystem convention
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    multilabel_confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from ..utils.utils import _filename_slug

logger = logging.getLogger(__name__)


def _target_tensor(labels: np.ndarray) -> torch.Tensor:
    """
    Return *labels* as the tensor their loss expects, decided by their rank.

    The rank is the formulation, which the label encoder already settled (see
    :func:`~xai4tsc.data.base._encoded_targets`): 1-D is one class index per
    sample, 2-D is one column per model output. The dtype follows from it, and
    torch is strict about both - :class:`~torch.nn.CrossEntropyLoss` takes
    ``int64`` indices and rejects ``int32`` and ``float32``, while
    :class:`~torch.nn.BCEWithLogitsLoss` needs float targets and rejects
    ``int64``.

    Parameters
    ----------
    labels : np.ndarray
        Encoded labels, 1-D class indices or a 2-D indicator array.

    Returns
    -------
    torch.Tensor
        ``int64`` for 1-D labels, ``float32`` for 2-D.
    """
    arr = np.asarray(labels)
    dtype = np.float32 if arr.ndim > 1 else np.int64
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=dtype))


def _batch_accuracy_counts(
    outputs: torch.Tensor, targets: torch.Tensor
) -> tuple[int, int]:
    """
    Return ``(correct, total)`` for one batch, by the rank of *targets*.

    1-D targets are class indices, so this is argmax accuracy. For 2-D targets
    argmax is meaningless, and comparing it against the rows is worse than
    useless: it raises when the batch size differs from the column count and
    silently reports above 100% when they happen to match. Those are scored per
    output instead, which for a multi-label head is ``1 - Hamming loss``.

    Parameters
    ----------
    outputs : torch.Tensor
        Raw logits of shape ``(batch, n_outputs)``.
    targets : torch.Tensor
        The batch's targets, 1-D indices or a 2-D indicator array.

    Returns
    -------
    tuple[int, int]
        Correct predictions and the number compared.
    """
    if targets.ndim > 1:
        preds = (outputs > 0).to(targets.dtype)
        return int((preds == targets).sum().item()), int(targets.numel())
    preds = torch.max(outputs.data, 1)[1]
    return int((preds == targets).sum().item()), int(targets.size(0))


def _json_safe(value: object) -> bool:
    """
    Return whether *value* can be stored in a checkpoint config file.

    Parameters
    ----------
    value : object
        A single constructor argument.

    Returns
    -------
    bool
        ``True`` when :mod:`json` can encode it.
    """
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


OUTPUT_ACTIVATIONS = ("softmax", "sigmoid")
"""
The output activations :func:`predict_probs` knows how to apply.

``"softmax"`` is the single-label default: the outputs compete and a row sums to
1. ``"sigmoid"`` is the multi-label one: each output is an independent
probability and a row does not sum to anything in particular.
"""


def check_output_activation(name: str) -> str:
    """
    Validate an output-activation name, returning it unchanged.

    Called from a model's constructor rather than from
    :func:`predict_probs`, so a typo fails when the model is built instead of
    silently changing every probability the model ever reports.

    Parameters
    ----------
    name : str
        A member of :data:`OUTPUT_ACTIVATIONS`.

    Returns
    -------
    str
        *name*.

    Raises
    ------
    ValueError
        If *name* is not a known activation.
    """
    if name not in OUTPUT_ACTIVATIONS:
        raise ValueError(
            f"output_activation must be one of {list(OUTPUT_ACTIVATIONS)}, got "
            f"{name!r}. 'softmax' is for one-of-K outputs, 'sigmoid' for "
            "independent per-label outputs trained with BCEWithLogitsLoss."
        )
    return name


def predict_probs(model: nn.Module, data: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Return class probabilities for *data* from any :class:`~torch.nn.Module`.

    Models here emit raw logits from ``forward()`` by contract, so the activation
    is applied on the way out. Which one comes from the model itself -
    ``model.output_activation``, defaulting to ``"softmax"`` when a module does
    not declare one, which keeps the "any ``nn.Module``" contract intact.

    That default is why the attribute belongs on the model's constructor: it is
    persisted in the checkpoint's ``init_params``, so a reloaded multi-label
    model still reports independent per-label probabilities. Were it a training
    argument instead, nothing in the checkpoint would record it and a softmax
    would be applied to sigmoid outputs with no sign that anything was wrong.

    The model's train/eval mode is left untouched and belongs to the caller.

    Parameters
    ----------
    model : torch.nn.Module
        Model to run. Must return logits of shape ``(n_samples, n_outputs)``.
    data : np.ndarray or torch.Tensor
        Input samples, shape ``(n_samples, n_channels, n_timesteps)``.

    Returns
    -------
    np.ndarray
        Probabilities of shape ``(n_samples, n_outputs)``. Rows sum to 1 under
        ``"softmax"``; under ``"sigmoid"`` each column is independent.
    """
    tensor = torch.as_tensor(data, dtype=torch.float32)
    param = next(model.parameters(), None)
    if param is not None:
        tensor = tensor.to(param.device)
    activation = getattr(model, "output_activation", "softmax")
    with torch.no_grad():
        logits = model(tensor)
        probs = (
            torch.sigmoid(logits)
            if check_output_activation(activation) == "sigmoid"
            else F.softmax(logits, dim=1)
        )
        return probs.detach().cpu().numpy()


def _confusion_counts(cm: np.ndarray) -> list[dict[str, int]]:
    """
    Derive the one-vs-rest confusion counts of every class from a full matrix.

    Parameters
    ----------
    cm : np.ndarray
        A ``(K, K)`` confusion matrix over *all* model classes, rows being the
        true class and columns the predicted one. Built over only the classes
        present in the data, the rows would no longer be indexed by class.

    Returns
    -------
    list of dict
        One entry per class, in class-index order, with ``samples``, ``tp``,
        ``tn``, ``fp`` and ``fn``.
    """
    cm = np.asarray(cm)
    total = int(cm.sum())
    counts = []
    for k in range(cm.shape[0]):
        tp = int(cm[k, k])
        samples = int(cm[k, :].sum())
        fp = int(cm[:, k].sum()) - tp
        fn = samples - tp
        counts.append(
            {
                "samples": samples,
                "tp": tp,
                "tn": total - tp - fp - fn,
                "fp": fp,
                "fn": fn,
            }
        )
    return counts


def _rates(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    """
    Derive the usual rates from one set of confusion counts.

    The rates are fractions in ``[0, 1]`` (``mcc`` in ``[-1, 1]``); one whose
    denominator is zero - a class with no true samples, or none predicted - is
    ``nan`` rather than an error, so a degenerate class does not abort the
    evaluation.

    ``accuracy`` is the one-vs-rest accuracy of this class alone, which is the
    only reading available per class: a class is either right or wrong for every
    sample, including the ones that belong to another class. For a binary
    problem class 1's is the overall accuracy, and for a multi-label output it is
    that output's own, which a subset accuracy hides. It is derived here rather
    than at the two call sites so the single-label and the per-output tables
    cannot drift apart on it.

    Parameters
    ----------
    tp, tn, fp, fn : int
        True positives, true negatives, false positives and false negatives.

    Returns
    -------
    dict
        ``accuracy``, ``sensitivity``, ``specificity``, ``ppv``, ``npv``, ``f1``
        and ``mcc``.
    """
    nan = float("nan")
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else nan
    # The geometric form of Matthews' correlation coefficient, which stays
    # computable from the rates alone (no large intermediate products).
    mcc = math.sqrt(sensitivity * specificity * ppv * npv) - math.sqrt(
        (1 - sensitivity) * (1 - specificity) * (1 - npv) * (1 - ppv)
    )
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "mcc": mcc,
    }


def _macro(rates: list[dict[str, float]], key: str) -> float:
    """
    Average one rate over the classes that define it.

    Parameters
    ----------
    rates : list of dict
        Per-class rates as returned by :func:`_rates`.
    key : str
        The rate to average.

    Returns
    -------
    float
        The unweighted mean over the classes where the rate is defined, or
        ``nan`` when it is defined for none - a class with no samples and no
        predictions must not drag the average to ``nan`` on its own.
    """
    values = [r[key] for r in rates if not math.isnan(r[key])]
    return sum(values) / len(values) if values else float("nan")


def _macro_auc(class_aucs: list[float], present: np.ndarray) -> float:
    """
    Average the one-vs-rest AUCs of the classes present in the data.

    This is the macro-average multiclass AUC, equal to what
    :func:`sklearn.metrics.roc_auc_score` computes with ``multi_class="ovr"``
    when every class is present. It is derived here instead so that a class
    missing from the test split yields a *partial* score, where that call
    raises: it requires the scores it is given to sum to 1 across their columns,
    which slicing them down to the present classes breaks.

    Parameters
    ----------
    class_aucs : list of float
        One AUC per class index, ``nan`` where undefined.
    present : np.ndarray
        The class indices present in the data.

    Returns
    -------
    float
        The unweighted mean over the present classes, or ``nan`` when none of
        them has a defined AUC.
    """
    values = [class_aucs[int(c)] for c in present if not math.isnan(class_aucs[int(c)])]
    return sum(values) / len(values) if values else float("nan")


def _display_names(names: Sequence[str] | None, n_classes: int) -> list[str]:
    """
    Resolve the display name of every class, falling back to the class index.

    Parameters
    ----------
    names : sequence of str or None
        One name per class index, e.g. from
        :func:`~xai4tsc.data.base.class_names`. ``None`` names classes by index.
    n_classes : int
        How many classes the model distinguishes.

    Returns
    -------
    list of str
        Exactly *n_classes* names. A list of the wrong length is a sign the
        names came from a different dataset, so it is reported and ignored
        rather than silently mislabelling the results.
    """
    if names is None:
        return [str(k) for k in range(n_classes)]
    resolved = [str(name) for name in names]
    if len(resolved) != n_classes:
        logger.warning(
            "Ignoring %d class names for a %d-class model; naming classes by "
            "index instead.",
            len(resolved),
            n_classes,
        )
        return [str(k) for k in range(n_classes)]
    return resolved


class ModelBase(nn.Module, ABC):
    """
    Base class for all time series classification models.

    Subclass this to add a custom model to xai4tsc.  Register the subclass
    with :func:`xai4tsc.register_model` to make it available by name.

    Subclasses must implement :meth:`forward`.  All training, prediction,
    evaluation, and persistence logic is provided here and inherited.

    :attr:`name` defaults to the subclass name, so a model constructed directly
    still names its own files.  :attr:`device` and :attr:`save_path` describe
    the machine and the run rather than the model, so they are not constructor
    arguments; set them with :meth:`configure`, which
    :func:`~xai4tsc.models.models.load_model` calls on the caller's behalf.
    Until then :attr:`save_path` is ``None`` and the methods that write refuse
    to guess a destination.

    A subclass records its own constructor arguments by calling
    :meth:`_record_init_params` with ``locals()`` as its first statement, which
    is what lets a checkpoint describe how to rebuild the model.

    Example::

        class MyCNN(ModelBase):
            def __init__(self, in_channels: int, num_classes: int):
                super().__init__()
                self._record_init_params(locals())
                self.conv = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
                self.fc = nn.Linear(32, num_classes)

            def forward(self, x):
                return self.fc(self.conv(x).mean(-1))

        xai4tsc.register_model("my_cnn", MyCNN)
    """

    name: str
    """Identifier used in file names and logs.  Defaults to the class name."""
    device: torch.device | None = None
    """Compute device the model lives on."""
    save_path: Path | None = None
    """Directory for checkpoints and diagnostic plots.  Always absolute: it is
    resolved when set, so it does not move with the working directory."""
    model_path: Path | None = None
    """Path to the best saved checkpoint (set after training)."""
    best_epoch: int | None = None
    """Total epochs of training the best checkpoint represents (``None`` if unknown)."""
    _init_params: dict | None = None
    """Constructor kwargs stored for checkpoint reloading."""
    _loaded_from: Path | None = None
    """Checkpoint this model's weights were loaded from, if any."""

    def __init__(self) -> None:
        super().__init__()
        self.name = type(self).__name__

    def _record_init_params(self, init_locals: dict) -> None:
        """
        Record the constructor arguments this model was built with.

        What is recorded is what the model was actually built with, defaults
        included, so :meth:`save_model` can write a checkpoint that rebuilds
        the same architecture even if a default changes in a later version.

        Parameters
        ----------
        init_locals : dict
            ``locals()`` from the top of the subclass ``__init__``.  ``self``
            and the ``__class__`` cell that a zero-argument ``super()`` call
            puts in scope are dropped; any ``kwargs`` mapping is flattened into
            the result.
        """
        params = {
            key: value
            for key, value in init_locals.items()
            if key != "self" and not key.startswith("__")
        }
        # A subclass taking **kwargs would otherwise record the mapping itself
        # under the key "kwargs", which no constructor accepts back.
        extra = params.pop("kwargs", None)
        if isinstance(extra, dict):
            params.update(extra)
        self._init_params = params

    def configure(
        self,
        device: str | torch.device | None = None,
        save_path: Path | str | None = None,
        name: str | None = None,
    ) -> "ModelBase":
        """
        Set where this model runs and where it writes, and return it.

        These are properties of the machine and the run, not of the
        architecture, which is why they are set here rather than passed to the
        constructor.  :func:`~xai4tsc.models.models.load_model` calls this after
        building a model; call it yourself when constructing one directly::

            model = FCN(in_channels=1, num_classes=2).configure(
                device="cpu", save_path="results/run_1"
            )

        Parameters
        ----------
        device : str or torch.device, optional
            Compute device.  The model is moved onto it.
        save_path : Path or str, optional
            Directory for checkpoints and diagnostic plots.  Resolved to an
            absolute path.
        name : str, optional
            Identifier used in file names and logs.  Overrides the default
            class name, which is how one class registered under two keys writes
            distinguishable files.

        Returns
        -------
        ModelBase
            ``self``, so the call can be chained onto the constructor.

        Notes
        -----
        An omitted argument leaves the current value alone, so a later call
        that sets only one attribute does not clear the others.
        """
        if name is not None:
            self.name = name
        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
            self.to(self.device)
        if save_path is not None:
            self.save_path = Path(save_path).resolve()
        return self

    def _resolve_save_path(self, save_path: Path | str | None) -> Path:
        """
        Resolve the supplied path, if none return self.save_path resolved.

        The result is run through :meth:`pathlib.Path.resolve`, so a relative
        destination does not move with the working directory and every path this
        class logs is absolute and pasteable.

        Parameters
        ----------
        save_path : str or Path or None
            The destination passed to this call, if any.  Falls back to
            :attr:`save_path`.

        Returns
        -------
        Path
            The absolute directory. :attr:`save_path` is left alone.

        Raises
        ------
        ValueError
            If neither *save_path* nor :attr:`save_path` is set.  Defaulting to
            the working directory instead would scatter checkpoints and plots
            wherever the caller happened to be.
        """
        if save_path is None and self.save_path is None:
            raise ValueError(
                "No save_path set, so there is nowhere to write: pass "
                "save_path=... to this call, or set one with "
                "load_model(..., save_path=...) or configure(save_path=...)."
            )
        return Path(save_path if save_path is not None else self.save_path).resolve()

    # ── Abstract ────────────────────────────────────────────────────────────

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, T)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """

    # ── Inference ───────────────────────────────────────────────────────────

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict(self, data: np.ndarray, labels: np.ndarray = None) -> tuple:
        """
        Run inference on a numpy array.

        Parameters
        ----------
        data : np.ndarray
            Shape ``(n_samples, n_channels, n_timesteps)``.
        labels : np.ndarray, optional
            Ground-truth labels for accuracy logging.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(out_classes, out_probs)`` — both numpy arrays.
        """
        self.eval()
        # Models emit raw logits; predict_probs softmaxes them so the returned array
        # is a true probability distribution (argmax is unchanged by softmax).
        out_probs = predict_probs(self, data)
        out_classes = np.argmax(out_probs, axis=1)
        if labels is not None:
            accuracy = sum(out_classes == labels) / len(labels)
            logger.info("Prediction accuracy: %s", accuracy)
        return out_classes, out_probs

    # ── Training ─────────────────────────────────────────────────────────────

    def train_model(
        self,
        data_train: np.ndarray,
        labels_train: np.ndarray,
        epochs: int = 20,
        batchsize: int = 8,
        learn_rate: float = 1e-3,
        patience: int = 3,
        min_delta: float = 5e-3,
        save_best: bool = True,
        loss_func: nn.Module | None = None,
        optimizer: Callable[..., Optimizer] | Optimizer | None = None,
        data_val: np.ndarray | None = None,
        labels_val: np.ndarray | None = None,
        save_path: Path | str | None = None,
        start_epoch: int | None = None,
    ) -> "ModelBase":
        """
        Run the full training loop with early stopping.

        Early stopping and best-checkpoint selection monitor **validation** loss
        when *data_val*/*labels_val* are supplied, and fall back to **training**
        loss otherwise.  Monitoring validation loss is the correct way to avoid
        selecting an overfit model.

        Parameters
        ----------
        data_train : np.ndarray
            Training data of shape ``(n_samples, n_channels, n_timesteps)``.
        labels_train : np.ndarray
            Integer class labels.
        epochs : int
            Number of epochs to run in this call, 20 by default.  A run resuming
            from a checkpoint continues its epoch count.
        batchsize : int
            Samples per batch, 8 by default.
        learn_rate : float
            Optimizer learning rate, ``1e-3`` by default.
        patience : int
            Epochs without improvement before early stopping, 3 by default.
        min_delta : float
            How far the monitored loss has to fall below its best value to count
            as an improvement, ``5e-3`` by default. An absolute difference, not a
            fraction of the loss.
        save_best : bool
            Whether to write a checkpoint whenever the monitored loss improves,
            ``True`` by default.
        loss_func : torch.nn.Module, optional
            The loss, as an object: any module whose call is
            ``(logits, targets) -> scalar tensor``, with its own arguments
            already applied (``nn.CrossEntropyLoss(weight=w)``,
            ``nn.BCEWithLogitsLoss(pos_weight=w)``, or a custom module). It is
            moved to this model's device, so a weight built on the CPU travels
            with it. ``None`` (default) uses
            :class:`~torch.nn.CrossEntropyLoss`. Losses needing more than the
            final logits - contrastive, triplet, distillation - are out of scope:
            the contract is the two-argument call above.
        optimizer : callable or torch.optim.Optimizer, optional
            The optimizer, in one of three forms. A **class**
            (``torch.optim.AdamW``) or any **callable** returning an optimizer (a
            :func:`functools.partial` carrying constructor arguments) is called
            as ``optimizer(self.parameters(), lr=learn_rate)``. An **instance**
            built over specific parameter groups is used as given, and
            *learn_rate* then does not apply, which is reported rather than
            silently ignored. ``None`` (default) uses
            :class:`~torch.optim.Adam`.
        data_val : np.ndarray, optional
            Validation data of shape ``(n_samples, n_channels, n_timesteps)``.
            When provided, validation loss drives early stopping and
            checkpointing; when omitted, both monitor **training** loss instead.
            Must be given together with *labels_val*: one without the other raises,
            rather than falling back to training loss as if nothing had been asked for.
        labels_val : np.ndarray, optional
            Integer class labels for *data_val*.  Required when *data_val* is
            given.
        save_path : str or Path, optional
            Directory for the best checkpoint and training plots.  Takes priority
            over the destination given to
            :func:`~xai4tsc.models.models.load_model`, and falls back to it.
            Raises if neither is set.
        start_epoch : int, optional
            Epochs of training the weights have already had, used to continue
            epoch numbering when resuming from a checkpoint.  Defaults to
            :attr:`best_epoch` (set by
            :func:`~xai4tsc.models.models.load_model`) and otherwise to ``0``.
            ``epochs`` is always the number of epochs to run **in this call**.

        Returns
        -------
        ModelBase
            ``self`` after training (best checkpoint restored).
        """
        self.save_path = self._resolve_save_path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        data_tensor = torch.from_numpy(
            np.ascontiguousarray(data_train, dtype=np.float32)
        )
        labels_tensor = _target_tensor(labels_train)

        start_epoch = start_epoch if start_epoch is not None else (self.best_epoch or 0)
        final_epoch = start_epoch + epochs

        loss_function = (nn.CrossEntropyLoss() if loss_func is None else loss_func).to(
            self.device
        )
        optimizer_instance = self._resolve_optimizer(optimizer, learn_rate)

        loader = DataLoader(
            TensorDataset(data_tensor, labels_tensor),
            batch_size=batchsize,
        )

        val_loader = self._validation_loader(data_val, labels_val, batchsize, epochs)
        self._check_loss(
            loss_function, data_tensor[:batchsize], labels_tensor[:batchsize]
        )

        acc_name = "bit_acc" if labels_tensor.ndim > 1 else "acc"
        best_loss = 1e6
        best_path = None
        cur_patience = 0
        epoch_losses = []
        epoch_accs = []
        epoch = start_epoch - 1

        for offset in range(epochs):
            epoch = start_epoch + offset  # 0-based global epoch index
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer_instance.zero_grad()
                outputs = self(x_batch)
                loss = loss_function(outputs, y_batch)
                loss.backward()
                optimizer_instance.step()

                running_loss += loss.item()
                batch_correct, batch_total = _batch_accuracy_counts(outputs, y_batch)
                correct += batch_correct
                total += batch_total

            epoch_loss = running_loss / len(loader)
            epoch_acc = correct / total
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)

            # Monitor validation loss when available, else training loss.
            val_loss = self._validation_loss(val_loader, loss_function)
            monitor_loss = val_loss if val_loss is not None else epoch_loss
            if val_loss is not None:
                logger.info(
                    "Epoch %2d/%d  loss: %.4f  %s: %.4f  val_loss: %.4f",
                    epoch + 1,
                    final_epoch,
                    epoch_loss,
                    acc_name,
                    epoch_acc,
                    val_loss,
                )
            else:
                logger.info(
                    "Epoch %2d/%d  loss: %.4f  %s: %.4f",
                    epoch + 1,
                    final_epoch,
                    epoch_loss,
                    acc_name,
                    epoch_acc,
                )

            if monitor_loss < (best_loss - min_delta):
                cur_patience = 0
                # Tracked whatever save_best says: patience counts from it.
                best_loss = monitor_loss
                if save_best:
                    if best_path is not None:
                        best_path.unlink()
                        # The superseded checkpoint's config goes with it,
                        # otherwise stale configs accumulate in the directory.
                        self.config_path_for(best_path).unlink(missing_ok=True)
                    best_path = self.save_path / f"{self.name}_epoch_{epoch + 1}.pt"
                    self.best_epoch = epoch + 1
                    self.save_model(best_path)
            else:
                cur_patience += 1
                if cur_patience == patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d, min_delta=%.4g)",
                        epoch + 1,
                        patience,
                        min_delta,
                    )
                    break

        self._plot_training_curve(self.save_path, epoch_losses, epoch_accs, start_epoch)

        if best_path is None:
            # No epoch improved on the initial loss, so nothing was checkpointed
            # along the way; persist the final weights under the last epoch run.
            self.best_epoch = epoch + 1
            best_path = self.save_path / f"{self.name}_epoch_{self.best_epoch}.pt"
            self.save_model(best_path)
        else:
            # Restore best checkpoint in-place
            self.load_state_dict(
                torch.load(best_path, weights_only=True, map_location=self.device)
            )
            self.eval()

        self.model_path = best_path
        return self

    _INDEX_TARGET_LOSSES = (nn.CrossEntropyLoss, nn.NLLLoss)
    """Losses that read the target as class indices rather than per-output rows."""

    def _resolve_optimizer(
        self,
        optimizer: Callable[..., Optimizer] | Optimizer | None,
        learn_rate: float,
    ) -> Optimizer:
        """
        Return the optimizer to train with, from whichever form was supplied.

        An optimizer cannot be built before the model exists, so unlike the loss
        it is accepted as a class or factory and constructed here over
        ``self.parameters()``. An already-built instance is honoured as well, for
        a caller who set up parameter groups.

        Parameters
        ----------
        optimizer : callable or Optimizer or None
            A class, a factory, an instance, or ``None`` for
            :class:`~torch.optim.Adam`.
        learn_rate : float
            Learning rate, applied only when this method does the constructing.

        Returns
        -------
        Optimizer
            The optimizer to step.
        """
        if optimizer is None:
            return optim.Adam(self.parameters(), lr=learn_rate)
        if isinstance(optimizer, Optimizer):
            found = sorted({group["lr"] for group in optimizer.param_groups})
            logger.warning(
                "Using the %s instance that was passed in, so learn_rate=%g does "
                "not apply; its parameter groups carry lr=%s.",
                type(optimizer).__name__,
                learn_rate,
                found,
            )
            return optimizer
        return optimizer(self.parameters(), lr=learn_rate)

    def _check_loss(
        self, loss_function: nn.Module, data: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """
        Check the loss against one real batch before the first epoch runs.

        Two layers, because torch will not do this for us. The specific one:
        :class:`~torch.nn.CrossEntropyLoss` **accepts** a 2-D float target
        without complaint, reading each row as soft class probabilities over a
        competitive softmax - so two co-occurring labels can each reach at most
        0.5, the curve looks healthy and the objective is wrong. The universal one:
        everything else: a shape mismatch, a dtype mismatch, a ``weight`` of the
        wrong length - surfaces as a startup error naming both sides instead of a
        traceback from inside the epoch loop.

        What neither layer catches: a custom loss that silently misreads its
        targets. The contract is ``(logits, targets) -> scalar tensor``.

        The check runs in eval mode under :func:`torch.no_grad`, and it is handed
        the tensors directly rather than a loader, because building a
        :class:`~torch.utils.data.DataLoader` iterator draws from the global
        torch generator to seed itself - which would shift every later weight
        initialisation in the process. As written it moves no batch-norm
        statistics and draws no randomness, so a trained model is bit-identical
        with and without it.

        Parameters
        ----------
        loss_function : torch.nn.Module
            The loss about to be used, already on this model's device.
        data : torch.Tensor
            One batch of inputs.
        targets : torch.Tensor
            That batch's targets.

        Raises
        ------
        ValueError
            If the loss cannot score this model's output, or reads the targets as
            something other than what they are.
        """
        if len(data) == 0:
            return
        x_batch, y_batch = data, targets
        if y_batch.ndim > 1 and isinstance(loss_function, self._INDEX_TARGET_LOSSES):
            raise ValueError(
                f"{type(loss_function).__name__} reads the target as class "
                f"indices, but the targets are {tuple(y_batch.shape)} - one column "
                "per model output. It would not raise: it reads each row as soft "
                "class probabilities over a competitive softmax and optimises a "
                "different objective. Use nn.BCEWithLogitsLoss (loss_func: "
                '"bcewithlogits") for per-output targets, or encode the labels to '
                "class indices."
            )
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                outputs = self(x_batch.to(self.device))
                try:
                    loss_function(outputs, y_batch.to(self.device))
                except Exception as exc:
                    raise ValueError(
                        f"{type(loss_function).__name__} cannot score this model's "
                        f"output: logits {tuple(outputs.shape)} {outputs.dtype} "
                        f"against targets {tuple(y_batch.shape)} {y_batch.dtype} "
                        f"({exc}). Check the loss against the label encoding and "
                        "the model's output width."
                    ) from exc
        finally:
            self.train(was_training)

    def _validation_loader(
        self,
        data_val: np.ndarray | None,
        labels_val: np.ndarray | None,
        batchsize: int,
        epochs: int,
    ) -> DataLoader | None:
        """
        Build the validation loader, or report why there will not be one.

        Early stopping and best-checkpoint selection monitor whatever this
        returns a loss for, so a validation set that goes missing here silently
        changes which checkpoint is kept. Half of a pair, or an empty one, is a
        mistake with nothing sensible to fall back to and raises; supplying
        neither is a documented choice and is logged, so the fallback leaves a
        trace in the log rather than only an absence.

        Parameters
        ----------
        data_val : np.ndarray or None
            Validation data, if any.
        labels_val : np.ndarray or None
            Validation labels, if any.
        batchsize : int
            Samples per batch, matching the training loader.
        epochs : int
            Epochs this call will run, named in the no-validation message.

        Returns
        -------
        DataLoader or None
            ``None`` when no validation data was supplied at all.

        Raises
        ------
        ValueError
            If exactly one of *data_val* / *labels_val* is given, or if either
            is empty.
        """
        if data_val is None and labels_val is None:
            logger.info(
                "No validation data: early stopping and best-checkpoint "
                "selection monitor TRAINING loss over %d epoch(s). Pass "
                "data_val and labels_val to select on validation loss instead.",
                epochs,
            )
            return None
        if data_val is None or labels_val is None:
            missing, given = (
                ("data_val", "labels_val")
                if data_val is None
                else ("labels_val", "data_val")
            )
            raise ValueError(
                f"train_model got {given} without {missing}. Validation needs "
                f"both; with only one, early stopping would silently fall back "
                f"to training loss. Pass {missing} as well, or neither."
            )
        if len(data_val) == 0 or len(labels_val) == 0:
            raise ValueError(
                f"train_model got an empty validation set "
                f"(data_val: {len(data_val)}, labels_val: {len(labels_val)}). "
                "There is nothing to validate on; pass neither argument to "
                "monitor training loss deliberately."
            )
        return DataLoader(
            TensorDataset(
                torch.from_numpy(np.ascontiguousarray(data_val, dtype=np.float32)),
                _target_tensor(labels_val),
            ),
            batch_size=batchsize,
        )

    def _validation_loss(
        self,
        val_loader: DataLoader | None,
        loss_func: nn.Module,
    ) -> float | None:
        """Return mean loss over *val_loader*, or ``None`` when no loader given."""
        if val_loader is None:
            return None
        self.eval()
        total = 0.0
        n_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                total += loss_func(self(x_batch), y_batch).item()
                n_batches += 1
        return total / n_batches if n_batches else None

    # ── Persistence ──────────────────────────────────────────────────────────

    @staticmethod
    def config_path_for(checkpoint_path: Path | str) -> Path:
        """
        Return the config file path belonging to *checkpoint_path*.

        The config file is named after its checkpoint with ``_config.json`` in
        place of the suffix, e.g. ``FCN_epoch_10.pt`` ->
        ``FCN_epoch_10_config.json``, so the two are unambiguously paired.

        Parameters
        ----------
        checkpoint_path : Path or str
            Path to a ``.pt`` checkpoint.

        Returns
        -------
        Path
            Path to the matching config file, in the same directory.
        """
        checkpoint_path = Path(checkpoint_path)
        return checkpoint_path.with_name(f"{checkpoint_path.stem}_config.json")

    def _save_checkpoint_config(self, checkpoint_path: Path, epoch: int | None) -> None:
        """
        Write the config file describing how to rebuild *checkpoint_path*.

        The checkpoints this class writes are bare state dicts, which carry no
        record of the class or the constructor arguments they belong to.  This
        file supplies both, so :func:`~xai4tsc.models.models.load_model` can
        reload a checkpoint given only its path.  Every built-in records its
        arguments in its constructor.

        Parameters
        ----------
        checkpoint_path : Path
            The checkpoint this config describes.
        epoch : int or None
            Total epochs of training the checkpoint represents; ``None`` when
            that is unknown, which is recorded rather than guessed as ``0``.
        """
        if not self._init_params:
            return
        cfg = {
            "model": self.name,
            "init_params": self._init_params,
            "epoch": epoch,
        }
        # Only recorded when there actually was a resume, so the file never
        # carries a key that says nothing.
        if self._loaded_from is not None:
            cfg["resumed_from"] = str(self._loaded_from)
        cfg_path = self.config_path_for(checkpoint_path)
        # Serialised in full before the file is opened: json.dump writes as it
        # goes, so a value it cannot encode would leave a truncated config
        # behind, which reloads worse than no config at all.
        try:
            payload = json.dumps(cfg, indent=2)
        except TypeError:
            unencodable = [
                key for key, value in self._init_params.items() if not _json_safe(value)
            ]
            logger.warning(
                "Not writing %s: init_params %s cannot be stored as JSON, so this "
                "checkpoint cannot record how to rebuild itself. Pass plain values "
                "(numbers, strings, lists) as constructor arguments to fix this.",
                cfg_path,
                unencodable or sorted(self._init_params),
            )
            return
        try:
            with open(cfg_path, "w") as f:
                f.write(payload)
        except OSError as exc:  # pragma: no cover - diagnostics only
            logger.warning("Could not write %s: %s", cfg_path, exc)

    def save_model(self, save_path: Path | str, epoch: int | None = None) -> None:
        """
        Save the model state dict to *save_path*, with its config file beside it.

        Parameters
        ----------
        save_path : Path or str
            Destination ``.pt`` file for the state dict.
        epoch : int, optional
            Total epochs of training these weights represent, recorded in the
            config file.  Defaults to :attr:`best_epoch`.
        """
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        logger.info("Saving model to %s", save_path.resolve())
        torch.save(self.state_dict(), save_path)
        self._save_checkpoint_config(
            save_path, epoch if epoch is not None else self.best_epoch
        )

    # ── Evaluation ───────────────────────────────────────────────────────────

    @property
    def epoch_label(self) -> int:
        """
        Epochs of training behind the weights being scored, for file names.

        :attr:`best_epoch` is ``None`` for a model this process never trained
        and whose checkpoint recorded no epoch, which as a file name reads
        ``_eNone_``. The result files, figures and log lines written here use
        this instead, where that case is ``0``. A caller tabulating results
        wants :attr:`best_epoch` itself, which can say "no epoch is known"
        without asserting the weights are untrained.

        Returns
        -------
        int
            :attr:`best_epoch`, or ``0`` when it is not set.
        """
        return self.best_epoch if self.best_epoch is not None else 0

    def evaluate_model(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        batchsize: int = 8,
        threshold: float = 0.5,
        save_path: Path | str | None = None,
        class_names: Sequence[str] | None = None,
    ) -> dict:
        """
        Evaluate on test data and return a metrics dict.

        Computes accuracy for all tasks and additionally sensitivity, specificity,
        PPV, NPV, and AUC for binary classification, where ``pos`` / ``neg`` are
        the class balance. For more than two classes those two would be the
        counts of class indices 1 and 0, so they are omitted and macro-averaged
        rates reported instead. Either way the one-vs-rest confusion counts of
        every class, and every rate derived from them - that class's own
        accuracy included - are written to a ``*_test_per_class.csv`` beside the
        summary, which holds only the aggregate. Saves ROC and confusion-matrix
        plots to *save_path*.

        The summary's columns depend on the formulation, not on the test split:
        a split with one class present reports a nan AUC and keeps every other
        column, so two runs of the same problem type return the same keys.

        A 2-D *labels* array is one column per model output, i.e. a multi-label
        problem, and is dispatched to :meth:`_evaluate_multi_label`: there is no
        single class per sample to take an argmax of, so it reports subset and
        bit accuracy, the Hamming loss and the macro AUC over the outputs, and
        draws a 2x2 panel per output in place of a K-by-K matrix. Its per-output
        table has the same columns as the per-class one. The formulation picks
        that branch rather than the output width, so a two-label problem is not
        mistaken for a binary one.

        Parameters
        ----------
        data : np.ndarray
            Test data.
        labels : np.ndarray
            Integer ground-truth labels: 1-D class indices, or a 2-D multi-hot
            array with one column per model output.
        batchsize : int
            Samples per batch, 8 by default.
        threshold : float
            Decision threshold for binary classification.
        save_path : Path or str, optional
            Directory for the ROC and confusion-matrix plots.  Takes priority over
            the destination given to
            :func:`~xai4tsc.models.models.load_model`, and falls back to it.
            Raises if neither is set.
        class_names : sequence of str, optional
            One display name per class index, as
            :func:`~xai4tsc.data.base.class_names` returns them, used to label
            the plots and the per-class results. Classes are named by index when
            this is not given.

        Returns
        -------
        dict
            Metric values keyed by name. The summary only; the per-class counts
            are written to file.
        """
        out_dir = self._resolve_save_path(save_path)
        self.eval()

        y_true, y_probs_list = [], []
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)),
            ),
            batch_size=batchsize,
            num_workers=4,
            drop_last=False,
        )
        for x, y in loader:
            # The explicit move keeps the async host-to-device copy; predict_probs
            # then finds the batch already on the model's device and does not move
            # it again.
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            y_true.append(y.detach().cpu().numpy())
            y_probs_list.append(predict_probs(self, x))

        y_true = np.concatenate(y_true).astype(np.int64)
        y_probs = np.concatenate(y_probs_list).astype(np.float32)

        # The FORMULATION picks the branch, not the output width. A 2-D target is
        # one column per output, and every step below - argmax, exact-match
        # accuracy, a K-by-K confusion matrix, one-vs-rest AUC by index equality -
        # assumes exactly one true class per sample. Deciding on the width instead
        # would send a two-label problem down the binary path, which would report
        # pos/neg as a class balance that does not exist.
        if y_true.ndim > 1:
            return self._evaluate_multi_label(
                out_dir, y_true, y_probs, threshold, class_names
            )

        unique_classes = np.unique(y_true)
        # Determine binary vs. multiclass from model output size, not test labels,
        # so a 3-class model whose test split happens to have 2 classes still uses
        # the correct (multiclass) path.
        n_model_classes = y_probs.shape[1]
        is_binary = n_model_classes == 2

        if is_binary:
            # The positive class is the model's second output column, which is
            # class index 1 because labels are encoded 0..K-1. Reading it as
            # "the second class present in the test labels" means the same thing
            # whenever both are, and an IndexError when only one is.
            y_prob = y_probs[:, 1]
            y_pred = (y_prob >= threshold).astype(np.int64)
        else:
            y_pred = np.argmax(y_probs, axis=1).astype(np.int64)

        n = len(y_true)
        accuracy = float((y_pred == y_true).sum()) / n if n > 0 else float("nan")
        # Over all model classes, not just those present: built over the latter,
        # rows would no longer be indexed by class and an absent class would
        # shift every one that follows it.
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_model_classes))
        per_class = _confusion_counts(cm)
        class_rates = [_rates(c["tp"], c["tn"], c["fp"], c["fn"]) for c in per_class]
        # One-vs-rest AUC per class, scored on that class's own probability
        # column. Averaged over the present classes this is the macro AUC the
        # multiclass summary reports, and it is also what the OvR curves below
        # are drawn from, so the number and the picture cannot disagree. A class
        # absent from the test split, or a split with nothing to separate, has no
        # AUC and stays nan rather than excluding the rest.
        class_aucs = [float("nan")] * n_model_classes
        if len(unique_classes) > 1:
            for cls in unique_classes:
                class_aucs[cls] = roc_auc_score(
                    (y_true == cls).astype(np.int64),
                    y_probs[:, cls].astype(np.float64),
                )

        results = {"samples": n}
        if is_binary:
            # The class balance, which is what pos/neg mean here. For more than
            # two classes they would be the counts of class indices 1 and 0 and
            # read as a binary balance that does not exist, so the per-class
            # table carries each class's samples instead.
            results |= {"pos": per_class[1]["samples"], "neg": per_class[0]["samples"]}
        results["accuracy"] = f"{accuracy:.4f}"

        auc = float("nan")
        fpr_tpr_per_class = None
        # Only the AUC needs two classes to exist; every rate below is arithmetic
        # over the confusion counts and is nan where undefined. So a degenerate
        # split takes the same branch as any other and reports a nan AUC, which
        # keeps the returned keys identical to those of any other split of the
        # same problem type.
        degenerate = len(unique_classes) < 2
        if degenerate:
            logger.warning("AUC undefined: only one class present in test labels.")

        if is_binary:
            # Class 1 is the positive class; its one-vs-rest counts are the
            # binary ones, so they come from the same table the CSV reports.
            tp, tn = per_class[1]["tp"], per_class[1]["tn"]
            fp, fn = per_class[1]["fp"], per_class[1]["fn"]
            rates = class_rates[1]
            sensitivity = rates["sensitivity"]
            specificity = rates["specificity"]
            ppv, npv = rates["ppv"], rates["npv"]
            f1, mcc = rates["f1"], rates["mcc"]
            if not degenerate:
                auc = roc_auc_score(y_true, y_prob)
            results.update(
                {
                    "sensitivity": f"{sensitivity:.4f}",
                    "specificity": f"{specificity:.4f}",
                    "auc": f"{auc:.4f}",
                    "f1": f"{f1:.4f}",
                    "mcc": f"{mcc:.4f}",
                    "ppv": f"{ppv:.4f}",
                    "npv": f"{npv:.4f}",
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                }
            )
        else:
            partial = not degenerate and len(unique_classes) < n_model_classes
            if partial:
                logger.warning(
                    "Partial AUC (OvR): %d of %d model classes absent from test data.",
                    n_model_classes - len(unique_classes),
                    n_model_classes,
                )
            if not degenerate:
                auc = _macro_auc(class_aucs, unique_classes)
                fpr_tpr_per_class = []
                for cls in unique_classes:
                    y_true_bin = (y_true == cls).astype(np.int64)
                    fpr_cls, tpr_cls, _ = roc_curve(y_true_bin, y_probs[:, cls])
                    fpr_tpr_per_class.append((fpr_cls, tpr_cls))
            auc_label = f"{auc:.4f}"
            if partial:
                auc_label += (
                    f" (partial, {len(unique_classes)}/{n_model_classes} classes)"
                )
            results["auc"] = auc_label
            # The binary summary reports the single positive class's rates; the
            # multiclass one averages them, the per-class table holding the rest.
            results["macro_f1"] = f"{_macro(class_rates, 'f1'):.4f}"
            results["macro_mcc"] = f"{_macro(class_rates, 'mcc'):.4f}"

        names = _display_names(class_names, n_model_classes)

        out_dir.mkdir(parents=True, exist_ok=True)
        if fpr_tpr_per_class is not None:
            self._plot_roc_multiclass(
                out_dir,
                fpr_tpr_per_class,
                auc,
                self.epoch_label,
                class_labels=[names[int(cls)] for cls in unique_classes],
                n_model_classes=n_model_classes,
            )
        elif len(unique_classes) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            self._plot_roc(out_dir, fpr, tpr, auc, self.epoch_label)
        else:
            # Nothing to separate, so there is no curve.
            logger.warning(
                "ROC curve not plotted: only one class present in test labels."
            )
        self._plot_confusion_matrix(out_dir, cm, self.epoch_label, display_labels=names)

        df = pd.DataFrame([results])
        results_path = out_dir / f"{self.name}_e{self.epoch_label}_test_results.csv"
        df.to_csv(results_path, index=False)
        logger.info("Test set results saved to %s", results_path)

        # The confusion counts, per class, for every problem:
        per_class_rows = [
            {
                "class": k,
                "name": names[k],
                **per_class[k],
                **{key: f"{value:.4f}" for key, value in class_rates[k].items()},
                "auc": f"{class_aucs[k]:.4f}",
            }
            for k in range(n_model_classes)
        ]
        per_class_path = out_dir / f"{self.name}_e{self.epoch_label}_test_per_class.csv"
        pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)
        logger.info("Per-class test results saved to %s", per_class_path)

        if is_binary:
            counts, rates = per_class[1], class_rates[1]
            logger.info(
                "Test results Epoch %s: samples=%d pos=%d neg=%d | acc=%.4f sens=%.4f "
                "spec=%.4f auc=%.4f f1=%.4f mcc=%.4f | TP=%d TN=%d FP=%d FN=%d",
                self.epoch_label,
                n,
                counts["samples"],
                per_class[0]["samples"],
                accuracy,
                rates["sensitivity"],
                rates["specificity"],
                auc,
                rates["f1"],
                rates["mcc"],
                counts["tp"],
                counts["tn"],
                counts["fp"],
                counts["fn"],
            )
        else:
            logger.info(
                "Test results Epoch %s: samples=%d | acc=%.4f auc=%.4f macro_f1=%.4f "
                "macro_mcc=%.4f",
                self.epoch_label,
                n,
                accuracy,
                auc,
                _macro(class_rates, "f1"),
                _macro(class_rates, "mcc"),
            )

        return results

    def _evaluate_multi_label(
        self,
        out_dir: Path,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold: float,
        class_names: Sequence[str] | None,
    ) -> dict:
        """
        Evaluate a per-output (multi-label) model and write its results.

        Each output is an independent binary decision, so there is no single
        "accuracy": the summary reports **subset accuracy** (every output of a
        sample correct, the strictest reading and the one the single-label path
        would have reported), **bit accuracy** (the mean over outputs, i.e.
        ``1 - hamming_loss``) and the Hamming loss itself. None of them is
        called ``accuracy``, deliberately: the key would then name a different
        quantity than it does on the single-label path, and nothing in the
        number itself would say which. The per-output table does carry a plain
        ``accuracy``, which there means one output's own one-vs-rest accuracy,
        exactly as it does per class on the single-label path.

        The per-output counts come from the four tallies directly rather than
        from :func:`_confusion_counts`, which derives ``tn`` from a full matrix
        total and is only valid for mutually exclusive classes. :func:`_rates`
        does transfer unchanged, being pure arithmetic over four scalars, so the
        two tables report the same columns computed the same way.

        Parameters
        ----------
        out_dir : Path
            Directory for the results files and the figure.
        y_true : np.ndarray
            Multi-hot ground truth, shape ``(n_samples, n_outputs)``.
        y_probs : np.ndarray
            Independent per-output probabilities, same shape.
        threshold : float
            Decision threshold applied to every output.
        class_names : sequence of str, optional
            One display name per output, as
            :func:`~xai4tsc.data.base.class_names` returns them. Outputs are
            named by index when this is not given.

        Returns
        -------
        dict
            The summary metrics. The per-output table is written to file.
        """
        n, n_outputs = y_true.shape
        y_pred = (y_probs >= threshold).astype(np.int64)

        if n:
            subset_accuracy = float((y_pred == y_true).all(axis=1).mean())
            hamming = float((y_pred != y_true).mean())
        else:
            subset_accuracy = hamming = float("nan")

        # One 2x2 per output, [[tn, fp], [fn, tp]] - which is also the
        # orientation ConfusionMatrixDisplay reads, so the same arrays are what
        # gets plotted. _confusion_counts does not transfer here: it derives
        # `tn` from a full matrix total, which only holds for mutually
        # exclusive classes.
        matrices = (
            multilabel_confusion_matrix(y_true, y_pred)
            if n
            else np.zeros((n_outputs, 2, 2), dtype=int)
        )

        per_output, output_aucs = [], []
        for j, matrix in enumerate(matrices):
            (tn, fp), (fn, tp) = matrix
            per_output.append(
                {
                    "samples": int(tp + fn),
                    "tp": int(tp),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                }
            )
            # An output that is always present or always absent in the test split
            # has nothing to separate, so it has no AUC and stays nan rather than
            # excluding the others from the average.
            truth = y_true[:, j]
            if np.unique(truth).size < 2:
                output_aucs.append(float("nan"))
            else:
                output_aucs.append(
                    roc_auc_score(truth, y_probs[:, j].astype(np.float64))
                )

        # Each output's own (tp + tn) / n - the figure a subset accuracy hides -
        # comes from `accuracy` in here, the same place the single-label table
        # reads it from.
        output_rates = [_rates(c["tp"], c["tn"], c["fp"], c["fn"]) for c in per_output]
        names = _display_names(class_names, n_outputs)

        # An output with nothing to separate has no AUC, so the macro average is
        # over the rest and says so - the same partial reading, and the same
        # label for it, as a class absent from a single-label test split.
        defined = [j for j in range(n_outputs) if not math.isnan(output_aucs[j])]
        auc = _macro_auc(output_aucs, np.arange(n_outputs))
        if len(defined) < n_outputs:
            logger.warning(
                "Partial AUC (OvR): %d of %d outputs have only one class in the "
                "test data.",
                n_outputs - len(defined),
                n_outputs,
            )
        auc_label = f"{auc:.4f}"
        if defined and len(defined) < n_outputs:
            # Labelled only where an average was actually taken; with no output
            # left to score it is a plain nan, as a degenerate single-label
            # split reports.
            auc_label += f" (partial, {len(defined)}/{n_outputs} outputs)"

        results = {
            "samples": n,
            "outputs": n_outputs,
            "subset_accuracy": f"{subset_accuracy:.4f}",
            "bit_accuracy": f"{(1 - hamming):.4f}",  # = 1 - hamming_loss
            "hamming_loss": f"{hamming:.4f}",
            "auc": auc_label,
            "macro_f1": f"{_macro(output_rates, 'f1'):.4f}",
            "macro_mcc": f"{_macro(output_rates, 'mcc'):.4f}",
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        if defined:
            fpr_tpr = [roc_curve(y_true[:, j], y_probs[:, j])[:2] for j in defined]
            self._plot_roc_multiclass(
                out_dir,
                fpr_tpr,
                auc,
                self.epoch_label,
                class_labels=[names[j] for j in defined],
                n_model_classes=n_outputs,
            )
        else:
            logger.warning(
                "ROC curve not plotted: no output has both classes in the test labels."
            )
        # A K-by-K matrix is meaningless when a sample can be several classes at
        # once, so each output gets its own 2x2 matrix instead.
        self._plot_confusion_matrices(out_dir, matrices, self.epoch_label, names)

        results_path = out_dir / f"{self.name}_e{self.epoch_label}_test_results.csv"
        pd.DataFrame([results]).to_csv(results_path, index=False)
        logger.info("Test set results saved to %s", results_path)

        per_output_rows = [
            {
                "class": j,
                "name": names[j],
                **per_output[j],
                **{key: f"{value:.4f}" for key, value in output_rates[j].items()},
                "auc": f"{output_aucs[j]:.4f}",
            }
            for j in range(n_outputs)
        ]
        per_output_path = (
            out_dir / f"{self.name}_e{self.epoch_label}_test_per_class.csv"
        )
        pd.DataFrame(per_output_rows).to_csv(per_output_path, index=False)
        logger.info("Per-output test results saved to %s", per_output_path)

        logger.info(
            "Test results Epoch %s: samples=%d outputs=%d | subset_acc=%.4f "
            "bit_acc=%.4f hamming=%.4f auc=%.4f",
            self.epoch_label,
            n,
            n_outputs,
            subset_accuracy,
            1 - hamming,
            hamming,
            auc,
        )
        # Logged per output, unlike the single-label path: which output lags is
        # the diagnostic a subset accuracy hides, and it is why this formulation
        # exists.
        for j in range(n_outputs):
            counts = per_output[j]
            logger.info(
                "  output %d (%s): samples=%d acc=%.4f f1=%.4f auc=%.4f "
                "| TP=%d TN=%d FP=%d FN=%d",
                j,
                names[j],
                counts["samples"],
                output_rates[j]["accuracy"],
                output_rates[j]["f1"],
                output_aucs[j],
                counts["tp"],
                counts["tn"],
                counts["fp"],
                counts["fn"],
            )

        return results

    # ── Private plot helpers ─────────────────────────────────────────────────

    def _plot_roc(
        self,
        save_dir: Path,
        fpr: np.ndarray | list,
        tpr: np.ndarray | list,
        auc: float,
        best_epoch: int,
    ) -> None:
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, color="red", lw=2, label=f"ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.6)
        path = save_dir / f"{self.name}_e{best_epoch}_roc.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("ROC curve saved to %s", path)
        plt.close()

    def _plot_roc_multiclass(
        self,
        save_dir: Path,
        fpr_tpr_per_class: list[tuple],
        auc: float,
        best_epoch: int,
        class_labels: Sequence[str] | None = None,
        n_model_classes: int | None = None,
    ) -> None:
        _fig, ax = plt.subplots(figsize=(6, 5))
        for i, (fpr_cls, tpr_cls) in enumerate(fpr_tpr_per_class):
            cls = class_labels[i] if class_labels is not None else i
            ax.plot(fpr_cls, tpr_cls, lw=1.5, label=f"Class {cls}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        title = f"ROC Curves (OvR) — macro AUC = {auc:.3f}"
        n_present = len(fpr_tpr_per_class)
        if n_model_classes is not None and n_present < n_model_classes:
            title += f"\n{n_present}/{n_model_classes} classes represented in test set"
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.6)
        path = save_dir / f"{self.name}_e{best_epoch}_roc.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("ROC curves saved to %s", path)
        plt.close()

    def _plot_confusion_matrix(
        self,
        save_dir: Path,
        cm: np.ndarray,
        best_epoch: int,
        display_labels: Sequence[str] | None = None,
    ) -> None:
        _fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(ax=ax)
        ax.set_title(f"Confusion Matrix - Epoch {best_epoch}")
        path = save_dir / f"{self.name}_e{best_epoch}_confusion_matrix.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", path)
        plt.close()

    def _plot_confusion_matrices(
        self,
        save_dir: Path,
        matrices: np.ndarray,
        best_epoch: int,
        display_labels: Sequence[str],
    ) -> None:
        """
        Draw one 2x2 confusion matrix per output of a multi-label model.

        A single K-by-K matrix cannot describe a problem where one sample is
        several classes at once - its rows would not sum to the sample count -
        so each output gets its own absent/present matrix, in its own file
        named after the class. Reading them side by side is what shows which
        label the model has and has not learnt, which a subset accuracy
        averages away.

        The filename carries the output's index as well as its name: names come
        from the encoder and may be supplied by the user, so two of them can
        reduce to the same slug, and the index keeps one from overwriting the
        other's figure.

        Parameters
        ----------
        save_dir : Path
            Directory to write the figures to.
        matrices : np.ndarray
            ``(n_outputs, 2, 2)`` as
            :func:`~sklearn.metrics.multilabel_confusion_matrix` returns it:
            rows are the truth and columns the prediction, the same orientation
            as the single-label matrix.
        best_epoch : int
            Epoch the evaluated weights come from, used in the title; ``0``
            when no epoch is known (see :attr:`ModelBase.epoch_label`).
        display_labels : sequence of str
            One name per output, in order.
        """
        for index, (matrix, name) in enumerate(
            zip(matrices, display_labels, strict=True)
        ):
            _fig, ax = plt.subplots(figsize=(5, 5))
            ConfusionMatrixDisplay(matrix, display_labels=["absent", "present"]).plot(
                ax=ax
            )
            ax.set_title(f"Confusion Matrix - {name} - Epoch {best_epoch}")
            slug = _filename_slug(name, str(index))
            path = (
                save_dir
                / f"{self.name}_e{best_epoch}_confusion_matrix_{index}_{slug}.png"
            )
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info("Confusion matrix for '%s' saved to %s", name, path)
            plt.close()

    def _plot_training_curve(
        self,
        save_dir: Path,
        epoch_losses: list[float],
        epoch_accs: list[float],
        start_epoch: int = 0,
    ) -> None:
        epochs_ran = list(range(start_epoch + 1, start_epoch + len(epoch_losses) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(epochs_ran, epoch_losses)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax2.plot(epochs_ran, epoch_accs)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training Accuracy")
        ax2.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        path = save_dir / f"{self.name}_training_curve.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Training curve saved to %s", path)
        plt.close()
