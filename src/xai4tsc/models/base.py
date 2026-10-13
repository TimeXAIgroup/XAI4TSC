"""``ModelBase`` ABC: training, prediction, evaluation, checkpoint persistence."""

import json
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812 — PyTorch ecosystem convention
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

t_con_defaults = {
    "epochs": 20,
    "batchsize": 8,
    "learn_rate": 1e-3,
    "patience": 3,
    "save_best": True,
    "loss_func": "crossentropy",
    "optimizer": "adam",
}
"""Training hyperparameter defaults.

:meth:`ModelBase.train_model` falls back to these per key, so a hyperparameter the
caller omits behaves the same whether the pipeline is driven from a YAML config or
from the package API. ``master.yaml`` documents them in comments and a test holds
the two in step; it does not supply them.
"""


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


def predict_probs(model: nn.Module, data: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Return softmax class probabilities for *data* from any :class:`~torch.nn.Module`.

    Models here emit raw logits from ``forward()`` by contract, so the softmax is
    applied on the way out. The model's train/eval mode is left untouched and belongs
    to the caller.

    Parameters
    ----------
    model : torch.nn.Module
        Model to run. Must return logits of shape ``(n_samples, n_classes)``.
    data : np.ndarray or torch.Tensor
        Input samples, shape ``(n_samples, n_channels, n_timesteps)``.

    Returns
    -------
    np.ndarray
        Probabilities of shape ``(n_samples, n_classes)``, rows summing to 1.
    """
    tensor = torch.as_tensor(data, dtype=torch.float32)
    param = next(model.parameters(), None)
    if param is not None:
        tensor = tensor.to(param.device)
    with torch.no_grad():
        return F.softmax(model(tensor), dim=1).detach().cpu().numpy()


class ModelBase(nn.Module, ABC):
    """
    Base class for all time series classification models.

    Subclass this to add a custom model to xai4tsc.  Register the subclass
    with :func:`xai4tsc.register_model` to make it available by name in
    experiment configs.

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
    """Directory for checkpoints and diagnostic plots."""
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
            Directory for checkpoints and diagnostic plots.
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
            self.save_path = Path(save_path)
        return self

    def _resolve_save_path(self, save_path: Path | str | None, caller: str) -> Path:
        """
        Settle where this call writes, and remember it for later ones.

        Parameters
        ----------
        save_path : str or Path or None
            The destination passed to this call, if any.
        caller : str
            Name of the calling method, used in the error message.

        Returns
        -------
        Path
            The resolved directory, also stored on :attr:`save_path`.

        Raises
        ------
        ValueError
            If neither *save_path* nor :attr:`save_path` is set.  Defaulting to
            the working directory instead would scatter checkpoints and plots
            wherever the caller happened to be.
        """
        if save_path is None and self.save_path is None:
            raise ValueError(
                f"{caller} needs a save_path: pass save_path=... or construct "
                "via load_model(..., save_path=...)."
            )
        self.save_path = Path(save_path if save_path is not None else self.save_path)
        return self.save_path

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
        hyperparams: dict,
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
        hyperparams : dict
            Training hyperparameters.  Recognised keys: ``epochs``,
            ``batchsize``, ``learn_rate``, ``patience``, ``loss_func``,
            ``optimizer``, ``save_best``.
        data_val : np.ndarray, optional
            Validation data of shape ``(n_samples, n_channels, n_timesteps)``.
            When provided, validation loss drives early stopping and checkpointing.
        labels_val : np.ndarray, optional
            Integer class labels for *data_val*.
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
        self._resolve_save_path(save_path, "train_model")
        self.save_path.mkdir(parents=True, exist_ok=True)

        data_tensor = torch.from_numpy(
            np.ascontiguousarray(data_train, dtype=np.float32)
        )
        labels_tensor = torch.from_numpy(
            np.ascontiguousarray(labels_train, dtype=np.int64)
        )

        epochs = hyperparams.get("epochs", t_con_defaults["epochs"])
        start_epoch = start_epoch if start_epoch is not None else (self.best_epoch or 0)
        final_epoch = start_epoch + epochs
        batchsize = hyperparams.get("batchsize", t_con_defaults["batchsize"])
        learn_rate = hyperparams.get("learn_rate", t_con_defaults["learn_rate"])
        patience = hyperparams.get("patience", t_con_defaults["patience"])
        loss_func_name = hyperparams.get(
            "loss_func", t_con_defaults["loss_func"]
        ).lower()
        if loss_func_name == "crossentropy":
            loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"Unsupported loss function '{loss_func_name}'. "
                "Supported: 'crossentropy'."
            )

        optimizer_name = hyperparams.get(
            "optimizer", t_con_defaults["optimizer"]
        ).lower()
        if optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        else:
            raise ValueError(
                f"Unsupported optimizer '{optimizer_name}'. Supported: 'adam'."
            )

        loader = DataLoader(
            TensorDataset(data_tensor, labels_tensor),
            batch_size=batchsize,
        )

        val_loader = None
        if data_val is not None and labels_val is not None:
            val_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(np.ascontiguousarray(data_val, dtype=np.float32)),
                    torch.from_numpy(np.ascontiguousarray(labels_val, dtype=np.int64)),
                ),
                batch_size=batchsize,
            )

        best_loss = 1e6
        best_path = None
        cur_patience = 0
        p_tol = 5e-3
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
                optimizer.zero_grad()
                outputs = self(x_batch)
                loss = loss_func(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = torch.max(outputs.data, 1)[1]
                total += y_batch.size(0)
                correct += (preds == y_batch).sum().item()

            epoch_loss = running_loss / len(loader)
            epoch_acc = 100 * correct / total
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)

            # Monitor validation loss when available, else training loss.
            val_loss = self._validation_loss(val_loader, loss_func)
            monitor_loss = val_loss if val_loss is not None else epoch_loss
            if val_loss is not None:
                logger.info(
                    "Epoch %2d/%d  loss: %.4f  acc: %.2f  val_loss: %.4f",
                    epoch + 1,
                    final_epoch,
                    epoch_loss,
                    epoch_acc,
                    val_loss,
                )
            else:
                logger.info(
                    "Epoch %2d/%d  loss: %.4f  acc: %.2f",
                    epoch + 1,
                    final_epoch,
                    epoch_loss,
                    epoch_acc,
                )

            if monitor_loss < (best_loss - p_tol):
                cur_patience = 0
                if hyperparams.get("save_best", t_con_defaults["save_best"]):
                    if best_path is not None:
                        best_path.unlink()
                        # The superseded checkpoint's config goes with it,
                        # otherwise stale configs accumulate in the directory.
                        self.config_path_for(best_path).unlink(missing_ok=True)
                    best_path = self.save_path / f"{self.name}_epoch_{epoch + 1}.pt"
                    best_loss = monitor_loss
                    self.best_epoch = epoch + 1
                    self.save_model(best_path)
            else:
                cur_patience += 1
                if cur_patience == patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d, tol=%.4f)",
                        epoch + 1,
                        patience,
                        p_tol,
                    )
                    break

        self._plot_training_curve(epoch_losses, epoch_accs, start_epoch)

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
        logger.info("Saving model to %s", str(save_path.absolute()))
        torch.save(self.state_dict(), save_path)
        self._save_checkpoint_config(
            save_path, epoch if epoch is not None else self.best_epoch
        )

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate_model(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        hyperparams: dict,
        threshold: float = 0.5,
        save_path: Path | str | None = None,
    ) -> dict:
        """
        Evaluate on test data and return a metrics dict.

        Computes accuracy for all tasks and additionally sensitivity, specificity,
        PPV, NPV, and AUC for binary classification.  Saves ROC and
        confusion-matrix plots to *save_path*.

        Parameters
        ----------
        data : np.ndarray
            Test data.
        labels : np.ndarray
            Integer ground-truth labels.
        hyperparams : dict
            Used for ``batchsize``.
        threshold : float
            Decision threshold for binary classification.
        save_path : Path or str, optional
            Directory for the ROC and confusion-matrix plots.  Takes priority over
            the destination given to
            :func:`~xai4tsc.models.models.load_model`, and falls back to it.
            Raises if neither is set.

        Returns
        -------
        dict
            Metric values keyed by name.
        """
        self._resolve_save_path(save_path, "evaluate_model")
        self.eval()

        y_true, y_probs_list = [], []
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)),
            ),
            batch_size=hyperparams.get("batchsize", t_con_defaults["batchsize"]),
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
        unique_classes = np.unique(y_true)
        # Determine binary vs. multiclass from model output size, not test labels,
        # so a 3-class model whose test split happens to have 2 classes still uses
        # the correct (multiclass) path.
        is_binary = y_probs.shape[1] == 2

        if is_binary:
            pos_cls = unique_classes[1]
            y_prob = y_probs[:, pos_cls]
            y_pred = (y_prob >= threshold).astype(np.int64)
        else:
            y_pred = np.argmax(y_probs, axis=1).astype(np.int64)

        n = len(y_true)
        pos = np.sum(y_true == 1)
        neg = np.sum(y_true == 0)
        accuracy = float((y_pred == y_true).sum()) / n if n > 0 else float("nan")
        results = {"n": n, "pos": pos, "neg": neg, "accuracy": f"{accuracy * 100:.2f}"}

        auc = float("nan")
        fpr, tpr = [0, 1], [0, 1]
        fpr_tpr_per_class = None

        if len(unique_classes) < 2:
            logger.warning("AUC undefined: only one class present in test labels.")
        elif is_binary:
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            auc = roc_auc_score(y_true, y_prob)
            f1 = 2 * tp / (2 * tp + fp + fn)
            tpr = tp / pos
            tnr = tn / neg
            ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
            fpr = 1 - tnr
            fnr = 1 - tpr
            _for = 1 - npv
            fdr = 1 - ppv
            mcc = math.sqrt(tpr * tnr * ppv * npv) - math.sqrt(fnr * fpr * _for * fdr)
            results.update(
                {
                    "sensitivity": f"{sensitivity * 100:.2f}",
                    "specificity": f"{specificity * 100:.2f}",
                    "auc": f"{auc:.4f}",
                    "f1": f"{f1 * 100:.2f}",
                    "mcc": f"{mcc * 100:.2f}",
                    "ppv": f"{ppv * 100:.2f}",
                    "npv": f"{npv * 100:.2f}",
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                }
            )

        else:
            n_model_classes = y_probs.shape[1]
            partial = len(unique_classes) < n_model_classes
            if partial:
                logger.warning(
                    "Partial AUC (OvR): %d of %d model classes absent from test data.",
                    n_model_classes - len(unique_classes),
                    n_model_classes,
                )
            auc = roc_auc_score(
                y_true,
                y_probs[:, unique_classes].astype(np.float64),
                multi_class="ovr",
                labels=unique_classes,
            )
            fpr_tpr_per_class = []
            for cls in unique_classes:
                y_true_bin = (y_true == cls).astype(np.int64)
                if len(np.unique(y_true_bin)) < 2:
                    fpr_tpr_per_class.append(([0, 1], [0, 1]))
                else:
                    fpr_cls, tpr_cls, _ = roc_curve(y_true_bin, y_probs[:, cls])
                    fpr_tpr_per_class.append((fpr_cls, tpr_cls))
            if partial:
                auc_label = (
                    f"{auc:.4f} (partial, "
                    f"{len(unique_classes)}/{n_model_classes} classes)"
                )
            else:
                auc_label = f"{auc:.4f}"
            results["auc"] = auc_label

        self.save_path.mkdir(parents=True, exist_ok=True)
        if fpr_tpr_per_class is not None:
            self._plot_roc_multiclass(
                fpr_tpr_per_class,
                auc,
                self.best_epoch,
                class_labels=unique_classes,
                n_model_classes=n_model_classes,
            )
        else:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            self._plot_roc(fpr, tpr, auc, self.best_epoch)
        self._plot_confusion_matrix(y_true, y_pred, self.best_epoch)

        df = pd.DataFrame([results])
        results_path = (
            self.save_path / f"{self.name}_e{self.best_epoch}_test_results.csv"
        )
        df.to_csv(results_path, index=False)
        logger.info("Test set results saved to %s", results_path)

        if is_binary:
            logger.info(
                "Test results Epoch %s: n=%d pos=%d neg=%d | acc=%.4f sens=%.4f "
                "spec=%.4f auc=%.4f f1=%.4f mcc=%.4f | TP=%d TN=%d FP=%d FN=%d",
                self.best_epoch,
                n,
                pos,
                neg,
                accuracy,
                sensitivity,
                specificity,
                auc,
                f1,
                mcc,
                tp,
                tn,
                fp,
                fn,
            )
        else:
            logger.info(
                "Test results Epoch %s: n=%d acc=%.4f auc=%.4f",
                self.best_epoch,
                n,
                accuracy,
                auc,
            )

        return results

    # ── Private plot helpers ─────────────────────────────────────────────────

    def _plot_roc(
        self,
        fpr: np.ndarray | list,
        tpr: np.ndarray | list,
        auc: float,
        best_epoch: int | None,
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
        path = self.save_path / f"{self.name}_e{best_epoch}_roc.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("ROC curve saved to %s", path)
        plt.close()

    def _plot_roc_multiclass(
        self,
        fpr_tpr_per_class: list[tuple],
        auc: float,
        best_epoch: int | None,
        class_labels: np.ndarray | None = None,
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
        path = self.save_path / f"{self.name}_e{best_epoch}_roc.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("ROC curves saved to %s", path)
        plt.close()

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        best_epoch: int | None,
    ) -> None:
        _fig, ax = plt.subplots(figsize=(5, 5))
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Confusion Matrix — Epoch {best_epoch}")
        path = self.save_path / f"{self.name}_e{best_epoch}_confusion_matrix.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", path)
        plt.close()

    def _plot_training_curve(
        self,
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
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training Accuracy")
        ax2.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        path = self.save_path / f"{self.name}_training_curve.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Training curve saved to %s", path)
        plt.close()
