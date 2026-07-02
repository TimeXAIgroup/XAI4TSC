"""``ModelBase`` ABC: training, prediction, evaluation, checkpoint persistence."""

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

from ..utils.defaults import t_con_defaults

logger = logging.getLogger(__name__)


class ModelBase(nn.Module, ABC):
    """
    Base class for all time series classification models.

    Subclass this to add a custom model to xai4tsc.  Register the subclass
    with :func:`xai4tsc.register_model` to make it available by name in
    experiment configs.

    Subclasses must implement :meth:`forward`.  All training, prediction,
    evaluation, and persistence logic is provided here and inherited.

    The following instance attributes are set by the model factory
    (:func:`~xai4tsc.models.models.load_model`) after construction:

    Example::

        class MyCNN(ModelBase):
            def __init__(self, in_channels: int, num_classes: int):
                super().__init__()
                self.conv = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
                self.fc = nn.Linear(32, num_classes)

            def forward(self, x):
                return self.fc(self.conv(x).mean(-1))

        xai4tsc.register_model("my_cnn", MyCNN)
    """

    name: str = None
    """Identifier used in file names and logs."""
    device: torch.device = None
    """Compute device the model lives on."""
    save_path: Path = None
    """Directory for checkpoints and diagnostic plots."""
    model_path: Path = None
    """Path to the best saved checkpoint (set after training)."""
    best_epoch: int = None
    """Epoch number of the best checkpoint (set after training)."""
    _init_params: dict = None
    """Constructor kwargs stored for checkpoint reloading."""

    def __init__(self) -> None:
        super().__init__()

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
        t_data = torch.from_numpy(data).type(torch.FloatTensor).to(self.device)
        self.eval()
        with torch.no_grad():
            # Models emit raw logits; softmax so the returned array is a true
            # probability distribution (argmax is unchanged by softmax).
            out_probs = F.softmax(self(t_data), dim=1).detach().cpu().numpy()
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
            over the destination set at construction time; falls back to that
            value and finally to the current working directory.

        Returns
        -------
        ModelBase
            ``self`` after training (best checkpoint restored).
        """
        self.save_path = Path(
            save_path if save_path is not None else (self.save_path or Path.cwd())
        )
        self.save_path.mkdir(parents=True, exist_ok=True)
        logger.info("Training %s...", self.name)

        data_tensor = torch.from_numpy(
            np.ascontiguousarray(data_train, dtype=np.float32)
        )
        labels_tensor = torch.from_numpy(
            np.ascontiguousarray(labels_train, dtype=np.int64)
        )

        epochs = hyperparams.get("epochs", t_con_defaults["epochs"])
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

        for epoch in range(epochs):
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
                    epochs,
                    epoch_loss,
                    epoch_acc,
                    val_loss,
                )
            else:
                logger.info(
                    "Epoch %2d/%d  loss: %.4f  acc: %.2f",
                    epoch + 1,
                    epochs,
                    epoch_loss,
                    epoch_acc,
                )

            if monitor_loss < (best_loss - p_tol):
                cur_patience = 0
                if hyperparams.get("save_best", t_con_defaults["save_best"]):
                    if best_path is not None:
                        best_path.unlink()
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

        if self.save_path is not None:
            self._plot_training_curve(epoch_losses, epoch_accs)

        if best_path is None:
            best_path = self.save_path / f"{self.name}_epoch_{epochs}.pt"
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

    def save_model(self, save_path: Path | str) -> None:
        """Save the model state dict to *save_path*."""
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        logger.info("Saving model to %s", str(save_path.absolute()))
        torch.save(self.state_dict(), save_path)

    @classmethod
    def load_from_checkpoint(
        cls,
        model_path: Path,
        device: str | torch.device = "cpu",
        eval: bool = True,
        **init_params: object,
    ) -> "ModelBase":
        """
        Instantiate the class and load weights from *model_path*.

        Parameters
        ----------
        model_path : Path
            Path to a ``.pt`` state-dict file produced by :meth:`save_model`.
        device : str or torch.device
            Target device.
        eval : bool
            Set the model to evaluation mode after loading.
        **init_params
            Keyword arguments forwarded to ``__init__``.

        Returns
        -------
        ModelBase
            A new instance with loaded weights.
        """
        instance = cls(**init_params).to(device)
        instance.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=device)
        )
        if eval:
            instance.eval()
        return instance

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
            the destination set at construction time; falls back to that value and
            finally to the current working directory.

        Returns
        -------
        dict
            Metric values keyed by name.
        """
        self.save_path = Path(
            save_path if save_path is not None else (self.save_path or Path.cwd())
        )
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
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            probs = F.softmax(self(x), dim=1)
            y_true.append(y.detach().cpu().numpy())
            y_probs_list.append(probs.detach().cpu().numpy())

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

        if self.save_path is not None:
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
                "Test results Epoch %s: n=%d pos=%d neg=%d | acc=%.4f sens=%.4f"
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
    ) -> None:
        epochs_ran = list(range(1, len(epoch_losses) + 1))
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
