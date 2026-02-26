"""
metrics.py
==========
Evaluation metrics for Res-TranBiLSTM per paper Section 4.3.

Paper Eqs. (9)-(12):
  Accuracy  = (TP+TN) / (TP+FP+TN+FN)        [Eq. 9]
  Precision = TP / (TP+FP)                     [Eq. 10]
  Recall    = TP / (TP+FN)                     [Eq. 11]
  F1-score  = 2 * (Precision*Recall) / (Precision+Recall)  [Eq. 12]

"For multiclassification problems, when evaluating the classification
of a certain traffic class, this traffic class is considered a positive
sample, while the remaining traffic classes are considered negative
samples." → macro-averaged metrics.

Author: FYP Implementation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute all paper metrics for multi-class classification.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_pred : np.ndarray, shape (N,)
    class_names : list of str, optional
    average : str — sklearn averaging strategy ('macro', 'weighted', etc.)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, plus per-class stats.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=average, zero_division=0)

    results = {
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
    }

    # Per-class metrics
    labels = list(range(len(class_names))) if class_names else None
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    results["per_class"] = report

    return results


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Run full evaluation on test set.

    Parameters
    ----------
    model      : trained ResTranBiLSTM
    test_loader: DataLoader with (x_img, x_seq, y) batches
    device     : torch device
    class_names: list of class label strings

    Returns
    -------
    dict with all metrics and raw predictions.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_img, x_seq, y in test_loader:
            x_img = x_img.to(device)
            x_seq = x_seq.to(device)
            logits = model(x_img, x_seq)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = compute_metrics(y_true, y_pred, class_names=class_names)
    metrics["y_true"] = y_true.tolist()
    metrics["y_pred"] = y_pred.tolist()

    logger.info(f"Test Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall:    {metrics['recall']:.4f}")
    logger.info(f"Test F1:        {metrics['f1']:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """
    Plot and optionally save confusion matrix.

    Paper Fig. 9 style: rows = actual class, columns = predicted class.

    Parameters
    ----------
    normalize : bool — if True, normalize by row (true label counts).
    save_path : str — if given, save figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            ax.text(
                j, i,
                format(cm_display[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=8,
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual Class", fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved: {save_path}")

    return fig


def plot_per_class_metrics(
    metrics: Dict,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Per-Class Performance",
) -> plt.Figure:
    """
    Bar chart of precision, recall, F1 per class (like paper Figs. 10-12).
    """
    precisions, recalls, f1s = [], [], []
    for cls in class_names:
        pc = metrics["per_class"].get(cls, {})
        precisions.append(pc.get("precision", 0.0) * 100)
        recalls.append(pc.get("recall", 0.0) * 100)
        f1s.append(pc.get("f1-score", 0.0) * 100)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.5), 5))
    ax.bar(x - width, precisions, width, label="Precision", color="#2196F3", alpha=0.85)
    ax.bar(x,         recalls,    width, label="Recall",    color="#FF9800", alpha=0.85)
    ax.bar(x + width, f1s,        width, label="F1-Score",  color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Per-class metrics plot saved: {save_path}")

    return fig


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    title: str = "Training History",
) -> plt.Figure:
    """Plot training/validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#F44336")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#2196F3")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct   = [a * 100 for a in history["val_acc"]]
    axes[1].plot(epochs, train_acc_pct, label="Train Acc", color="#F44336")
    axes[1].plot(epochs, val_acc_pct,   label="Val Acc",   color="#2196F3")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training history plot saved: {save_path}")

    return fig


def save_metrics_json(metrics: Dict, path: str) -> None:
    """Save metrics dict to JSON (excluding large arrays)."""
    save = {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred")}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(save, f, indent=2)
    logger.info(f"Metrics saved: {path}")


# ---------------------------------------------------------------------------
# DoS-specific class names
# ---------------------------------------------------------------------------

DOS_CLASS_NAMES = [
    "BENIGN",
    "DoS_slowloris",
    "DoS_Slowhttptest",
    "DoS_Hulk",
    "DoS_GoldenEye",
]

CICIDS_CLASS_NAMES = [
    "Normal", "Bot", "Brute Force", "DDoS", "DoS", "PortScan", "Web Attack"
]