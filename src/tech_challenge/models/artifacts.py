from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axis, colorbar=False)
    axis.set_title(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_pr_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axis)
    axis.set_title(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=axis)
    axis.set_title(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_training_history(
    history: list[dict[str, float]], output_path: Path, title: str = "Training History"
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    valid_loss = [item["valid_loss"] for item in history]
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, train_loss, label="train_loss")
    axis.plot(epochs, valid_loss, label="valid_loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(title)
    axis.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
