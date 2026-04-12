"""Plotting helpers, console logging, and artifact persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # must precede pyplot import for script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

import config


# ── Console logging ───────────────────────────────────────────────────────────

def log_stage(name: str) -> None:
    """Print a visible stage header."""
    bar = "#" * 64
    print(f"\n{bar}")
    print(f"#  {name}")
    print(bar)


# ── Figure helpers ────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, filename: str) -> Path:
    """Save figure to artifacts/ and close it."""
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_learning_curves(
    fold_histories: list[pd.DataFrame],
    phase1_epochs: int,
    title: str = "Learning Curves",
) -> plt.Figure:
    """Plot train/val F1 per fold with a vertical phase-boundary line."""
    n = len(fold_histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for i, (history, ax) in enumerate(zip(fold_histories, axes)):
        epochs = range(1, len(history) + 1)
        ax.plot(epochs, history["train_f1"], "o-", label="Train F1")
        ax.plot(epochs, history["val_f1"], "s-", label="Val F1")
        ax.axvline(phase1_epochs + 0.5, color="gray", ls="--", alpha=0.6, label="Phase 2 start")
        gap = history["train_f1"].iloc[-1] - history["val_f1"].iloc[-1]
        status = "OVERFIT" if gap > 0.15 else ("UNDERFIT" if gap < -0.05 else "healthy")
        ax.set_title(f"Fold {i+1}  gap={gap:.3f} [{status}]")
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("F1")
        ax.legend(fontsize=8)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


def plot_loss_curves(
    fold_histories: list[pd.DataFrame],
    title: str = "Loss Curves",
) -> plt.Figure:
    """Plot train/val loss per fold."""
    n = len(fold_histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for i, (history, ax) in enumerate(zip(fold_histories, axes)):
        epochs = range(1, len(history) + 1)
        ax.plot(epochs, history["train_loss"], "o-", label="Train Loss")
        ax.plot(epochs, history["val_loss"], "s-", label="Val Loss")
        ax.set_title(f"Fold {i+1}")
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.legend(fontsize=8)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(confusion_matrix: np.ndarray, f1: float) -> plt.Figure:
    """Plot a confusion matrix labelled with class names."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix, display_labels=["real", "ai_generated"]
    ).plot(ax=ax, colorbar=False)
    ax.set_title(f"OOF Confusion Matrix  (F1={f1:.3f})")
    plt.tight_layout()
    return fig


# ── Artifact persistence ──────────────────────────────────────────────────────

def save_metrics(metrics: dict[str, Any], filename: str = "metrics_summary.json") -> Path:
    """Serialize a metrics dict to JSON in artifacts/."""
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACTS_DIR / filename

    def _serialize(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, pd.DataFrame):
            return v.to_dict(orient="records")
        if isinstance(v, list):
            return [_serialize(i) for i in v]
        return v

    with open(path, "w") as f:
        json.dump({k: _serialize(v) for k, v in metrics.items()}, f, indent=2)

    print(f"  Saved metrics: {path}")
    return path


def save_history(histories: list[pd.DataFrame], filename: str = "training_history.csv") -> Path:
    """Concatenate per-fold histories and save to a single CSV."""
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACTS_DIR / filename
    combined = pd.concat(
        [h.assign(fold=i + 1) for i, h in enumerate(histories)],
        ignore_index=True,
    )
    combined.to_csv(path, index=False)
    print(f"  Saved training history: {path}")
    return path
