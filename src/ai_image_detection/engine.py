from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm.auto import tqdm


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(
    model: nn.Module,
    data_loader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    running_loss = 0.0
    predictions = []
    targets = []

    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            targets.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = running_loss / max(len(data_loader.dataset), 1)
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, zero_division=0)
    return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int = 5,
) -> pd.DataFrame:
    history: list[dict[str, float]] = []
    model.to(device)

    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        epoch_start = time.perf_counter()
        train_metrics = _run_epoch(model, train_loader, loss_fn, device, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, loss_fn, device)
        epoch_seconds = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"| val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f}"
        )

    return pd.DataFrame(history)


def evaluate(model: nn.Module, data_loader, *, loss_fn: nn.Module, device: torch.device) -> dict[str, object]:
    model.to(device)
    metrics = _run_epoch(model, data_loader, loss_fn, device)
    y_true = []
    y_pred = []

    model.eval()
    with torch.inference_mode():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())

    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "y_true": y_true,
        "y_pred": y_pred,
    }


def predict_probabilities(model: nn.Module, data_loader, *, device: torch.device) -> np.ndarray:
    model.to(device)
    model.eval()
    probabilities = []

    with torch.inference_mode():
        for batch in data_loader:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            inputs = inputs.to(device)
            probs = torch.softmax(model(inputs), dim=1)[:, 1]
            probabilities.extend(probs.cpu().numpy().tolist())

    return np.asarray(probabilities)


def build_submission_frame(records: pd.DataFrame, probabilities: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    submission = records.loc[records["is_available"], ["image_id"]].copy()
    submission["ground_truth"] = (probabilities >= threshold).astype(int)
    return submission.reset_index(drop=True)


def save_model(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
