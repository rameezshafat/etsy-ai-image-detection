"""Evaluation functions: OOF metrics, model comparison, robustness, error analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from ai_image_detection.pretrained import (
    FineTuneImageDataset,
    build_eval_transforms,
    evaluate_predictions,
    find_best_f1_threshold,
    predict_probabilities,
)


def compute_oof_metrics(
    all_labels: np.ndarray,
    oof_probs: np.ndarray,
    fold_val_indices: list[np.ndarray],
    fold_thresholds: list[float],
) -> dict:
    """Compute aggregate OOF metrics and per-fold F1 scores."""
    oof_threshold, oof_f1 = find_best_f1_threshold(all_labels, oof_probs)
    oof_metrics = evaluate_predictions(all_labels, oof_probs, oof_threshold)

    fold_f1_per_fold = [
        find_best_f1_threshold(
            all_labels[fold_val_indices[i]],
            oof_probs[fold_val_indices[i]],
        )[1]
        for i in range(config.N_FOLDS)
    ]

    cv_summary = pd.DataFrame({
        "fold": [f"Fold {i+1}" for i in range(config.N_FOLDS)] + ["OOF (aggregate)"],
        "val_f1": [round(f, 4) for f in fold_f1_per_fold] + [round(oof_f1, 4)],
        "threshold": [round(t, 3) for t in fold_thresholds] + [round(oof_threshold, 3)],
    })

    print("\nK-Fold CV Summary:")
    print(cv_summary.to_string(index=False))
    print(f"\nOOF accuracy:   {oof_metrics['accuracy']:.4f}")
    print(f"OOF precision:  {oof_metrics['precision']:.4f}")
    print(f"OOF recall:     {oof_metrics['recall']:.4f}")

    return {
        "oof_threshold": oof_threshold,
        "oof_f1": oof_f1,
        "oof_metrics": oof_metrics,
        "fold_f1_per_fold": fold_f1_per_fold,
        "cv_summary": cv_summary,
    }


def model_comparison(
    oof_f1: float,
    oof_metrics: dict,
    cnn_fold0_f1: float,
    fold0_cnn_metrics: dict,
    fft_fold0_f1: float,
    fft_metrics: dict,
    vit_fold0_f1: float,
    vit_metrics: dict,
    include_fft: bool,
    include_vit: bool,
) -> pd.DataFrame:
    """Build a sorted comparison table of all model variants."""
    rows = [
        {
            "model": "EfficientNet-B0 CNN (OOF, k=3)",
            "val_f1": oof_f1,
            "precision": oof_metrics["precision"],
            "recall": oof_metrics["recall"],
            "accuracy": oof_metrics["accuracy"],
            "note": "Full K-Fold OOF — most reliable estimate",
        },
        {
            "model": "EfficientNet-B0 CNN (fold-0 only)",
            "val_f1": cnn_fold0_f1,
            "precision": fold0_cnn_metrics["precision"],
            "recall": fold0_cnn_metrics["recall"],
            "accuracy": fold0_cnn_metrics["accuracy"],
            "note": "Fold-0 reference for FFT/ViT comparison",
        },
        {
            "model": "FFT Dual-Branch (fold-0)",
            "val_f1": fft_fold0_f1,
            "precision": fft_metrics["precision"],
            "recall": fft_metrics["recall"],
            "accuracy": fft_metrics["accuracy"],
            "note": f"Include in ensemble: {include_fft}",
        },
        {
            "model": "ViT-B/16 (fold-0)",
            "val_f1": vit_fold0_f1,
            "precision": vit_metrics["precision"],
            "recall": vit_metrics["recall"],
            "accuracy": vit_metrics["accuracy"],
            "note": f"Include in ensemble: {include_vit}",
        },
    ]
    df = pd.DataFrame(rows).sort_values("val_f1", ascending=False).reset_index(drop=True)
    print("\nModel comparison (sorted by val F1):")
    print(df.to_string(index=False))
    return df


def robustness_eval(
    model: torch.nn.Module,
    val_df: pd.DataFrame,
    threshold: float,
    val_probs_clean: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate model on clean, JPEG-compressed, and blurred validation images."""
    eval_transform = build_eval_transforms(image_size=config.IMAGE_SIZE)

    def _perturbed_metrics(perturbation: str) -> dict:
        ds = FineTuneImageDataset(val_df, image_transform=eval_transform, perturbation=perturbation)
        loader = DataLoader(
            ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
        )
        probs = predict_probabilities(model, loader, device=device)
        return evaluate_predictions(val_df["ground_truth"].to_numpy(), probs, threshold)

    clean_m = evaluate_predictions(val_df["ground_truth"].to_numpy(), val_probs_clean, threshold)
    jpeg_m = _perturbed_metrics("jpeg")
    blur_m = _perturbed_metrics("blur")

    robustness_df = pd.DataFrame([
        {"scenario": "clean", **{k: round(clean_m[k], 4) for k in ("f1", "precision", "recall", "accuracy")}},
        {"scenario": "jpeg",  **{k: round(jpeg_m[k], 4)  for k in ("f1", "precision", "recall", "accuracy")}},
        {"scenario": "blur",  **{k: round(blur_m[k], 4)  for k in ("f1", "precision", "recall", "accuracy")}},
    ])
    robustness_df["f1_drop"] = (clean_m["f1"] - robustness_df["f1"]).round(4)

    print("\nRobustness evaluation (fold-0 model):")
    print(robustness_df.to_string(index=False))
    return robustness_df


def error_analysis(
    train_df: pd.DataFrame,
    oof_probs: np.ndarray,
    oof_threshold: float,
) -> pd.DataFrame:
    """Classify OOF predictions as correct / FP / FN and print a summary."""
    oof_preds = (oof_probs >= oof_threshold).astype(int)
    err_df = train_df[["image_id", "image_path", "ground_truth"]].copy()
    err_df["prob_ai"] = oof_probs
    err_df["pred_label"] = oof_preds
    err_df["error_type"] = "correct"
    err_df.loc[
        (err_df["ground_truth"] == 0) & (err_df["pred_label"] == 1), "error_type"
    ] = "false_positive_real_as_ai"
    err_df.loc[
        (err_df["ground_truth"] == 1) & (err_df["pred_label"] == 0), "error_type"
    ] = "false_negative_ai_as_real"

    summary = err_df["error_type"].value_counts().to_frame("count")
    summary["rate_%"] = (summary["count"] / len(err_df) * 100).round(2)

    print("\nOOF prediction outcome summary:")
    print(summary.to_string())
    return err_df
