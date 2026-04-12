"""
Pretrained Fine-Tuning AI Image Detection Pipeline
===================================================
Reproduces notebooks/02_pretrained_finetuning_ai_detection.ipynb as a
standalone command-line script. Run from the repo root or from this directory:

    python pretrained_model_project/main.py
    # or
    cd pretrained_model_project && python main.py

All outputs are written to artifacts/ at the repo root.

Stages
------
  0.  Environment setup and path validation
  1.  Load data
  2.  K-Fold EfficientNet-B0 training (phase-1 head warmup + phase-2 fine-tune)
  3.  OOF evaluation (accuracy, precision, recall, F1, confusion matrix)
  4.  FFT dual-branch model (fold-0 comparison)
  5.  ViT-B/16 model (fold-0 comparison)
  6.  Model comparison table
  7.  Ensemble construction and submission generation
  8.  Robustness evaluation (JPEG / blur perturbations)
  9.  Error analysis (false positives / false negatives)
  10. Save metrics summary
"""
from __future__ import annotations

import sys
from pathlib import Path

# Register src/ before any local imports so ai_image_detection is available.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from torch import nn

import config
from ai_image_detection.engine import seed_everything, select_device
from ai_image_detection.pretrained import evaluate_predictions, find_best_f1_threshold
from data_loader import load_data
from evaluate import (
    compute_oof_metrics,
    error_analysis,
    model_comparison,
    robustness_eval,
)
from inference import build_ensemble_probs, generate_submission
from train import train_fft_model, train_kfold, train_vit_model
from utils import (
    log_stage,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_loss_curves,
    save_fig,
    save_history,
    save_metrics,
)


def _validate_paths() -> None:
    missing = [
        str(p) for p in [config.TRAIN_CSV, config.TEST_CSV, config.IMAGE_DIR]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing dataset paths: " + ", ".join(missing))
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Project root:", config.PROJECT_ROOT)
    print("Artifacts:   ", config.ARTIFACTS_DIR)


def main() -> None:
    # ── 0. Setup ──────────────────────────────────────────────────────────────
    log_stage("0. Environment Setup")
    _validate_paths()
    seed_everything(config.SEED)
    device = select_device()
    criterion = nn.BCEWithLogitsLoss()
    print("Device:", device)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log_stage("1. Load Data")
    train_df, test_df = load_data()
    print(f"Train: {len(train_df):,}   Test: {len(test_df):,}")
    print("Class distribution:")
    print(train_df["label_name"].value_counts().to_string())

    # ── 2. K-Fold EfficientNet-B0 training ───────────────────────────────────
    log_stage("2. K-Fold EfficientNet-B0 Training")
    kfold = train_kfold(train_df, device, criterion)

    oof_probs         = kfold["oof_probs"]
    fold_models       = kfold["fold_models"]
    fold_histories    = kfold["fold_histories"]
    fold_thresholds   = kfold["fold_thresholds"]
    fold_val_indices  = kfold["fold_val_indices"]
    fold_train_indices = kfold["fold_train_indices"]
    all_labels        = kfold["all_labels"]

    save_history(fold_histories, "pretrained_training_history.csv")

    # ── 3. OOF Evaluation ─────────────────────────────────────────────────────
    log_stage("3. OOF Evaluation")
    oof_eval = compute_oof_metrics(all_labels, oof_probs, fold_val_indices, fold_thresholds)
    oof_threshold     = oof_eval["oof_threshold"]
    oof_f1            = oof_eval["oof_f1"]
    oof_metrics       = oof_eval["oof_metrics"]
    fold_f1_per_fold  = oof_eval["fold_f1_per_fold"]

    fig = plot_confusion_matrix(oof_metrics["confusion_matrix"], oof_f1)
    save_fig(fig, "pretrained_oof_confusion_matrix.png")

    fig = plot_learning_curves(
        fold_histories, config.PHASE1_EPOCHS, "EfficientNet-B0 K-Fold Learning Curves"
    )
    save_fig(fig, "pretrained_learning_curves.png")

    fig = plot_loss_curves(fold_histories, "EfficientNet-B0 K-Fold Loss Curves")
    save_fig(fig, "pretrained_loss_curves.png")

    # ── 4. FFT dual-branch model ──────────────────────────────────────────────
    log_stage("4. FFT Dual-Branch Model (fold-0)")
    fft_train_df  = train_df.iloc[fold_train_indices[0]].reset_index(drop=True)
    fft_val_df    = train_df.iloc[fold_val_indices[0]].reset_index(drop=True)
    fft_val_targets = fft_val_df["ground_truth"].to_numpy()
    cnn_fold0_f1  = fold_f1_per_fold[0]

    spectrum_model, fft_history, fft_val_probs, fft_threshold, fft_fold0_f1 = train_fft_model(
        fft_train_df, fft_val_df, device, criterion
    )
    fft_metrics = evaluate_predictions(fft_val_targets, fft_val_probs, fft_threshold)

    fft_delta  = fft_fold0_f1 - cnn_fold0_f1
    include_fft = fft_fold0_f1 >= cnn_fold0_f1 - config.INCLUDE_THRESHOLD
    print(f"Include FFT in ensemble: {include_fft}  (delta={fft_delta:+.4f})")

    # ── 5. ViT-B/16 model ────────────────────────────────────────────────────
    log_stage("5. ViT-B/16 Model (fold-0)")
    encoder_model, vit_history, vit_val_probs, vit_threshold, vit_fold0_f1 = train_vit_model(
        fft_train_df, fft_val_df, device, criterion
    )
    vit_metrics = evaluate_predictions(fft_val_targets, vit_val_probs, vit_threshold)

    vit_delta  = vit_fold0_f1 - cnn_fold0_f1
    include_vit = vit_fold0_f1 >= cnn_fold0_f1 - config.INCLUDE_THRESHOLD
    print(f"Include ViT in ensemble: {include_vit}  (delta={vit_delta:+.4f})")

    # ── 6. Model comparison ───────────────────────────────────────────────────
    log_stage("6. Model Comparison")
    fold0_cnn_metrics = evaluate_predictions(
        fft_val_targets, oof_probs[fold_val_indices[0]], fold_thresholds[0]
    )
    comparison_df = model_comparison(
        oof_f1, oof_metrics,
        cnn_fold0_f1, fold0_cnn_metrics,
        fft_fold0_f1, fft_metrics,
        vit_fold0_f1, vit_metrics,
        include_fft, include_vit,
    )
    comparison_df.to_csv(config.ARTIFACTS_DIR / "pretrained_model_comparison.csv", index=False)

    # ── 7. Ensemble & submission ──────────────────────────────────────────────
    log_stage("7. Ensemble & Submission")
    final_test_probs = build_ensemble_probs(
        fold_models, test_df,
        spectrum_model if include_fft else None,
        encoder_model if include_vit else None,
        include_fft, include_vit,
        device,
    )
    generate_submission(test_df, final_test_probs, oof_threshold)

    # ── 8. Robustness evaluation ──────────────────────────────────────────────
    log_stage("8. Robustness Evaluation")
    fold0_clean_probs = oof_probs[fold_val_indices[0]]
    robustness_df = robustness_eval(
        fold_models[0], fft_val_df, fold_thresholds[0], fold0_clean_probs, device
    )
    robustness_df.to_csv(config.ARTIFACTS_DIR / "pretrained_robustness.csv", index=False)

    # ── 9. Error analysis ─────────────────────────────────────────────────────
    log_stage("9. Error Analysis")
    error_analysis(train_df, oof_probs, oof_threshold)

    # ── 10. Save metrics summary ──────────────────────────────────────────────
    log_stage("10. Metrics Summary")
    metrics_summary = {
        "model": "EfficientNet-B0 K-Fold + FFT + ViT ensemble",
        "n_folds": config.N_FOLDS,
        "oof_f1": float(oof_f1),
        "oof_threshold": float(oof_threshold),
        "oof_accuracy": float(oof_metrics["accuracy"]),
        "oof_precision": float(oof_metrics["precision"]),
        "oof_recall": float(oof_metrics["recall"]),
        "fold_f1": [float(f) for f in fold_f1_per_fold],
        "fold_thresholds": [float(t) for t in fold_thresholds],
        "fft_fold0_f1": float(fft_fold0_f1),
        "vit_fold0_f1": float(vit_fold0_f1),
        "include_fft": bool(include_fft),
        "include_vit": bool(include_vit),
        "phase1_epochs": config.PHASE1_EPOCHS,
        "phase2_epochs": config.PHASE2_EPOCHS,
        "head_lr": config.HEAD_LR,
        "phase2_head_lr": config.PHASE2_HEAD_LR,
        "backbone_lr_factor": config.BACKBONE_LR_FACTOR,
    }
    save_metrics(metrics_summary, "pretrained_metrics_summary.json")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Pipeline complete.")
    print(f"  OOF F1:        {oof_f1:.4f}")
    print(f"  OOF Accuracy:  {oof_metrics['accuracy']:.4f}")
    print(f"  OOF Precision: {oof_metrics['precision']:.4f}")
    print(f"  OOF Recall:    {oof_metrics['recall']:.4f}")
    print(f"  Threshold:     {oof_threshold:.3f}")
    print(f"  Submission:    {config.ARTIFACTS_DIR / 'submission_v2_ensemble.csv'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
