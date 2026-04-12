"""Ensemble construction and submission CSV generation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import config
from ai_image_detection.pretrained import build_submission, predict_with_tta
from data_loader import make_test_loader


def build_ensemble_probs(
    fold_models: list[torch.nn.Module],
    test_df: pd.DataFrame,
    spectrum_model: torch.nn.Module | None,
    encoder_model: torch.nn.Module | None,
    include_fft: bool,
    include_vit: bool,
    device: torch.device,
) -> np.ndarray:
    """Build final test probabilities by averaging CNN folds (TTA) and
    optionally adding FFT and ViT components.

    CNN folds are weighted by N_FOLDS; auxiliary models receive weight 1.0.
    All predictions use horizontal-flip TTA.
    """
    print("\n--- Building ensemble test predictions ---")
    test_loader = make_test_loader(test_df)

    print(f"  Computing CNN TTA predictions ({len(fold_models)} folds)...")
    cnn_probs_list = [predict_with_tta(m, test_loader, device=device) for m in fold_models]
    cnn_ensemble = np.mean(cnn_probs_list, axis=0)

    components: list[tuple[np.ndarray, float]] = [(cnn_ensemble, float(config.N_FOLDS))]

    if include_fft and spectrum_model is not None:
        print("  Adding FFT dual-branch to ensemble...")
        fft_loader = make_test_loader(test_df, fft=True)
        fft_probs = predict_with_tta(spectrum_model, fft_loader, device=device)
        components.append((fft_probs, 1.0))
    else:
        print("  FFT model skipped (F1 below inclusion threshold)")

    if include_vit and encoder_model is not None:
        print("  Adding ViT-B/16 to ensemble...")
        vit_probs = predict_with_tta(encoder_model, test_loader, device=device)
        components.append((vit_probs, 1.0))
    else:
        print("  ViT model skipped (F1 below inclusion threshold)")

    total_weight = sum(w for _, w in components)
    final_probs = np.sum([p * w for p, w in components], axis=0) / total_weight
    print(f"  Ensemble: {len(components)} component(s), total weight = {total_weight:.1f}")
    return final_probs


def generate_submission(
    test_df: pd.DataFrame,
    final_test_probs: np.ndarray,
    oof_threshold: float,
    filename: str = "submission_v2_ensemble.csv",
) -> pd.DataFrame:
    """Apply threshold and write submission CSV to artifacts/."""
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    submission_df = build_submission(test_df, final_test_probs, oof_threshold)
    path = config.ARTIFACTS_DIR / filename
    submission_df.to_csv(path, index=False)

    dist = submission_df["label"].value_counts().rename_axis("label").to_frame("count")
    dist["share_%"] = (dist["count"] / len(submission_df) * 100).round(1)

    print(f"\nSubmission saved: {path}")
    print(f"Threshold used:   {oof_threshold:.3f}")
    print("Label distribution:")
    print(dist.to_string())
    return submission_df
