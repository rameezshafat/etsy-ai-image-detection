"""Training loops for K-Fold CNN, FFT dual-branch, and ViT-B/16 models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from ai_image_detection.engine import seed_everything
from ai_image_detection.pretrained import (
    find_best_f1_threshold,
    freeze_backbone,
    get_discriminative_param_groups,
    predict_probabilities,
    train_binary_model,
)
from data_loader import make_fold_loaders
from models import make_cnn_model, make_fft_model, make_vit_model


def train_kfold(
    train_df: pd.DataFrame,
    device: torch.device,
    criterion: nn.Module,
) -> dict:
    """Run K-Fold cross-validation with two-phase EfficientNet-B0 fine-tuning.

    Phase 1: backbone frozen, head-only warmup.
    Phase 2: full unfreeze with discriminative learning rates.

    Returns a dict with OOF predictions, models, histories, and fold indices.
    """
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    all_labels = train_df["ground_truth"].to_numpy()

    oof_probs: np.ndarray = np.zeros(len(train_df), dtype=np.float32)
    fold_models: list[nn.Module] = []
    fold_histories: list[pd.DataFrame] = []
    fold_thresholds: list[float] = []
    fold_val_indices: list[np.ndarray] = []
    fold_train_indices: list[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(train_df)), all_labels)
    ):
        print(f"\n{'='*55}")
        print(f"  FOLD {fold_idx + 1} / {config.N_FOLDS}")
        print(f"{'='*55}")
        seed_everything(config.SEED + fold_idx)

        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        train_loader, val_loader = make_fold_loaders(fold_train_df, fold_val_df)

        bundle = make_cnn_model()
        model = bundle.model.to(device)

        # Phase 1: head-only warmup
        print("  Phase 1: head warmup (backbone frozen)")
        freeze_backbone(model)
        opt1 = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.HEAD_LR,
            weight_decay=config.HEAD_WEIGHT_DECAY,
        )
        sch1 = CosineAnnealingLR(opt1, T_max=config.PHASE1_EPOCHS)
        h1 = train_binary_model(
            model, train_loader, val_loader,
            device=device, epochs=config.PHASE1_EPOCHS,
            optimizer=opt1, scheduler=sch1, loss_fn=criterion,
        )

        # Phase 2: discriminative-LR fine-tuning
        print("  Phase 2: discriminative LR fine-tuning")
        for param in model.parameters():
            param.requires_grad = True
        opt2 = AdamW(
            get_discriminative_param_groups(
                model, head_lr=config.PHASE2_HEAD_LR, backbone_lr_factor=config.BACKBONE_LR_FACTOR
            ),
            weight_decay=config.HEAD_WEIGHT_DECAY,
        )
        sch2 = CosineAnnealingLR(opt2, T_max=config.PHASE2_EPOCHS)
        h2 = train_binary_model(
            model, train_loader, val_loader,
            device=device, epochs=config.PHASE2_EPOCHS,
            optimizer=opt2, scheduler=sch2, loss_fn=criterion,
        )

        fold_val_probs = predict_probabilities(model, val_loader, device=device)
        fold_threshold, fold_f1 = find_best_f1_threshold(
            fold_val_df["ground_truth"].to_numpy(), fold_val_probs
        )

        oof_probs[val_idx] = fold_val_probs
        fold_models.append(model)
        fold_histories.append(pd.concat([h1, h2], ignore_index=True))
        fold_thresholds.append(fold_threshold)
        fold_val_indices.append(val_idx)
        fold_train_indices.append(train_idx)

        print(f"  Fold {fold_idx + 1} val F1: {fold_f1:.4f}  threshold: {fold_threshold:.3f}")

    print("\n--- K-Fold CV complete ---")
    return {
        "oof_probs": oof_probs,
        "fold_models": fold_models,
        "fold_histories": fold_histories,
        "fold_thresholds": fold_thresholds,
        "fold_val_indices": fold_val_indices,
        "fold_train_indices": fold_train_indices,
        "all_labels": all_labels,
    }


def train_fft_model(
    fft_train_df: pd.DataFrame,
    fft_val_df: pd.DataFrame,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[nn.Module, pd.DataFrame, np.ndarray, float, float]:
    """Train FFT dual-branch EfficientNet-B0 on fold-0 split (phase-1 only)."""
    print("\n--- Training FFT dual-branch model ---")
    seed_everything(config.SEED)

    train_loader, val_loader = make_fold_loaders(fft_train_df, fft_val_df, fft=True)

    bundle = make_fft_model()
    model = bundle.model.to(device)

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=config.HEAD_WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=config.PHASE1_EPOCHS)
    history = train_binary_model(
        model, train_loader, val_loader,
        device=device, epochs=config.PHASE1_EPOCHS,
        optimizer=opt, scheduler=sch, loss_fn=criterion,
    )

    val_probs = predict_probabilities(model, val_loader, device=device)
    threshold, f1 = find_best_f1_threshold(fft_val_df["ground_truth"].to_numpy(), val_probs)
    print(f"FFT fold-0 val F1: {f1:.4f}  threshold: {threshold:.3f}")
    return model, history, val_probs, threshold, f1


def train_vit_model(
    fft_train_df: pd.DataFrame,
    fft_val_df: pd.DataFrame,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[nn.Module, pd.DataFrame, np.ndarray, float, float]:
    """Train ViT-B/16 on fold-0 split with two-phase fine-tuning.

    Uses a very conservative backbone LR (factor=0.05) to protect pretrained
    patch-embedding weights from large gradient updates.
    """
    print("\n--- Training ViT-B/16 model ---")
    seed_everything(config.SEED)

    train_loader, val_loader = make_fold_loaders(fft_train_df, fft_val_df)

    bundle = make_vit_model()
    model = bundle.model.to(device)

    # Phase 1: head warmup
    print("  ViT Phase 1: head warmup")
    freeze_backbone(model)
    opt1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.HEAD_LR,
        weight_decay=config.HEAD_WEIGHT_DECAY,
    )
    sch1 = CosineAnnealingLR(opt1, T_max=config.PHASE1_EPOCHS)
    h1 = train_binary_model(
        model, train_loader, val_loader,
        device=device, epochs=config.PHASE1_EPOCHS,
        optimizer=opt1, scheduler=sch1, loss_fn=criterion,
    )

    # Phase 2: full unfreeze, very conservative backbone LR for ViT
    print("  ViT Phase 2: full unfreeze (backbone_lr_factor=0.05)")
    for param in model.parameters():
        param.requires_grad = True
    opt2 = AdamW(
        get_discriminative_param_groups(model, head_lr=1e-4, backbone_lr_factor=0.05),
        weight_decay=config.HEAD_WEIGHT_DECAY,
    )
    sch2 = CosineAnnealingLR(opt2, T_max=config.PHASE2_EPOCHS)
    h2 = train_binary_model(
        model, train_loader, val_loader,
        device=device, epochs=config.PHASE2_EPOCHS,
        optimizer=opt2, scheduler=sch2, loss_fn=criterion,
    )

    history = pd.concat([h1, h2], ignore_index=True)
    val_probs = predict_probabilities(model, val_loader, device=device)
    threshold, f1 = find_best_f1_threshold(fft_val_df["ground_truth"].to_numpy(), val_probs)
    print(f"ViT fold-0 val F1: {f1:.4f}  threshold: {threshold:.3f}")
    return model, history, val_probs, threshold, f1
