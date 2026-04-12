"""Data loading and DataLoader construction."""
from __future__ import annotations

import pandas as pd
from torch.utils.data import DataLoader

import config
from ai_image_detection.data import load_metadata
from ai_image_detection.pretrained import (
    FineTuneImageDataset,
    build_augmented_train_transforms,
    build_eval_transforms,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and filter train/test metadata. Only keeps rows with available images."""
    train_df = load_metadata(config.TRAIN_CSV, config.IMAGE_DIR)
    test_df = load_metadata(config.TEST_CSV, config.IMAGE_DIR)

    train_df = train_df.loc[train_df["is_available"]].copy().reset_index(drop=True)
    test_df = test_df.loc[test_df["is_available"]].copy().reset_index(drop=True)
    train_df["label_name"] = train_df["ground_truth"].map({0: "real", 1: "ai_generated"})

    return train_df, test_df


def make_fold_loaders(
    fold_train_df: pd.DataFrame,
    fold_val_df: pd.DataFrame,
    *,
    fft: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders for one fold.

    Train split uses augmented transforms (JPEG simulation included).
    Val split uses plain eval transforms.
    """
    train_transform = build_augmented_train_transforms(image_size=config.IMAGE_SIZE)
    eval_transform = build_eval_transforms(image_size=config.IMAGE_SIZE)

    train_ds = FineTuneImageDataset(
        fold_train_df, image_transform=train_transform, include_fft_image=fft
    )
    val_ds = FineTuneImageDataset(
        fold_val_df, image_transform=eval_transform, include_fft_image=fft
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )
    return train_loader, val_loader


def make_test_loader(test_df: pd.DataFrame, *, fft: bool = False) -> DataLoader:
    """Build the test DataLoader (no labels, eval transforms)."""
    eval_transform = build_eval_transforms(image_size=config.IMAGE_SIZE)
    test_ds = FineTuneImageDataset(
        test_df,
        image_transform=eval_transform,
        target_column=None,
        include_fft_image=fft,
    )
    return DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )
