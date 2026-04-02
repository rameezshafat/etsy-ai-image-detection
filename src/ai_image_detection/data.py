from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class MetadataSummary:
    total_rows: int
    available_rows: int
    missing_rows: int


def load_metadata(csv_path: str | Path, image_dir: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)

    df = pd.read_csv(csv_path).copy()
    df["image_path"] = df["image_id"].map(lambda name: image_dir / name)
    df["is_available"] = df["image_path"].map(Path.exists)
    return df


def summarize_records(df: pd.DataFrame) -> MetadataSummary:
    available_rows = int(df["is_available"].sum())
    total_rows = int(len(df))
    return MetadataSummary(
        total_rows=total_rows,
        available_rows=available_rows,
        missing_rows=total_rows - available_rows,
    )


def create_train_val_split(
    df: pd.DataFrame,
    *,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = df.loc[df["is_available"]].copy()
    if "ground_truth" not in available.columns:
        raise ValueError("Expected a 'ground_truth' column for train/validation splitting.")

    train_df, val_df = train_test_split(
        available,
        test_size=val_size,
        random_state=random_state,
        stratify=available["ground_truth"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_default_transforms(image_size: int = 128) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return train_transform, eval_transform


class ChallengeImageDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        *,
        transform=None,
        target_column: Optional[str] = "ground_truth",
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.transform = transform
        self.target_column = target_column if target_column in self.records.columns else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_column is None:
            return image

        label = int(row[self.target_column])
        return image, label
