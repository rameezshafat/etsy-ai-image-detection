from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelBundle:
    model: nn.Module
    model_name: str
    input_size: int
    feature_dim: int


@dataclass
class EpochMetrics:
    loss: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    threshold: float


def build_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class RandomJPEGCompression:
    """PIL transform: randomly apply JPEG compression to simulate real-world encoding artifacts."""

    def __init__(self, quality_min: int = 65, quality_max: int = 95, p: float = 0.3) -> None:
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def build_augmented_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Train transforms with JPEG compression simulation for AI-artifact robustness."""
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            RandomJPEGCompression(quality_min=70, quality_max=95, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_fft_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Train transforms for log-magnitude FFT spectra.

    Uses the same ImageNet normalisation as the RGB branch so the pretrained
    EfficientNet backbone receives activations in its expected range.  Colour
    jitter is deliberately omitted — spectra are greyscale-derived and
    channel-wise colour shifts have no physical meaning there.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_fft_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Eval transforms for log-magnitude FFT spectra."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _bytes_to_tensor(path: Path) -> torch.Tensor:
    return torch.tensor([path.stat().st_size / 1024.0], dtype=torch.float32)


def _extract_handcrafted_features(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        exif = img.getexif()
        has_exif = 1.0 if exif and len(exif) > 0 else 0.0
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)

    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))
    gray = arr.mean(axis=2)
    pixel_std = float(gray.std())

    gy, gx = np.gradient(gray)
    laplacian_proxy = float(np.var(gx) + np.var(gy))

    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(fft)
    cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    radius = max(4, min(magnitude.shape) // 8)
    yy, xx = np.ogrid[: magnitude.shape[0], : magnitude.shape[1]]
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high_freq_energy = magnitude[distance >= radius].sum()
    total_energy = magnitude.sum()
    fft_ratio = float(high_freq_energy / total_energy) if total_energy else 0.0

    return np.asarray(
        [
            rgb_mean[0],
            rgb_mean[1],
            rgb_mean[2],
            rgb_std[0],
            rgb_std[1],
            rgb_std[2],
            pixel_std,
            laplacian_proxy,
            fft_ratio,
            has_exif,
            path.stat().st_size / 1024.0,
        ],
        dtype=np.float32,
    )


class FineTuneImageDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        *,
        image_transform: Callable,
        target_column: str | None = "ground_truth",
        include_fft_image: bool = False,
        fft_transform: Callable | None = None,
        include_handcrafted: bool = False,
        perturbation: str | None = None,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.image_transform = image_transform
        self.target_column = target_column if target_column in self.records.columns else None
        self.include_fft_image = include_fft_image
        # Fall back to the RGB transform when no FFT-specific transform is given,
        # preserving backward-compatibility with existing call sites.
        self.fft_transform = fft_transform if fft_transform is not None else image_transform
        self.include_handcrafted = include_handcrafted
        self.perturbation = perturbation

    def __len__(self) -> int:
        return len(self.records)

    def _apply_perturbation(self, img: Image.Image) -> Image.Image:
        if self.perturbation is None:
            return img
        if self.perturbation == "jpeg":
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=45)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        if self.perturbation == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=1.5))
        raise ValueError(f"Unknown perturbation: {self.perturbation}")

    def _make_fft_image(self, img: Image.Image) -> Image.Image:
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
        spectrum = spectrum - spectrum.min()
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()
        spectrum_uint8 = (spectrum * 255.0).astype(np.uint8)
        return Image.fromarray(spectrum_uint8, mode="L").convert("RGB")

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image_path = Path(row["image_path"])

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = self._apply_perturbation(img)
            image_tensor = self.image_transform(img)

            batch = {
                "image": image_tensor,
                "image_id": row["image_id"],
            }
            if self.include_fft_image:
                batch["fft_image"] = self.fft_transform(self._make_fft_image(img))
            if self.include_handcrafted:
                batch["handcrafted"] = torch.tensor(_extract_handcrafted_features(image_path), dtype=torch.float32)

        if self.target_column is not None:
            batch["target"] = torch.tensor(float(row[self.target_column]), dtype=torch.float32)
        return batch


class FFTDualBranchClassifier(nn.Module):
    def __init__(self, image_branch: nn.Module, freq_branch: nn.Module, feature_dim: int, handcrafted_dim: int = 0) -> None:
        super().__init__()
        self.image_branch = image_branch
        self.freq_branch = freq_branch
        fusion_dim = feature_dim * 2 + handcrafted_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, image: torch.Tensor, fft_image: torch.Tensor, handcrafted: torch.Tensor | None = None) -> torch.Tensor:
        image_features = self.image_branch(image)
        freq_features = self.freq_branch(fft_image)
        pieces = [image_features, freq_features]
        if handcrafted is not None:
            pieces.append(handcrafted)
        fused = torch.cat(pieces, dim=1)
        return self.classifier(fused).squeeze(1)


class EmbeddingClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, handcrafted_dim: int = 0) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + handcrafted_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, image: torch.Tensor, handcrafted: torch.Tensor | None = None) -> torch.Tensor:
        features = self.backbone(image)
        # ViT's MultiheadAttention can produce non-contiguous tensors after
        # internal transpose ops. .contiguous() is a no-op for EfficientNet.
        features = features.contiguous()
        if handcrafted is not None:
            features = torch.cat([features, handcrafted], dim=1)
        return self.classifier(features).squeeze(1)


def _replace_head_and_get_dim(model: nn.Module, model_name: str) -> tuple[nn.Module, int]:
    if model_name.startswith("resnet"):
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    if model_name.startswith("efficientnet"):
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    if model_name.startswith("vit"):
        feature_dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
        return model, feature_dim
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_pretrained_backbone(model_name: str = "efficientnet_b0", pretrained: bool = True) -> ModelBundle:
    def _safe_weights(default_weights):
        return default_weights if pretrained else None

    def _build_with_fallback(builder, weights):
        try:
            return builder(weights=weights)
        except Exception as exc:
            if weights is not None:
                print(
                    f"Warning: failed to load pretrained weights for {model_name}. "
                    f"Falling back to randomly initialized weights. Details: {exc}"
                )
                return builder(weights=None)
            raise

    if model_name == "resnet50":
        weights = _safe_weights(models.ResNet50_Weights.DEFAULT)
        backbone = _build_with_fallback(models.resnet50, weights)
    elif model_name == "efficientnet_b0":
        weights = _safe_weights(models.EfficientNet_B0_Weights.DEFAULT)
        backbone = _build_with_fallback(models.efficientnet_b0, weights)
    elif model_name == "efficientnet_b2":
        weights = _safe_weights(models.EfficientNet_B2_Weights.DEFAULT)
        backbone = _build_with_fallback(models.efficientnet_b2, weights)
    elif model_name == "efficientnet_b3":
        weights = _safe_weights(models.EfficientNet_B3_Weights.DEFAULT)
        backbone = _build_with_fallback(models.efficientnet_b3, weights)
    elif model_name == "vit_b_16":
        weights = _safe_weights(models.ViT_B_16_Weights.DEFAULT)
        backbone = _build_with_fallback(models.vit_b_16, weights)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    backbone, feature_dim = _replace_head_and_get_dim(backbone, model_name)
    return ModelBundle(model=backbone, model_name=model_name, input_size=224, feature_dim=feature_dim)


def build_artifact_model(model_name: str = "efficientnet_b0", pretrained: bool = True, handcrafted_dim: int = 0) -> ModelBundle:
    bundle = build_pretrained_backbone(model_name=model_name, pretrained=pretrained)
    model = EmbeddingClassifier(bundle.model, bundle.feature_dim, handcrafted_dim=handcrafted_dim)
    return ModelBundle(model=model, model_name=f"artifact_{model_name}", input_size=bundle.input_size, feature_dim=bundle.feature_dim)


def build_spectrum_model(model_name: str = "efficientnet_b0", pretrained: bool = True, handcrafted_dim: int = 0) -> ModelBundle:
    image_bundle = build_pretrained_backbone(model_name=model_name, pretrained=pretrained)
    freq_bundle = build_pretrained_backbone(model_name=model_name, pretrained=pretrained)
    model = FFTDualBranchClassifier(
        image_branch=image_bundle.model,
        freq_branch=freq_bundle.model,
        feature_dim=image_bundle.feature_dim,
        handcrafted_dim=handcrafted_dim,
    )
    return ModelBundle(model=model, model_name=f"spectrum_{model_name}", input_size=image_bundle.input_size, feature_dim=image_bundle.feature_dim * 2)


def build_encoder_model(model_name: str = "vit_b_16", pretrained: bool = True, handcrafted_dim: int = 0) -> ModelBundle:
    bundle = build_pretrained_backbone(model_name=model_name, pretrained=pretrained)
    model = EmbeddingClassifier(bundle.model, bundle.feature_dim, handcrafted_dim=handcrafted_dim)
    return ModelBundle(model=model, model_name=f"encoder_{model_name}", input_size=bundle.input_size, feature_dim=bundle.feature_dim)


def get_discriminative_param_groups(
    model: nn.Module,
    head_lr: float,
    backbone_lr_factor: float = 0.1,
) -> list[dict]:
    """Return AdamW param groups with a lower LR for the backbone than the classifier head.

    The classifier head gets ``head_lr``; all other trainable parameters get
    ``head_lr * backbone_lr_factor``.  This prevents catastrophic forgetting of
    pretrained features while still allowing fine-tuning.
    """
    head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    head_trainable = [p for p in head_params if p.requires_grad]
    return [
        {"params": backbone_params, "lr": head_lr * backbone_lr_factor},
        {"params": head_trainable, "lr": head_lr},
    ]


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_layers(model: nn.Module, last_n_children: int = 2) -> None:
    for param in model.parameters():
        param.requires_grad = False
    children = list(model.children())
    for child in children[-last_n_children:]:
        for param in child.parameters():
            param.requires_grad = True
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def forward_batch(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "fft_image" in batch and "handcrafted" in batch:
        return model(batch["image"], batch["fft_image"], batch["handcrafted"])
    if "fft_image" in batch:
        return model(batch["image"], batch["fft_image"])
    if "handcrafted" in batch:
        return model(batch["image"], batch["handcrafted"])
    return model(batch["image"])


def find_best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray | None = None) -> tuple[float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.3, 0.7, 81)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    preds = (probabilities >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "accuracy": accuracy_score(y_true, preds),
        "confusion_matrix": confusion_matrix(y_true, preds),
    }


class FocalLoss(nn.Module):
    """Binary Focal Loss (Lin et al., 2017) operating on raw logits.

    Focal Loss down-weights well-classified examples so training focuses on
    the hard ones.  With ``gamma=0`` it reduces to standard BCE; increasing
    ``gamma`` increases the focusing effect.

    ``alpha`` is the positive-class weight (balancing factor).  Tuning it
    alongside ``gamma`` typically yields better precision–recall trade-offs
    than plain BCE on near-balanced datasets.

    Args:
        alpha: weight applied to the positive class (ai_generated).
               A value of 0.25 slightly emphasises the minority class.
        gamma: focusing exponent.  Recommended range: 1–3.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t) ** self.gamma * bce).mean()


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_training = optimizer is not None
    model.train(is_training)
    running_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for batch in data_loader:
            batch = _batch_to_device(batch, device)
            logits = forward_batch(model, batch)
            targets = batch["target"]
            loss = loss_fn(logits, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * targets.size(0)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probabilities)
            all_targets.append(targets.detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    avg_loss = running_loss / max(len(data_loader.dataset), 1)
    return avg_loss, y_true, y_prob


def train_binary_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    loss_fn: nn.Module | None = None,
) -> pd.DataFrame:
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_y, train_prob = run_epoch(
            model,
            train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        val_loss, val_y, val_prob = run_epoch(
            model,
            val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
        )
        threshold, _ = find_best_f1_threshold(val_y, val_prob)
        train_metrics = evaluate_predictions(train_y, train_prob, threshold)
        val_metrics = evaluate_predictions(val_y, val_prob, threshold)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "threshold": threshold,
            "train_f1": train_metrics["f1"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_accuracy": train_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"| val_f1={row['val_f1']:.4f} val_precision={row['val_precision']:.4f} "
            f"val_recall={row['val_recall']:.4f} threshold={threshold:.3f}"
        )

        if scheduler is not None:
            scheduler.step()

    return pd.DataFrame(history)


def predict_probabilities(model: nn.Module, data_loader: DataLoader, *, device: torch.device) -> np.ndarray:
    model.eval()
    all_probs = []
    with torch.inference_mode():
        for batch in data_loader:
            batch = _batch_to_device(batch, device)
            logits = forward_batch(model, batch)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs)


def predict_logits(model: nn.Module, data_loader: DataLoader, *, device: torch.device) -> np.ndarray:
    """Return raw pre-sigmoid logits for binary classification (needed for temperature scaling)."""
    model.eval()
    all_logits: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in data_loader:
            batch = _batch_to_device(batch, device)
            logits = forward_batch(model, batch)
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits)


def predict_with_tta(model: nn.Module, data_loader: DataLoader, *, device: torch.device) -> np.ndarray:
    """Predict probabilities using test-time augmentation (original + horizontal flip averaged)."""
    probs_orig = predict_probabilities(model, data_loader, device=device)
    model.eval()
    all_probs_flipped: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in data_loader:
            batch = _batch_to_device(batch, device)
            flipped: dict = {k: v for k, v in batch.items()}
            flipped["image"] = torch.flip(batch["image"], dims=[-1])
            if "fft_image" in flipped:
                flipped["fft_image"] = torch.flip(batch["fft_image"], dims=[-1])
            logits = forward_batch(model, flipped)
            all_probs_flipped.append(torch.sigmoid(logits).cpu().numpy())
    probs_flipped = np.concatenate(all_probs_flipped)
    return (probs_orig + probs_flipped) / 2.0


def build_submission(records: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    submission = records[["image_id"]].copy()
    submission["label"] = (probabilities >= threshold).astype(int)
    return submission
