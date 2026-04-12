"""Model factory functions wrapping src/ai_image_detection/pretrained.py."""
from __future__ import annotations

import config  # noqa: F401 — side-effect: registers src/ on sys.path
from ai_image_detection.pretrained import (
    ModelBundle,
    build_artifact_model,
    build_encoder_model,
    build_spectrum_model,
)


def make_cnn_model(pretrained: bool = True) -> ModelBundle:
    """EfficientNet-B0 with classification head replaced by identity + linear."""
    return build_artifact_model(model_name="efficientnet_b0", pretrained=pretrained)


def make_fft_model(pretrained: bool = True) -> ModelBundle:
    """Dual-branch EfficientNet-B0: one branch for RGB, one for FFT magnitude."""
    return build_spectrum_model(model_name="efficientnet_b0", pretrained=pretrained)


def make_vit_model(pretrained: bool = True) -> ModelBundle:
    """ViT-B/16 with classification head replaced by identity + linear."""
    return build_encoder_model(model_name="vit_b_16", pretrained=pretrained)
