"""Centralized configuration: hyperparameters and file paths.

All tunable constants live here. Import this module in other files;
do not hard-code values elsewhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent  # repo root

# Add src/ to path so ai_image_detection is importable
_SRC = PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DATA_DIR = PROJECT_ROOT / "Data"
IMAGE_DIR = DATA_DIR / "images_final_sample"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ── Data ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

# ── Cross-validation ─────────────────────────────────────────────────────────
N_FOLDS = 3

# ── Training — Phase 1 (head warmup, backbone frozen) ────────────────────────
PHASE1_EPOCHS = 3
HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 1e-4

# ── Training — Phase 2 (discriminative-LR fine-tuning) ───────────────────────
PHASE2_EPOCHS = 2
PHASE2_HEAD_LR = 3e-4
BACKBONE_LR_FACTOR = 0.1  # backbone gets 10x lower LR than head

# ── Ensemble ──────────────────────────────────────────────────────────────────
# Include FFT/ViT if fold-0 F1 >= CNN fold-0 F1 - INCLUDE_THRESHOLD
INCLUDE_THRESHOLD = 0.01

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
