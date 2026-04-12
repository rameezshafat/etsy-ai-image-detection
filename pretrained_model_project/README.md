# Pretrained Model Project

Script conversion of `notebooks/02_pretrained_finetuning_ai_detection.ipynb`.

## What it does

Trains a multi-model ensemble for AI-generated image detection:

1. **EfficientNet-B0 K-Fold** — 3-fold cross-validation with two-phase fine-tuning:
   - Phase 1: backbone frozen, head-only warmup (3 epochs, LR=1e-3)
   - Phase 2: full unfreeze with discriminative LRs (2 epochs, backbone LR=3e-5)
2. **FFT dual-branch** — second EfficientNet-B0 branch on frequency-domain images (fold-0 only)
3. **ViT-B/16** — vision transformer with the same two-phase strategy (fold-0 only)
4. **Ensemble** — TTA-averaged CNN folds + optional FFT/ViT (included if within 1% F1 of CNN)

## Dataset layout

Expected at repo root (not inside this folder):

```
Data/
  train.csv
  test.csv
  images_final_sample/
artifacts/           ← outputs written here
src/
  ai_image_detection/
```

## Run

```bash
# From repo root:
pip install -r pretrained_model_project/requirements.txt
python pretrained_model_project/main.py

# Or from inside the project folder:
cd pretrained_model_project
pip install -r requirements.txt
python main.py
```

## Outputs

All written to `artifacts/` at the repo root:

| File | Description |
|------|-------------|
| `submission_v2_ensemble.csv` | Final submission with predicted labels |
| `pretrained_metrics_summary.json` | OOF F1, accuracy, precision, recall, threshold |
| `pretrained_training_history.csv` | Per-epoch train/val loss and F1 for all folds |
| `pretrained_model_comparison.csv` | OOF vs fold-0 vs FFT vs ViT comparison table |
| `pretrained_robustness.csv` | F1 under clean / JPEG / blur perturbations |
| `pretrained_learning_curves.png` | Train/val F1 per fold with phase boundary |
| `pretrained_loss_curves.png` | Train/val loss per fold |
| `pretrained_oof_confusion_matrix.png` | Confusion matrix on OOF predictions |

## Module structure

| File | Responsibility |
|------|---------------|
| `config.py` | All hyperparameters and paths |
| `data_loader.py` | Metadata loading and DataLoader construction |
| `models.py` | Model factory (CNN / FFT / ViT) |
| `train.py` | K-Fold, FFT, and ViT training loops |
| `evaluate.py` | OOF metrics, model comparison, robustness, error analysis |
| `inference.py` | Ensemble construction and submission CSV |
| `utils.py` | Plotting and metric serialisation |
| `main.py` | Orchestration entry point |

## Tuning

All hyperparameters are in `config.py`. Key settings:

```python
N_FOLDS = 3
PHASE1_EPOCHS = 3      # head-only warmup
PHASE2_EPOCHS = 2      # discriminative-LR fine-tuning
INCLUDE_THRESHOLD = 0.01  # include FFT/ViT if fold-0 F1 >= CNN - this value
SEED = 42
```

## Differences from notebook

- `display()` calls replaced with `print(...to_string())` for script compatibility
- Matplotlib uses the `Agg` backend (no GUI required)
- Sample-image visualisation cells omitted (no display context available)
- Misclassified-example image grids omitted (same reason)
- PCA probability-distribution plot omitted
- All other numeric outputs and artifact files are reproduced faithfully
