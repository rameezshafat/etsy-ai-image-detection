# Notebook Explanation: Pretrained Model Fine-Tuning for AI-Generated Image Detection

---

## High-Level Context

**Goal:** Binary classification — real vs. AI-generated images — optimized for **F1 score**, not accuracy.

**Why F1?** Class imbalance or asymmetric detection costs make accuracy a misleading metric. F1 balances precision and recall, which is what you actually care about in a detector.

**Problem being solved:** AI-generated images may differ from real ones in multiple complementary signal spaces — spatial texture artifacts, frequency-domain patterns, and high-level semantic features. No single model family dominates all three. This notebook systematically compares them.

---

## Section-by-Section Breakdown

### 1. Project Setup

- Dynamically resolves the project root (handles both `notebooks/` and root working directories).
- Adds `src/` to `sys.path` so internal modules (`ai_image_detection.*`) are importable without installation.
- Separates `DATA_DIR`, `ARTIFACTS_DIR`, and source directories explicitly — a clean reproducibility practice.

**Why this matters:** Jupyter notebooks are notoriously fragile about working directories. Explicit path resolution prevents silent data-loading failures.

---

### 2. Data Loading and Split

- Loads metadata CSVs, filters to available images only.
- Stratified 80/20 train/val split via `create_train_val_split` (uses `random_state=42` for reproducibility).
- Prints class distribution to surface any imbalance early.

**Design choice:** Filtering unavailable images before splitting avoids leaking knowledge about missing data into the val set.

---

### 3. Data Pipeline and Augmentations

| Split | Transforms |
|---|---|
| Train | Resize → RandomCrop → HorizontalFlip → ColorJitter → Normalize |
| Val/Test | Resize → CenterCrop/Resize → Normalize |

- Uses **ImageNet normalization** (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) because all three pretrained backbones were trained on ImageNet.
- Augmentations are train-only — val/test paths are deterministic to ensure reproducible evaluation.

**Trade-off:** ColorJitter modifies image statistics; aggressive jitter could wash out subtle AI artifacts. It is kept mild here intentionally.

---

### 4. F1-Driven Training Strategy

```
Optimizer:  AdamW (weight_decay=1e-4)
Scheduler:  CosineAnnealingLR
Loss:       BCEWithLogitsLoss
Metric:     Validation F1 (not accuracy)
Threshold:  Swept in [0.3, 0.7] at 81 steps → pick argmax(F1)
```

**Why threshold search?** The sigmoid output is a probability, not a binary label. The default `0.5` cutoff is arbitrary. By sweeping and picking the threshold that maximizes val-F1, the model's decision boundary is tuned to the actual objective. This is equivalent to calibrating the classifier post-training.

**Why CosineAnnealingLR?** It smoothly decays the learning rate to near-zero, avoiding abrupt drops that can destabilize fine-tuning of pretrained weights.

---

### 5. Model 1: Artifact CNN (EfficientNet-B0)

**Architecture:** Pretrained EfficientNet-B0 with the classification head replaced by a linear layer for binary output.

**Training strategy:** Two-phase
1. **Freeze backbone** → train only the new head (fast convergence, low risk of destroying pretrained features).
2. *(Optional)* Unfreeze deeper layers for fine-grained adaptation.

**Why EfficientNet-B0?**
- Compound scaling makes it parameter-efficient and fast.
- Strong ImageNet features generalize well to texture/artifact detection.
- B0 is the smallest variant — suitable for limited GPU environments.

**What it captures:** Local texture inconsistencies, compression artifacts, and spatial irregularities that GAN/diffusion models leave behind.

---

### 6. Model 2: Spectrum Dual-Branch (FFT + RGB)

**Architecture:** `FFTDualBranchClassifier` — two parallel EfficientNet-B0 backbones:

```
Image (RGB)      → EfficientNet-B0 → feature vector
FFT Spectrum     → EfficientNet-B0 → feature vector
                                    ↓
              Concat → Linear(512) → ReLU → Dropout(0.3) → Linear(1)
```

**How the FFT image is made:**
```python
gray = image.convert("L")                       # grayscale
spectrum = np.abs(np.fft.fft2(gray))            # 2D FFT
spectrum = np.fft.fftshift(spectrum)            # center low freqs
spectrum = np.log1p(spectrum)                   # compress dynamic range
spectrum = normalize to [0, 255]               # treat as image
```

The log-magnitude spectrum highlights periodic patterns (grid artifacts, checkerboard noise) that are invisible in the spatial domain — a well-known fingerprint of upsampling layers in GANs and diffusion models.

**Trade-off:** Doubles the input channels and model parameters. Training is slower and more prone to overfitting without sufficient data.

---

### 7. Model 3: Vision Transformer (ViT-B/16)

**Architecture:** `EmbeddingClassifier` — pretrained ViT-B/16 backbone with a frozen transformer body and a trainable head:

```
Image → ViT-B/16 [CLS token] → Linear(512) → ReLU → Dropout(0.3) → Linear(1)
```

**Why ViT?**
- Transformers learn global, attention-based features rather than local convolutional filters.
- The hypothesis: high-level semantic structure may generalize better to unseen generation styles (e.g., new diffusion models not seen during training).

**Trade-off:** ViT is significantly larger than EfficientNet-B0. Fine-tuning only the head with 3 epochs may underfit; deeper unfreezing risks catastrophic forgetting.

---

### 8. Robustness Checks Under Perturbations

Tests each model under:
- **JPEG compression** (lossy codec changes pixel statistics)
- **Gaussian blur** (removes high-frequency detail)

**Why this matters:** A model that relies heavily on compression artifacts or pixel-level noise will degrade under JPEG re-encoding or blurring — common operations in real-world image pipelines. The perturbation delta reveals the model's fragility.

**Design:** Perturbations are applied at dataset load time via `FineTuneImageDataset(perturbation=...)`, keeping the evaluation clean and reproducible.

---

### 9. Optional Handcrafted Feature Fusion

Extracts image-level statistics:
- `rgb_mean_r/g/b` — channel means
- `laplacian_variance` — sharpness proxy
- `fft_high_freq_ratio` — high-frequency energy fraction
- `exif_present` — metadata availability flag
- `file_size_kb` — compression behavior signal

These are concatenated with deep features before the final classifier:

```
[deep features | handcrafted features] → Linear → binary output
```

**Why?** The profiling notebook (notebook 01) identified these as separating signals. This section operationalizes those findings. The `include_handcrafted=True` flag is off by default — it is experimental infrastructure, not a production default.

---

### 10. Model Comparison

A unified table of val-F1, precision, recall, accuracy, and optimal threshold across all three model families — enabling direct, apples-to-apples comparison.

---

### 11. Failure Analysis

Categorizes validation errors into:
- **False positives** — real images flagged as AI-generated
- **False negatives** — AI-generated images missed as real

Visualizes misclassified examples side-by-side. This is where model diagnostics become actionable — patterns in errors often point to data artifacts, augmentation gaps, or model blind spots.

---

### 12. Feature-Space Visualization

Uses PCA on the model's probability scores (and optionally handcrafted features) to produce a 1D distribution plot overlaid by class label.

**Limitation:** PCA on a single scalar (`prob_ai`) is trivial — it is just a sorted projection. This is lightweight by design for notebook runtime. In a production context you would extract penultimate-layer embeddings and apply t-SNE or UMAP for meaningful geometry.

---

### 13. Final Test Predictions

- Selects the best model by validation F1 from the comparison table.
- Runs inference on the held-out test set.
- Outputs `submission.csv` with columns `image_id` and `label`.

The best model and loader are selected programmatically — no hardcoding — which makes the notebook rerunnable with different results automatically reflected in the submission.

---

## Key Concepts and Design Choices

| Concept | Implementation | Rationale |
|---|---|---|
| Transfer learning | Pretrained ImageNet weights | Saves data; ImageNet features generalize |
| Head-first training | `freeze_backbone` → train head | Stabilizes early epochs |
| FFT spectrum | Log-magnitude, centered, normalized | Detects frequency artifacts invisible spatially |
| Threshold sweep | 81 points in [0.3, 0.7] | Decouples threshold from training |
| Robustness eval | JPEG + blur perturbations | Tests real-world distribution shift |
| Modular dataset | `include_fft_image`, `include_handcrafted` flags | Supports all three model families from one class |

---

## Alternatives

### Alternative Backbones
| Option | Pros | Cons |
|---|---|---|
| ResNet-50 | Simpler, well-understood | Weaker than EfficientNet at same size |
| ConvNeXt | Strong SOTA CNN | Heavier than B0 |
| CLIP image encoder | Zero-shot generalization | Requires access to larger model |
| DINOv2 | Self-supervised, no label bias | Harder to integrate |

### Alternative Frequency Analysis
- **Wavelet decomposition** instead of FFT: multi-scale frequency bands, better spatial localization.
- **DCT coefficients** (used in JPEG): directly models compression artifacts.
- **Patch-level spectrum**: per-patch FFT to preserve spatial frequency structure.

### Alternative Fusion Strategies
- **Attention-based fusion**: learn which branch (RGB vs FFT) to trust per image.
- **Late fusion with ensemble voting**: train branches independently, combine at decision time.
- **Early fusion**: stack FFT as a 4th channel input — simpler but loses independent feature learning.

### Alternative Threshold Strategy
- **Precision-recall curve** + F1-maximizing threshold (equivalent, more standard).
- **Cost-sensitive threshold**: set from false-positive vs. false-negative cost ratio for deployment.

---

## Practical Takeaway

**How to think about applying or modifying this notebook:**

1. **Start with the artifact CNN (Model 1)** — it is the simplest, fastest, and most often the strongest baseline. Run it first.

2. **Add the spectrum branch (Model 2) only if the CNN underperforms on frequency-artifact-heavy generators** (e.g., GAN outputs with upsampling checkerboard patterns). Expect it to win on GAN images, lose on diffusion images.

3. **Use ViT (Model 3) if you have enough data and compute**, or if you expect the deployment distribution to include generation methods not seen during training. ViT generalizes at the cost of data hunger.

4. **Always run the robustness check** before trusting val-F1. A model that degrades 15+ F1 points under JPEG compression is not production-ready.

5. **Threshold tuning is not optional** — sweep it on val, not test. Using 0.5 blindly on a skewed detection problem is a common mistake.

6. **The handcrafted fusion section is exploratory** — run it only after you have a strong deep baseline. Handcrafted features rarely help when the deep model already has sufficient capacity and data.
