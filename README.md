# ECE6364 — Building Damage Assessment from Satellite Imagery

Semantic segmentation of post-disaster satellite imagery for building damage assessment. Compares two architectures (U-Net, SegFormer) and two training strategies (weighted cross-entropy, balanced patch sampling) across a 5-fold cross-validation scheme with event-level held-out generalization evaluation.

---

## Table of Contents

- [Dataset](#dataset)
- [Folder Structure](#folder-structure)
- [Environment Setup](#environment-setup)
- [Workflow](#workflow)
- [Scripts Reference](#scripts-reference)
- [Output Structure](#output-structure)

---

## Dataset

The project uses the **xView2 Building Damage Assessment** dataset (XBD), consisting of post-disaster satellite imagery paired with polygon-annotated building damage labels.

**10 disaster events:**

| Event | Train Images | Test Images |
|---|---|---|
| socal-fire | 823 | 307 |
| hurricane-florence | 319 | 108 |
| hurricane-harvey | 319 | 108 |
| hurricane-michael | 343 | 98 |
| midwest-flooding | 279 | 80 |
| santa-rosa-wildfire | 226 | 74 |
| hurricane-matthew | 238 | 73 |
| palu-tsunami | 113 | 42 |
| mexico-earthquake | 121 | 38 |
| guatemala-volcano | 18 | 5 |
| **Total** | **2,799** | **933** |

**Damage classes (5):**

| ID | Class |
|---|---|
| 0 | Background |
| 1 | No-damage |
| 2 | Minor-damage |
| 3 | Major-damage |
| 4 | Destroyed |

Images are 1024×1024 PNG with per-building polygon annotations in JSON/WKT format.

---

## Folder Structure

```
ECE6364/
├── Data/
│   ├── train/
│   │   ├── images/          # Pre/post-disaster satellite images
│   │   ├── labels/          # Per-building JSON annotations (WKT polygons)
│   │   ├── targets/         # Binary target masks (generated)
│   │   └── damage_masks/    # Multi-class damage masks (generated)
│   └── test/
│       ├── images/
│       ├── labels/
│       ├── targets/
│       └── damage_masks/
│
├── Scripts/
│   ├── week1_trainfolder_testfolder_seen_unseen_split.py
│   ├── week2_train_unet_weighted_ce.py
│   ├── week2_test_unet_weighted_ce.py
│   ├── week3_train_segformer_weighted_ce.py
│   ├── week3_test_segformer_weighted_ce.py
│   ├── week4_train_unet_patch_sampling_seen_unseen.py
│   ├── week4_test_unet_patch_sampling.py
│   ├── week4_train_segformer_patch_sampling_seen_unseen.py
│   ├── week4_test_segformer_patch_sampling.py
│   ├── damage_mapping_utils.py
│   ├── week4_patch_sampling_utils.py
│   ├── week4_balanced_patch_sampling_utils.py
│   ├── overall_confusion_matrix.py
│   ├── class_distribution.py
│   ├── sanity_check.py
│   ├── plot_results_summary.py
│   ├── plot_pixel_metrics.py
│   ├── plot_building_metrics.py
│   ├── plot_confusion_matrices.py
│   └── plot_building_confusion_matrices.py
│
└── Outputs/
    ├── week1_trainfolder_testfolder_seen_unseen_split/
    ├── week2_train_unet_weighted_ce/
    ├── week2_test_unet_weighted_ce/
    ├── week3_train_segformer_weighted_ce/
    ├── week3_test_segformer_weighted_ce/
    ├── week4_train_unet_patch_sampling/
    ├── week4_test_unet_patch_sampling/
    ├── week4_train_segformer_patch_sampling/
    └── week4_test_segformer_patch_sampling/
```

---

## Environment Setup

### Requirements

| Package | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `transformers` | SegFormer architecture (Hugging Face) |
| `segmentation-models-pytorch` | U-Net architecture |
| `albumentations` | Data augmentation |
| `opencv-python` | Image I/O and preprocessing |
| `shapely` | WKT polygon parsing for building annotations |
| `pandas` | Manifest/CSV handling |
| `numpy` | Numerical operations |
| `scikit-learn` | Train/val splitting utilities |
| `matplotlib` | Metrics and confusion matrix visualization |

### Conda (recommended)

```bash
conda create -n ece6364 python=3.10
conda activate ece6364
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers segmentation-models-pytorch albumentations \
            opencv-python shapely pandas numpy scikit-learn matplotlib
```

### pip

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers segmentation-models-pytorch albumentations \
            opencv-python shapely pandas numpy scikit-learn matplotlib
```

### Verify GPU

```python
import torch
print(torch.cuda.is_available())   # should be True
print(torch.cuda.get_device_name(0))
```

---

## Workflow

Run the stages in order. Each stage depends on outputs from the previous one.

### Stage 1 — Data Preparation

Generates per-pixel damage masks from JSON polygon annotations and creates 5-fold cross-validation splits with event-level holdout.

```bash
cd Scripts
python week1_trainfolder_testfolder_seen_unseen_split.py
```

**Output:** `Outputs/week1_trainfolder_testfolder_seen_unseen_split/`
- `fold_0*/train.csv`, `val.csv`, `seen_test.csv`, `unseen_test.csv`
- `week1_trainfolder_testfolder_seen_unseen_summary.json`

**Fold holdout design:**

| Fold | Held-out events (unseen) |
|---|---|
| 0 | socal-fire, guatemala-volcano |
| 1 | hurricane-michael, palu-tsunami |
| 2 | hurricane-harvey, mexico-earthquake |
| 3 | hurricane-florence, santa-rosa-wildfire |
| 4 | midwest-flooding, hurricane-matthew |

### Stage 2 — Training

Train all 4 model variants across 5 folds. Each script iterates all folds automatically.

```bash
# U-Net with weighted cross-entropy
python week2_train_unet_weighted_ce.py

# SegFormer with weighted cross-entropy
python week3_train_segformer_weighted_ce.py

# U-Net with balanced patch sampling
python week4_train_unet_patch_sampling_seen_unseen.py

# SegFormer with balanced patch sampling
python week4_train_segformer_patch_sampling_seen_unseen.py
```

### Stage 3 — Evaluation

Evaluate saved checkpoints on seen and unseen test splits. Computes both pixel-level and building-level metrics.

```bash
python week2_test_unet_weighted_ce.py
python week3_test_segformer_weighted_ce.py
python week4_test_unet_patch_sampling.py
python week4_test_segformer_patch_sampling.py
```

### Stage 4 — Visualization

Generate all result plots and confusion matrix figures.

```bash
python overall_confusion_matrix.py
python plot_results_summary.py
python plot_pixel_metrics.py
python plot_building_metrics.py
python plot_confusion_matrices.py
python plot_building_confusion_matrices.py
```

---

## Scripts Reference

### Data & Utilities

| Script | Description |
|---|---|
| `week1_trainfolder_testfolder_seen_unseen_split.py` | Generates damage masks from JSON polygon annotations; creates 5-fold event-holdout splits with `train/val/seen_test/unseen_test` CSVs |
| `damage_mapping_utils.py` | Shared utilities: `DamageDataset`, ImageNet normalization, confusion matrix computation, `metrics_from_cm`, class weight estimation, prediction visualization |
| `week4_patch_sampling_utils.py` | Patch extraction utilities: sliding window sampling with damage-class filtering and validation-set handling |
| `week4_balanced_patch_sampling_utils.py` | `BalancedPatchSamplingDataset` — oversamples patches containing rare damage classes during training |
| `class_distribution.py` | Computes and plots pixel-level class frequency across the training set |
| `sanity_check.py` | Validates that train/val/test splits have no image overlap across folds |

### Training

| Script | Architecture | Loss Strategy |
|---|---|---|
| `week2_train_unet_weighted_ce.py` | U-Net (ResNet-34 encoder) | Weighted cross-entropy |
| `week3_train_segformer_weighted_ce.py` | SegFormer-B2 | Weighted cross-entropy |
| `week4_train_unet_patch_sampling_seen_unseen.py` | U-Net (ResNet-34 encoder) | CE + balanced patch sampling |
| `week4_train_segformer_patch_sampling_seen_unseen.py` | SegFormer-B2 | CE + balanced patch sampling |

All training scripts: 20 epochs, LR 1e-3, batch size 4, 512×512 crops, best checkpoint saved by validation mIoU.

### Evaluation

| Script | Description |
|---|---|
| `week2_test_unet_weighted_ce.py` | Evaluates U-Net (weighted CE) on seen and unseen test splits; saves pixel and building confusion matrices and metrics JSON |
| `week3_test_segformer_weighted_ce.py` | Same for SegFormer (weighted CE) |
| `week4_test_unet_patch_sampling.py` | Same for U-Net (patch sampling); upscales patch predictions back to 1024×1024 for building-level evaluation |
| `week4_test_segformer_patch_sampling.py` | Same for SegFormer (patch sampling) |

Each test script produces per-fold: pixel metrics, building metrics, confusion matrices (CSV), per-class summaries, and prediction visualizations.

### Visualization

| Script | Output |
|---|---|
| `overall_confusion_matrix.py` | Aggregates raw CMs across folds; saves normalized CSVs for pixel and building levels |
| `plot_results_summary.py` | Bar chart comparing mean pixel mIoU and building macro-F1 (seen vs unseen) across all 4 models |
| `plot_pixel_metrics.py` | Per-model/per-class pixel IoU and macro-F1 bar charts with fold std dev |
| `plot_building_metrics.py` | Per-model/per-class building F1 bar charts (damage classes only, no background) |
| `plot_confusion_matrices.py` | 2×4 grid of row-normalized pixel confusion matrices (row = seen/unseen, col = model) |
| `plot_building_confusion_matrices.py` | 2×4 grid of row-normalized building confusion matrices (damage classes only) |

---

## Output Structure

Each training and testing run saves results organized by fold:

```
Outputs/week2_train_unet_weighted_ce/
└── fold_00_holdout_socal-fire__guatemala-volcano/
    ├── checkpoints/
    │   └── best_model.pt
    ├── config.json
    ├── class_weights.json
    └── train_history.csv

Outputs/week2_test_unet_weighted_ce/
└── fold_00_holdout_socal-fire__guatemala-volcano/
    ├── seen_test_pixel_confusion_matrix.csv
    ├── seen_test_building_confusion_matrix.csv
    ├── seen_test_pixel_metrics.json
    ├── seen_test_building_metrics.json
    ├── seen_test_pixel_per_class_metrics.csv
    ├── seen_test_building_per_class_metrics.csv
    ├── unseen_test_pixel_confusion_matrix.csv
    ├── unseen_test_building_confusion_matrix.csv
    ├── unseen_test_pixel_metrics.json
    ├── unseen_test_building_metrics.json
    └── predictions/

Outputs/week2_test_unet_weighted_ce/
└── all_folds_seen_unseen_damage_class_summary.csv   # aggregated across folds
```

**Key metrics reported:**

- **Pixel-level:** mIoU, macro-F1, per-class IoU/F1/precision/recall (with and without background)
- **Building-level:** per-building accuracy, F1, precision per damage class; overall macro-F1
