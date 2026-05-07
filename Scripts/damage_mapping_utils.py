
from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def print_header(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rgb_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_label_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int64)


class DamageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment=None):
        self.df = df.reset_index(drop=True).copy()
        self.augment = augment
        required = {"image_path", "damage_mask_path"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in dataframe: {missing}")
        if len(self.df) == 0:
            raise RuntimeError("DamageDataset received an empty dataframe.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = read_rgb_image(row["image_path"])
        mask = read_label_mask(row["damage_mask_path"])

        if self.augment is not None:
            out = self.augment(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]

        image = image.astype(np.float32) / 255.0
        image = (image - IMAGE_MEAN) / IMAGE_STD
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.astype(np.int64)).long()

        return {
            "image": image,
            "mask": mask,
            "image_id": str(row.get("image_id", Path(str(row["image_path"])).stem)),
            "event_name": str(row.get("event_name", "unknown")),
            "image_path": str(row["image_path"]),
            "damage_mask_path": str(row["damage_mask_path"]),
        }


def compute_class_weights_from_manifest(
    train_df: pd.DataFrame,
    num_classes: int,
    mode: str = "median_frequency",
    background_multiplier: float = 1.0,
    clamp_max: float = 8.0,
    clamp_min: Optional[float] = None,
) -> Dict:
    counts = np.zeros(num_classes, dtype=np.float64)

    for mask_path in train_df["damage_mask_path"].tolist():
        mask = read_label_mask(mask_path)
        counts += np.bincount(mask.reshape(-1), minlength=num_classes)[:num_classes]

    total = max(float(counts.sum()), 1.0)
    freq = counts / total

    if mode == "none":
        weights = np.ones(num_classes, dtype=np.float64)
    elif mode == "inverse_frequency":
        weights = 1.0 / np.maximum(freq, 1e-12)
    elif mode == "median_frequency":
        nonzero = freq[freq > 0]
        median = float(np.median(nonzero)) if len(nonzero) else 1.0
        weights = median / np.maximum(freq, 1e-12)
    else:
        raise ValueError(f"Unknown class_weight_mode: {mode}")

    if num_classes > 0:
        weights[0] *= float(background_multiplier)

    if clamp_min is not None:
        weights = np.maximum(weights, float(clamp_min))
    if clamp_max is not None:
        weights = np.minimum(weights, float(clamp_max))

    weights = weights / max(float(weights.mean()), 1e-12)

    return {
        "mode": mode,
        "num_classes": int(num_classes),
        "class_names": CLASS_NAMES[:num_classes],
        "pixel_counts": counts.astype(int).tolist(),
        "pixel_frequencies": freq.tolist(),
        "background_multiplier": float(background_multiplier),
        "clamp_min": None if clamp_min is None else float(clamp_min),
        "clamp_max": None if clamp_max is None else float(clamp_max),
        "weights": weights.astype(float).tolist(),
    }


def confusion_matrix_from_tensors(preds: torch.Tensor, masks: torch.Tensor, num_classes: int) -> np.ndarray:
    pred_np = preds.detach().cpu().numpy().astype(np.int64)
    mask_np = masks.detach().cpu().numpy().astype(np.int64)

    valid = (mask_np >= 0) & (mask_np < num_classes)
    hist = np.bincount(
        num_classes * mask_np[valid] + pred_np[valid],
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    return hist.astype(np.int64)


def metrics_from_cm(cm: np.ndarray, exclude_background: bool = False) -> Dict:
    """
    Correct metric calculation.

    Important:
    - IoU/F1 are always computed from the full confusion matrix first.
    - exclude_background=True only changes which already-computed classes are averaged.
    - It does NOT slice cm[1:, 1:], because that would remove foreground/background
      confusions and artificially inflate foreground metrics.
    """
    cm = cm.astype(np.float64)

    tp = np.diag(cm)
    true = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    denom_iou = true + pred - tp

    iou = np.divide(tp, denom_iou, out=np.zeros_like(tp), where=denom_iou > 0)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    idx = list(range(1, len(iou))) if exclude_background else list(range(len(iou)))
    labels = [CLASS_NAMES[i] for i in idx]

    pixel_acc = float(tp.sum() / max(cm.sum(), 1.0))

    return {
        "pixel_accuracy": pixel_acc,
        "mIoU": float(np.mean(iou[idx])) if idx else 0.0,
        "macroF1": float(np.mean(f1[idx])) if idx else 0.0,
        "per_class_iou": {labels[j]: float(iou[i]) for j, i in enumerate(idx)},
        "per_class_f1": {labels[j]: float(f1[i]) for j, i in enumerate(idx)},
        "per_class_precision": {labels[j]: float(precision[i]) for j, i in enumerate(idx)},
        "per_class_recall": {labels[j]: float(recall[i]) for j, i in enumerate(idx)},
    }


def forward_segmentation_model(model, images: torch.Tensor) -> torch.Tensor:
    try:
        out = model(images)
    except TypeError:
        out = model(pixel_values=images)

    if hasattr(out, "logits"):
        out = out.logits
    elif isinstance(out, dict) and "logits" in out:
        out = out["logits"]

    if out.shape[-2:] != images.shape[-2:]:
        out = F.interpolate(out, size=images.shape[-2:], mode="bilinear", align_corners=False)

    return out


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.array([
        [0, 0, 0],
        [0, 255, 0],
        [255, 255, 0],
        [255, 165, 0],
        [255, 0, 0],
    ], dtype=np.uint8)

    mask = np.asarray(mask).astype(np.int64)
    mask = np.clip(mask, 0, len(palette) - 1)
    return palette[mask]


@torch.no_grad()
def save_prediction_examples(model, dataset, device, out_dir: str | Path, num_examples: int, prefix: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    n = min(int(num_examples), len(dataset))
    if n <= 0:
        return 0

    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    saved = 0

    for k, idx in enumerate(indices):
        sample = dataset[int(idx)]
        image_t = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].cpu().numpy().astype(np.int64)

        logits = forward_segmentation_model(model, image_t)
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.int64)

        img = sample["image"].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img * IMAGE_STD + IMAGE_MEAN)
        img = np.clip(img, 0, 1)

        gt_color = colorize_mask(mask)
        pred_color = colorize_mask(pred)
        overlay = (0.65 * (img * 255.0) + 0.35 * pred_color).clip(0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img)
        axes[0].set_title("image")
        axes[1].imshow(gt_color)
        axes[1].set_title("ground truth")
        axes[2].imshow(pred_color)
        axes[2].set_title("prediction")
        axes[3].imshow(overlay)
        axes[3].set_title("prediction overlay")

        for ax in axes:
            ax.axis("off")

        title = str(sample.get("image_id", f"sample_{idx}"))
        fig.suptitle(title)
        fig.tight_layout()

        safe_title = title.replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"{prefix}_{k:02d}_{safe_title}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    return saved
