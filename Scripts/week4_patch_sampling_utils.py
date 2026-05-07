from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if mask.ndim != 2:
        raise ValueError(f"Expected single-channel mask, got {mask.shape} for {path}")
    return mask.astype(np.uint8)


def compute_class_weights_from_manifest(
    train_df: pd.DataFrame,
    num_classes: int,
    mode: str = "median_frequency",
    background_multiplier: float = 0.25,
    clamp_max: float = 8.0,
) -> Dict:
    counts = np.zeros(num_classes, dtype=np.int64)
    for mask_path_str in train_df["damage_mask_path"].tolist():
        mask = load_mask(Path(mask_path_str))
        binc = np.bincount(mask.reshape(-1), minlength=num_classes)
        counts += binc[:num_classes]

    total = counts.sum()
    freqs = counts.astype(np.float64) / max(total, 1)
    safe = np.clip(freqs, 1e-12, None)

    if mode == "median_frequency":
        nz = safe[safe > 1e-12]
        median_freq = float(np.median(nz)) if len(nz) else 1.0
        weights = median_freq / safe
    elif mode == "inverse_frequency":
        weights = 1.0 / safe
    elif mode == "inverse_sqrt_frequency":
        weights = 1.0 / np.sqrt(safe)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    weights = np.clip(weights, 0.0, clamp_max)
    weights[0] *= background_multiplier
    weights = weights / np.mean(weights)

    return {
        "mode": mode,
        "pixel_counts": counts.tolist(),
        "pixel_frequencies": freqs.tolist(),
        "weights": weights.tolist(),
        "background_multiplier": background_multiplier,
        "clamp_max": clamp_max,
    }


def confusion_matrix_from_tensors(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    pred = pred.reshape(-1).to(torch.int64)
    target = target.reshape(-1).to(torch.int64)

    valid = (target >= 0) & (target < num_classes)
    pred = pred[valid]
    target = target[valid]

    idx = target * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.cpu().numpy()


def metrics_from_cm(cm: np.ndarray, exclude_background: bool = False) -> Dict[str, float]:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    denom_iou = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn

    iou = np.divide(tp, denom_iou, out=np.zeros_like(tp), where=denom_iou > 0)
    f1 = np.divide(2 * tp, denom_f1, out=np.zeros_like(tp), where=denom_f1 > 0)

    start = 1 if exclude_background else 0
    return {
        "pixel_accuracy": float(tp.sum() / np.clip(cm.sum(), 1.0, None)),
        "mIoU": float(iou[start:].mean()),
        "macroF1": float(f1[start:].mean()),
    }


def _choose_crop_origin(center_y: int, center_x: int, crop_h: int, crop_w: int, h: int, w: int) -> Tuple[int, int]:
    y0 = int(center_y - crop_h // 2)
    x0 = int(center_x - crop_w // 2)
    y0 = max(0, min(y0, h - crop_h))
    x0 = max(0, min(x0, w - crop_w))
    return y0, x0


def _random_crop_origin(crop_h: int, crop_w: int, h: int, w: int) -> Tuple[int, int]:
    y0 = 0 if h == crop_h else np.random.randint(0, h - crop_h + 1)
    x0 = 0 if w == crop_w else np.random.randint(0, w - crop_w + 1)
    return int(y0), int(x0)


def sample_patch(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    building_center_prob: float = 0.70,
    rare_class_focus_prob: float = 0.40,
    rare_classes: Tuple[int, ...] = (2, 3, 4),
    non_empty_retry: int = 8,
):
    h, w = mask.shape
    crop_h = min(crop_size, h)
    crop_w = min(crop_size, w)

    fg_coords = np.argwhere(mask > 0)
    rare_coords = np.argwhere(np.isin(mask, np.array(rare_classes, dtype=np.uint8)))
    mode = "random"

    r = np.random.rand()
    if len(fg_coords) > 0 and r < building_center_prob:
        if len(rare_coords) > 0 and np.random.rand() < rare_class_focus_prob:
            coords = rare_coords
            mode = "rare_center"
        else:
            coords = fg_coords
            mode = "building_center"
        cy, cx = coords[np.random.randint(0, len(coords))]
        y0, x0 = _choose_crop_origin(int(cy), int(cx), crop_h, crop_w, h, w)
    else:
        found = False
        for _ in range(non_empty_retry):
            y0, x0 = _random_crop_origin(crop_h, crop_w, h, w)
            patch_mask = mask[y0:y0 + crop_h, x0:x0 + crop_w]
            if np.any(patch_mask > 0):
                mode = "non_empty_random"
                found = True
                break
        if not found:
            y0, x0 = _random_crop_origin(crop_h, crop_w, h, w)

    patch_img = image[y0:y0 + crop_h, x0:x0 + crop_w]
    patch_mask = mask[y0:y0 + crop_h, x0:x0 + crop_w]
    info = {
        "mode": mode,
        "top": int(y0),
        "left": int(x0),
        "has_foreground": bool(np.any(patch_mask > 0)),
        "has_rare": bool(np.any(np.isin(patch_mask, np.array(rare_classes, dtype=np.uint8)))),
    }
    return patch_img, patch_mask, info


def simple_train_augment(image: np.ndarray, mask: np.ndarray):
    if np.random.rand() < 0.5:
        image = np.ascontiguousarray(np.flip(image, axis=1))
        mask = np.ascontiguousarray(np.flip(mask, axis=1))
    if np.random.rand() < 0.5:
        image = np.ascontiguousarray(np.flip(image, axis=0))
        mask = np.ascontiguousarray(np.flip(mask, axis=0))

    k = np.random.randint(0, 4)
    if k > 0:
        image = np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))
        mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))

    if np.random.rand() < 0.3:
        alpha = 1.0 + np.random.uniform(-0.15, 0.15)
        beta = np.random.uniform(-12.0, 12.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return image, mask


def resize_pair(image: np.ndarray, mask: np.ndarray, size: int):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return image, mask


class FullImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = load_rgb(Path(row["image_path"]))
        mask = load_mask(Path(row["damage_mask_path"]))
        image, mask = resize_pair(image, mask, self.image_size)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.long),
            "image_path": str(row["image_path"]),
            "mask_path": str(row["damage_mask_path"]),
        }


class PatchSamplingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        crop_size: int,
        epoch_multiplier: int = 2,
        building_center_prob: float = 0.70,
        rare_class_focus_prob: float = 0.40,
        rare_classes: Tuple[int, ...] = (2, 3, 4),
        use_train_aug: bool = True,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.crop_size = int(crop_size)
        self.epoch_multiplier = int(epoch_multiplier)
        self.building_center_prob = float(building_center_prob)
        self.rare_class_focus_prob = float(rare_class_focus_prob)
        self.rare_classes = tuple(rare_classes)
        self.use_train_aug = bool(use_train_aug)
        self.mode_counter = {"rare_center": 0, "building_center": 0, "non_empty_random": 0, "random": 0}

    def __len__(self) -> int:
        return len(self.df) * max(self.epoch_multiplier, 1)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx % len(self.df)]
        image = load_rgb(Path(row["image_path"]))
        mask = load_mask(Path(row["damage_mask_path"]))

        image, mask, info = sample_patch(
            image=image,
            mask=mask,
            crop_size=self.crop_size,
            building_center_prob=self.building_center_prob,
            rare_class_focus_prob=self.rare_class_focus_prob,
            rare_classes=self.rare_classes,
        )
        self.mode_counter[info["mode"]] = self.mode_counter.get(info["mode"], 0) + 1

        if self.use_train_aug:
            image, mask = simple_train_augment(image, mask)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.long),
            "image_path": str(row["image_path"]),
            "mask_path": str(row["damage_mask_path"]),
            "sample_mode": info["mode"],
        }


def save_prediction_examples(model, dataset, device: torch.device, out_dir: Path, num_examples: int = 6) -> None:
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for idx in range(min(num_examples, len(dataset))):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["mask"].cpu().numpy()
        image_vis = sample["image"].permute(1, 2, 0).cpu().numpy()

        with torch.no_grad():
            outputs = model(pixel_values=image)
            logits = F.interpolate(outputs.logits, size=gt_mask.shape, mode="bilinear", align_corners=False)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(np.clip(image_vis, 0, 1))
        axes[0].set_title("image")
        axes[0].axis("off")
        axes[1].imshow(gt_mask, cmap="tab10", vmin=0, vmax=4)
        axes[1].set_title("ground truth")
        axes[1].axis("off")
        axes[2].imshow(pred, cmap="tab10", vmin=0, vmax=4)
        axes[2].set_title("prediction")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"example_{idx:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
