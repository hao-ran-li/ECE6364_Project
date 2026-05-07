from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from damage_mapping_utils import IMAGE_MEAN, IMAGE_STD, read_rgb_image, read_label_mask


CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]


def _normalize_image(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def _to_mask_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.int64)).long()


def _resize_pair(image: np.ndarray, mask: np.ndarray, size: int):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def _crop_with_padding(image: np.ndarray, mask: np.ndarray, cx: int, cy: int, crop_size: int):
    h, w = mask.shape[:2]
    half = crop_size // 2

    x1 = int(cx) - half
    y1 = int(cy) - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.copyMakeBorder(
            mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    if crop_img.shape[0] != crop_size or crop_img.shape[1] != crop_size:
        crop_img = cv2.resize(crop_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        crop_mask = cv2.resize(crop_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    return crop_img, crop_mask


def _random_center(mask: np.ndarray, crop_size: int, rng: np.random.Generator):
    h, w = mask.shape[:2]
    if w <= crop_size:
        cx = w // 2
    else:
        cx = int(rng.integers(crop_size // 2, w - crop_size // 2))
    if h <= crop_size:
        cy = h // 2
    else:
        cy = int(rng.integers(crop_size // 2, h - crop_size // 2))
    return cx, cy


def _sample_class_center(mask: np.ndarray, classes: Sequence[int], rng: np.random.Generator):
    valid = np.isin(mask, np.array(list(classes), dtype=np.int64))
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return None
    idx = int(rng.integers(0, len(xs)))
    return int(xs[idx]), int(ys[idx])


def _apply_light_train_aug(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    # Horizontal flip
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    # Vertical flip
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])
    # RandomRotate90 — mask rotated identically
    k = int(rng.integers(0, 4))
    if k > 0:
        image = np.ascontiguousarray(np.rot90(image, k))
        mask = np.ascontiguousarray(np.rot90(mask, k))
    # Brightness/contrast jitter — image only, helps cross-event generalization
    if rng.random() < 0.3:
        brightness = rng.uniform(0.7, 1.3)
        image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        mean = image.mean()
        contrast = rng.uniform(0.7, 1.3)
        image = np.clip((image.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)
    return image, mask


class FullImageDataset(Dataset):
    """
    Full-image evaluation/validation dataset.
    Resizes image and mask to image_size.
    """

    def __init__(self, df: pd.DataFrame, image_size: int = 512):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = int(image_size)

        required = {"image_path", "damage_mask_path"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"FullImageDataset missing columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = read_rgb_image(row["image_path"])
        mask = read_label_mask(row["damage_mask_path"])

        image, mask = _resize_pair(image, mask, self.image_size)

        return {
            "image": _normalize_image(image),
            "mask": _to_mask_tensor(mask),
            "image_id": str(row.get("image_id", Path(str(row["image_path"])).stem)),
            "event_name": str(row.get("event_name", "unknown")),
            "image_path": str(row["image_path"]),
            "damage_mask_path": str(row["damage_mask_path"]),
        }


class BalancedPatchSamplingDataset(Dataset):
    """
    Balanced patch sampler for xView2/xBD damage mapping.

    The goal is not to only sample rare damage. It mixes:
      1. rare_damage: centered on class 2/3/4 pixels
      2. no_damage_building: centered on class 1 pixels
      3. hard_background: mostly-background/context crops
      4. random: normal random crops

    This avoids the common failure mode where rare-only sampling improves recall
    but destroys precision/no-damage/background performance.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        crop_size: int = 512,
        epoch_multiplier: int = 2,
        rare_damage_prob: float = 0.25,
        no_damage_building_prob: float = 0.25,
        hard_background_prob: float = 0.25,
        random_prob: float = 0.25,
        rare_damage_classes: Sequence[int] = (2, 3, 4),
        no_damage_classes: Sequence[int] = (1,),
        hard_background_max_building_frac: float = 0.02,
        hard_background_tries: int = 12,
        use_train_aug: bool = True,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.crop_size = int(crop_size)
        self.epoch_multiplier = int(epoch_multiplier)
        self.rare_damage_classes = tuple(int(x) for x in rare_damage_classes)
        self.no_damage_classes = tuple(int(x) for x in no_damage_classes)
        self.hard_background_max_building_frac = float(hard_background_max_building_frac)
        self.hard_background_tries = int(hard_background_tries)
        self.use_train_aug = bool(use_train_aug)
        self.seed = int(seed)

        required = {"image_path", "damage_mask_path"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"BalancedPatchSamplingDataset missing columns: {missing}")

        probs = {
            "rare_damage": float(rare_damage_prob),
            "no_damage_building": float(no_damage_building_prob),
            "hard_background": float(hard_background_prob),
            "random": float(random_prob),
        }
        total = sum(max(v, 0.0) for v in probs.values())
        if total <= 0:
            raise ValueError("At least one patch sampling probability must be positive.")

        self.modes = list(probs.keys())
        self.probs = np.array([max(probs[m], 0.0) / total for m in self.modes], dtype=np.float64)

        self.mode_counter = Counter()
        self.fallback_counter = Counter()

        # Event-balanced sampling: map each event to its row indices
        if "event_name" in self.df.columns:
            self.unique_events = self.df["event_name"].unique().tolist()
            self.event_to_indices = {
                ev: self.df.index[self.df["event_name"] == ev].tolist()
                for ev in self.unique_events
            }
        else:
            self.unique_events = None
            self.event_to_indices = None

    def __len__(self):
        return len(self.df) * max(self.epoch_multiplier, 1)

    def _rng_for_idx(self, idx: int) -> np.random.Generator:
        # Deterministic-ish per sample index while still changing across epochs via idx.
        return np.random.default_rng(self.seed + int(idx) * 9973)

    def _choose_mode(self, rng: np.random.Generator) -> str:
        return str(rng.choice(self.modes, p=self.probs))

    def _background_center(self, mask: np.ndarray, rng: np.random.Generator):
        # Try to find a crop with very little building area.
        best_center = _random_center(mask, self.crop_size, rng)
        best_building_frac = 1.0

        for _ in range(self.hard_background_tries):
            cx, cy = _random_center(mask, self.crop_size, rng)
            _, crop_mask = _crop_with_padding(
                np.zeros((*mask.shape, 3), dtype=np.uint8),
                mask,
                cx,
                cy,
                self.crop_size,
            )
            building_frac = float(np.mean(crop_mask > 0))
            if building_frac < best_building_frac:
                best_building_frac = building_frac
                best_center = (cx, cy)
            if building_frac <= self.hard_background_max_building_frac:
                return cx, cy

        return best_center

    def __getitem__(self, idx: int):
        rng = self._rng_for_idx(idx)

        # Event-balanced image selection: sample event uniformly, then image within event
        if self.unique_events is not None:
            event = self.unique_events[int(rng.integers(0, len(self.unique_events)))]
            event_indices = self.event_to_indices[event]
            real_idx = int(rng.choice(event_indices))
        else:
            real_idx = int(idx) % len(self.df)
        row = self.df.iloc[real_idx]

        image = read_rgb_image(row["image_path"])
        mask = read_label_mask(row["damage_mask_path"])

        mode = self._choose_mode(rng)
        chosen_mode = mode

        if mode == "rare_damage":
            center = _sample_class_center(mask, self.rare_damage_classes, rng)
            if center is None:
                chosen_mode = "random_fallback_from_rare"
                center = _random_center(mask, self.crop_size, rng)

        elif mode == "no_damage_building":
            center = _sample_class_center(mask, self.no_damage_classes, rng)
            if center is None:
                chosen_mode = "random_fallback_from_no_damage"
                center = _random_center(mask, self.crop_size, rng)

        elif mode == "hard_background":
            center = self._background_center(mask, rng)

        elif mode == "random":
            center = _random_center(mask, self.crop_size, rng)

        else:
            chosen_mode = "random_fallback_unknown_mode"
            center = _random_center(mask, self.crop_size, rng)

        self.mode_counter[chosen_mode] += 1
        if chosen_mode != mode:
            self.fallback_counter[chosen_mode] += 1

        crop_img, crop_mask = _crop_with_padding(image, mask, center[0], center[1], self.crop_size)

        if self.use_train_aug:
            crop_img, crop_mask = _apply_light_train_aug(crop_img, crop_mask, rng)

        return {
            "image": _normalize_image(crop_img),
            "mask": _to_mask_tensor(crop_mask),
            "image_id": str(row.get("image_id", Path(str(row["image_path"])).stem)),
            "event_name": str(row.get("event_name", "unknown")),
            "sample_mode": chosen_mode,
            "image_path": str(row["image_path"]),
            "damage_mask_path": str(row["damage_mask_path"]),
        }