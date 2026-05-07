from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Dict, List, Tuple
from xml.parsers.expat import model

import albumentations as A
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from damage_mapping_utils import (
    DamageDataset,
    compute_class_weights_from_manifest,
    confusion_matrix_from_tensors,
    metrics_from_cm,
    print_header,
    save_prediction_examples,
    set_seed,
)

CONFIG = {
    "project_root": "/workspace/ECE6364",
    "splits_root": r"Outputs/week1_trainfolder_testfolder_seen_unseen_split",
    "output_subdir": r"Outputs/week2_train_unet_weighted_ce",

    # "single", "list", or "all"
    "fold_mode": "single",
    # "fold_name": "fold_00_holdout_socal-fire__guatemala-volcano",
    "fold_name": "fold_04_holdout_midwest-flooding__hurricane-matthew",
    "fold_names": [],

    "train_csv_name": "train.csv",
    "val_csv_name": "val.csv",

    # Fair-comparison training configs
    "image_size": 512,
    "batch_size": 4,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 8,
    "pin_memory": True,
    "seed": 42,
    "num_classes": 5,
    "debug_overfit_n": 0,

    # U-Net architecture configs
    "model_name": "U-Net",
    "encoder_name": "resnet34",
    "encoder_weights": None,
    "encoder_depth": 5,
    "decoder_channels": (256, 128, 64, 32, 16),
    "decoder_use_batchnorm": True,
    "decoder_attention_type": None,
    "activation": None,

    # Weighted CE configs
    "class_weight_mode": "median_frequency",
    "background_weight_multiplier": 0.25,
    "class_weight_clamp_max": 8.0,

    # Model selection
    "best_metric": "val_balanced_mIoU",
    "save_prediction_examples": 4,
    # Unused in this training-only script; kept for config compatibility.
    "save_test_prediction_examples": 4,
}

CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]


def build_transforms(cfg: Dict, train: bool):
    size = int(cfg["image_size"])
    if train:
        return A.Compose([
            A.Resize(size, size, interpolation=1, mask_interpolation=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])
    return A.Compose([A.Resize(size, size, interpolation=1, mask_interpolation=0)])


def build_model(cfg: Dict):
    return smp.Unet(
        encoder_name=cfg["encoder_name"],
        encoder_weights=cfg["encoder_weights"],
        encoder_depth=int(cfg["encoder_depth"]),
        decoder_channels=tuple(cfg["decoder_channels"]),
        decoder_use_batchnorm=bool(cfg["decoder_use_batchnorm"]),
        decoder_attention_type=cfg["decoder_attention_type"],
        in_channels=3,
        classes=int(cfg["num_classes"]),
        activation=cfg["activation"],
    )


def balanced_bg_fg_miou(cm: np.ndarray) -> float:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    true = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    denom = true + pred - tp
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(0.5 * iou[0] + 0.5 * np.mean(iou[1:]))


def run_epoch(model, loader, optimizer, criterion, device, num_classes: int, train: bool):
    model.train(train)
    total_loss = 0.0
    total_items = 0
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_batches = len(loader)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        cm_total += confusion_matrix_from_tensors(
            preds.detach().cpu(),
            masks.detach().cpu(),
            num_classes,
        )

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_items += bs

        if step == 1 or step % 25 == 0 or step == n_batches:
            mode = "train" if train else "eval"
            print(f"  [{mode}] batch {step}/{n_batches} | loss={loss.item():.4f}")

    return total_loss / max(total_items, 1), cm_total


def metric_row(prefix: str, loss: float, cm: np.ndarray) -> Dict[str, float]:
    all_m = metrics_from_cm(cm, exclude_background=False)
    fg_m = metrics_from_cm(cm, exclude_background=True)
    return {
        f"{prefix}_loss": float(loss),
        f"{prefix}_mIoU": float(all_m["mIoU"]),
        f"{prefix}_macroF1": float(all_m["macroF1"]),
        f"{prefix}_mIoU_fg": float(fg_m["mIoU"]),
        f"{prefix}_macroF1_fg": float(fg_m["macroF1"]),
        f"{prefix}_balanced_mIoU": balanced_bg_fg_miou(cm),
    }


def evaluate_split(model, df: pd.DataFrame, cfg: Dict, device, split_name: str, output_dir: Path):
    ds = DamageDataset(df, augment=build_transforms(cfg, train=False))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]) and device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
        prefetch_factor=2 if int(cfg["num_workers"]) > 0 else None,
    )
    criterion = nn.CrossEntropyLoss()
    loss, cm = run_epoch(model, loader, None, criterion, device, int(cfg["num_classes"]), train=False)

    all_m = metrics_from_cm(cm, exclude_background=False)
    fg_m = metrics_from_cm(cm, exclude_background=True)
    result = {
        "split": split_name,
        "num_samples": int(len(df)),
        "loss": float(loss),
        "mIoU": float(all_m["mIoU"]),
        "macroF1": float(all_m["macroF1"]),
        "mIoU_fg": float(fg_m["mIoU"]),
        "macroF1_fg": float(fg_m["macroF1"]),
        "balanced_mIoU": balanced_bg_fg_miou(cm),
        "metrics_all": all_m,
        "metrics_fg": fg_m,
        "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / f"{split_name}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_dir / f"{split_name}_confusion_matrix.csv")

    if int(cfg["save_test_prediction_examples"]) > 0:
        save_prediction_examples(
            model,
            ds,
            device,
            output_dir / f"predictions_{split_name}",
            int(cfg["save_test_prediction_examples"]),
            f"unet_{split_name}",
        )
    return result


def read_fold_data(fold_dir: Path, cfg: Dict):
    # Training-only script: only load train/val splits.
    # Formal seen/unseen test evaluation should be handled by the separate testing script.
    paths = {
        "train": fold_dir / cfg["train_csv_name"],
        "val": fold_dir / cfg["val_csv_name"],
    }
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing split CSV: {p}")
    return {k: pd.read_csv(v) for k, v in paths.items()}


def maybe_debug_overfit(train_df, val_df, cfg):
    n = int(cfg.get("debug_overfit_n", 0))
    if n > 0:
        small = train_df.head(n).copy()
        return small, small.copy()
    return train_df, val_df


def fold_sort_key(path: Path):
    m = re.search(r"fold_(\d+)", path.name)
    return int(m.group(1)) if m else path.name


def find_fold_dirs(cfg: Dict) -> List[Path]:
    root = Path(cfg["project_root"]) / Path(cfg["splits_root"])
    if cfg["fold_mode"] == "single":
        return [root / cfg["fold_name"]]
    if cfg["fold_mode"] == "list":
        return [root / name for name in cfg["fold_names"]]
    if cfg["fold_mode"] == "all":
        folds = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("fold_")], key=fold_sort_key)
        if not folds:
            raise RuntimeError(f"No fold folders found under {root}")
        return folds
    raise ValueError(f"Unknown fold_mode: {cfg['fold_mode']}")


def print_event_distribution(name: str, df: pd.DataFrame):
    print(f"\n{name}: {len(df)} samples")
    if "event_name" in df.columns:
        for ev, cnt in df["event_name"].value_counts().sort_index().items():
            print(f"  - {ev}: {cnt}")


def run_one_fold(cfg: Dict, fold_dir: Path, fold_index: int) -> pd.DataFrame:
    set_seed(int(cfg["seed"]) + fold_index)

    project_root = Path(cfg["project_root"])
    output_dir = project_root / Path(cfg["output_subdir"]) / fold_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_header(f"U-Net weighted CE | {fold_dir.name}")
    print("fold dir   :", fold_dir)
    print("output dir :", output_dir)
    print("device     :", device)

    data = read_fold_data(fold_dir, cfg)
    train_df, val_df = maybe_debug_overfit(data["train"], data["val"], cfg)

    print_header("Split summary")
    print_event_distribution("train", train_df)
    print_event_distribution("val", val_df)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    weight_info = compute_class_weights_from_manifest(
        train_df=train_df,
        num_classes=int(cfg["num_classes"]),
        mode=str(cfg["class_weight_mode"]),
        background_multiplier=float(cfg["background_weight_multiplier"]),
        clamp_max=float(cfg["class_weight_clamp_max"]),
    )
    with open(output_dir / "class_weights.json", "w") as f:
        json.dump(weight_info, f, indent=2)

    class_weights = torch.tensor(weight_info["weights"], dtype=torch.float32, device=device)
    print_header("Class weights")
    print(json.dumps(weight_info, indent=2))

    train_ds = DamageDataset(train_df, augment=build_transforms(cfg, train=True))
    val_ds = DamageDataset(val_df, augment=build_transforms(cfg, train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]) and device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
        prefetch_factor=2 if int(cfg["num_workers"]) > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]) and device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
        prefetch_factor=2 if int(cfg["num_workers"]) > 0 else None,
    )

    print_header("Building model")
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["epochs"]), eta_min=1e-6
    )

    best_value = -1.0
    best_epoch = -1
    best_path = ckpt_dir / "best_unet_weighted_ce.pt"
    metric_name = str(cfg["best_metric"])
    history = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        print_header(f"Epoch {epoch}/{cfg['epochs']}")
        train_loss, train_cm = run_epoch(model, train_loader, optimizer, criterion, device, int(cfg["num_classes"]), True)
        val_loss, val_cm = run_epoch(model, val_loader, optimizer, val_criterion, device, int(cfg["num_classes"]), False)

        row = {"fold": fold_dir.name, "epoch": epoch}
        row.update(metric_row("train", train_loss, train_cm))
        row.update(metric_row("val", val_loss, val_cm))
        history.append(row)

        pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{cfg['epochs']} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"val_mIoU={row['val_mIoU']:.4f} val_mIoU_fg={row['val_mIoU_fg']:.4f}"
        )

        if float(row[metric_name]) > best_value:
            best_value = float(row[metric_name])
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "fold_dir": str(fold_dir),
                "best_metric_name": metric_name,
                "best_metric_value": best_value,
                "class_weights": weight_info,
            }, best_path)
            print(f"  [saved] best checkpoint by {metric_name}: {best_value:.4f}")

    print_header("Training complete")
    print(f"Best epoch by {metric_name}: {best_epoch} | best value={best_value:.4f}")
    # print("Saved history to:", output_dir / "train_history.csv")
    print("Saved checkpoint to:", best_path)
    print("Saved sampling counts to:", output_dir / "sampling_mode_counts.json")
    
    # Reload the best validation checkpoint before saving optional validation examples.
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

    if int(cfg["save_prediction_examples"]) > 0:
        save_prediction_examples(
            model,
            val_ds,
            device,
            output_dir / "predictions_val",
            int(cfg["save_prediction_examples"]),
            "segformer_val",
        )

    out = pd.DataFrame([{
        "fold": fold_dir.name,
        "best_epoch": best_epoch,
        "best_metric": metric_name,
        "best_value": best_value,
        "checkpoint": str(best_path),
    }])
    out.to_csv(output_dir / "training_summary.csv", index=False)
    return out


def main():
    cfg = CONFIG
    folds = find_fold_dirs(cfg)

    print_header("U-Net weighted CE fold plan")
    for i, fd in enumerate(folds):
        print(f"{i}: {fd}")

    all_rows = []
    for i, fold_dir in enumerate(folds):
        all_rows.append(run_one_fold(cfg, fold_dir, i))

    output_root = Path(cfg["project_root"]) / Path(cfg["output_subdir"])
    all_results = pd.concat(all_rows, ignore_index=True)
    all_results.to_csv(output_root / "all_folds_training_summary.csv", index=False)

    print_header("All-fold training summary")
    print(all_results)


if __name__ == "__main__":
    main()
