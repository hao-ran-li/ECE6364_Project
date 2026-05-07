from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from week4_balanced_patch_sampling_utils import (
    BalancedPatchSamplingDataset,
    FullImageDataset,
)

from damage_mapping_utils import (
    compute_class_weights_from_manifest,
    confusion_matrix_from_tensors,
    metrics_from_cm,
    print_header,
    save_prediction_examples,
    set_seed,
)


CONFIG = {
    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    "project_root": "/workspace/ECE6364",
    "splits_root": r"Outputs/week1_trainfolder_testfolder_seen_unseen_split",
    "output_subdir": r"Outputs/week4_train_unet_patch_sampling",

    # If fold_mode == "all", train all fold_* folders under splits_root.
    # If fold_mode == "single", train only fold_name.
    # If fold_mode == "list", train folders listed in fold_names.
    "fold_mode": "single",  # "all", "single", or "list"
    # "fold_name": "fold_00_holdout_socal-fire__guatemala-volcano",
    "fold_name": "fold_04_holdout_midwest-flooding__hurricane-matthew",
    "fold_names": [],

    # Split CSV filenames inside each fold folder
    "train_csv_name": "train.csv",
    "val_csv_name": "val.csv",
    "seen_test_csv_name": "seen_test.csv",
    "unseen_test_csv_name": "unseen_test.csv",

    # ------------------------------------------------------------
    # Fair-comparison training setup
    # ------------------------------------------------------------
    "image_size": 512,
    "patch_size": 512,
    "train_epoch_multiplier": 2,
    "batch_size": 4,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 8,
    "pin_memory": True,
    "seed": 42,
    "num_classes": 5,
    "debug_overfit_n": 0,

    # ------------------------------------------------------------
    # U-Net architecture configs
    # ------------------------------------------------------------
    "model_name": "U-Net",
    "encoder_name": "resnet34",
    "encoder_weights": None,  # from scratch for fair comparison
    "encoder_depth": 5,
    "decoder_channels": (256, 128, 64, 32, 16),
    "decoder_use_batchnorm": True,
    "decoder_attention_type": None,
    "activation": None,

    # ------------------------------------------------------------
    # Weighted cross entropy
    # ------------------------------------------------------------
    "class_weight_mode": "median_frequency", # 
    "background_weight_multiplier": 0.25,
    "class_weight_clamp_max": 8.0,

    # ------------------------------------------------------------
    # Patch sampling extension
    # ------------------------------------------------------------
    "rare_damage_prob": 0.35,          # minor / major / destroyed
    "no_damage_building_prob": 0.25,   # no-damage building examples
    "hard_background_prob": 0.15,      # mostly-background/context patches
    "random_prob": 0.25,               # preserve normal distribution

    "rare_damage_classes": [2, 3, 4],
    "no_damage_classes": [1],
    "hard_background_max_building_frac": 0.02,
    "hard_background_tries": 12,


    # ------------------------------------------------------------
    # Model selection and outputs
    # ------------------------------------------------------------
    "best_metric": "val_balanced_mIoU",
    "save_prediction_examples": 4,
    "save_test_prediction_examples": 4,
}


CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]


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


def run_epoch(model, loader, optimizer, device, num_classes: int, class_weights, train: bool):
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
            loss = F.cross_entropy(logits, masks, weight=class_weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        cm_total += confusion_matrix_from_tensors(preds.detach().cpu(), masks.detach().cpu(), num_classes)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_items += bs

        if step == 1 or step % 25 == 0 or step == n_batches:
            mode = "train" if train else "eval"
            print(f"  [{mode}] batch {step}/{n_batches} | loss={loss.item():.4f}")

    avg_loss = total_loss / max(total_items, 1)
    return avg_loss, cm_total


def balanced_bg_fg_miou(cm: np.ndarray) -> float:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    true = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    denom = true + pred - tp
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(0.5 * iou[0] + 0.5 * np.mean(iou[1:]))


def build_metric_row(prefix: str, loss: float, cm: np.ndarray) -> Dict[str, float]:
    metrics_all = metrics_from_cm(cm, exclude_background=False)
    metrics_fg = metrics_from_cm(cm, exclude_background=True)
    balanced_miou = balanced_bg_fg_miou(cm)
    return {
        f"{prefix}_loss": float(loss),
        f"{prefix}_mIoU": float(metrics_all["mIoU"]),
        f"{prefix}_macroF1": float(metrics_all["macroF1"]),
        f"{prefix}_mIoU_fg": float(metrics_fg["mIoU"]),
        f"{prefix}_macroF1_fg": float(metrics_fg["macroF1"]),
        f"{prefix}_balanced_mIoU": float(balanced_miou),
    }


def evaluate_split(model, df: pd.DataFrame, cfg: Dict, device, split_name: str, output_dir: Path):
    ds = FullImageDataset(df=df, image_size=int(cfg["image_size"]))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]) and device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
        prefetch_factor=2 if int(cfg["num_workers"]) > 0 else None,
    )

    # Unweighted CE for reporting loss only; metrics are from predictions.
    eval_loss, cm = run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=device,
        num_classes=int(cfg["num_classes"]),
        class_weights=None,
        train=False,
    )

    metrics_all = metrics_from_cm(cm, exclude_background=False)
    metrics_fg = metrics_from_cm(cm, exclude_background=True)
    balanced_miou = balanced_bg_fg_miou(cm)
    result = {
        "split": split_name,
        "num_samples": int(len(df)),
        "loss": float(eval_loss),
        "mIoU": float(metrics_all["mIoU"]),
        "macroF1": float(metrics_all["macroF1"]),
        "mIoU_fg": float(metrics_fg["mIoU"]),
        "macroF1_fg": float(metrics_fg["macroF1"]),
        "balanced_mIoU": float(balanced_miou),
        "metrics_all": metrics_all,
        "metrics_fg": metrics_fg,
        "confusion_matrix": cm.tolist(),
    }

    with open(output_dir / f"{split_name}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_dir / f"{split_name}_confusion_matrix.csv")

    if int(cfg["save_test_prediction_examples"]) > 0:
        save_prediction_examples(
            model=model,
            dataset=ds,
            device=device,
            out_dir=output_dir / f"predictions_{split_name}",
            num_examples=int(cfg["save_test_prediction_examples"]),
            prefix=f"unet_patch_{split_name}",
        )
    return result, ds


def read_fold_data(fold_dir: Path, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = fold_dir / cfg["train_csv_name"]
    val_path = fold_dir / cfg["val_csv_name"]
    seen_test_path = fold_dir / cfg["seen_test_csv_name"]
    unseen_test_path = fold_dir / cfg["unseen_test_csv_name"]
    for p in [train_path, val_path, seen_test_path, unseen_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split CSV: {p}")
    return pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(seen_test_path), pd.read_csv(unseen_test_path)


def maybe_debug_overfit(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Dict):
    n = int(cfg.get("debug_overfit_n", 0))
    if n > 0:
        small = train_df.head(n).copy()
        return small, small.copy()
    return train_df, val_df


def print_event_distribution(name: str, df: pd.DataFrame):
    print(f"\n{name}: {len(df)} samples")
    if "event_name" in df.columns:
        counts = df["event_name"].value_counts().sort_index()
        for ev, cnt in counts.items():
            print(f"  - {ev}: {cnt}")


def fold_sort_key(path: Path):
    m = re.search(r"fold_(\d+)", path.name)
    return int(m.group(1)) if m else path.name


def find_fold_dirs(cfg: Dict) -> List[Path]:
    project_root = Path(cfg["project_root"])
    splits_root = project_root / Path(cfg["splits_root"])
    if cfg["fold_mode"] == "single":
        return [splits_root / cfg["fold_name"]]
    if cfg["fold_mode"] == "list":
        return [splits_root / name for name in cfg["fold_names"]]
    if cfg["fold_mode"] == "all":
        fold_dirs = sorted([p for p in splits_root.iterdir() if p.is_dir() and p.name.startswith("fold_")], key=fold_sort_key)
        if not fold_dirs:
            raise RuntimeError(f"No fold directories found under {splits_root}")
        return fold_dirs
    raise ValueError(f"Unknown fold_mode: {cfg['fold_mode']}")


def make_seen_unseen_gap_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["mIoU", "macroF1", "mIoU_fg", "macroF1_fg", "balanced_mIoU"]
    for fold, g in results_df.groupby("fold"):
        seen = g[g["split"] == "seen_test"]
        unseen = g[g["split"] == "unseen_test"]
        if len(seen) != 1 or len(unseen) != 1:
            continue
        seen = seen.iloc[0]
        unseen = unseen.iloc[0]
        row = {"fold": fold}
        for m in metrics:
            row[f"seen_{m}"] = float(seen[m])
            row[f"unseen_{m}"] = float(unseen[m])
            row[f"drop_{m}"] = float(seen[m] - unseen[m])
        rows.append(row)
    return pd.DataFrame(rows)


def run_one_fold(cfg: Dict, fold_dir: Path, fold_index: int):
    set_seed(int(cfg["seed"]) + fold_index)
    project_root = Path(cfg["project_root"])
    output_root = project_root / Path(cfg["output_subdir"])
    output_dir = output_root / fold_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_header(f"ECE 6364 - Week 4 U-Net patch sampling | {fold_dir.name}")
    print("project root:", project_root)
    print("fold dir    :", fold_dir)
    print("output dir  :", output_dir)
    print("device      :", device)

    train_df, val_df, seen_test_df, unseen_test_df = read_fold_data(fold_dir, cfg)
    train_df, val_df = maybe_debug_overfit(train_df, val_df, cfg)

    print_header("Split summary")
    print_event_distribution("train", train_df)
    print_event_distribution("val", val_df)
    print_event_distribution("seen_test", seen_test_df)
    print_event_distribution("unseen_test", unseen_test_df)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    weight_info = compute_class_weights_from_manifest(
        train_df=train_df,
        num_classes=int(cfg["num_classes"]),
        mode=str(cfg["class_weight_mode"]),
        background_multiplier=float(cfg["background_weight_multiplier"]),
        clamp_max=float(cfg["class_weight_clamp_max"]),
    )
    class_weights = torch.tensor(weight_info["weights"], dtype=torch.float32, device=device)
    with open(output_dir / "class_weights.json", "w") as f:
        json.dump(weight_info, f, indent=2)

    sample_cfg = {
        "rare_damage_prob": float(cfg["rare_damage_prob"]),
        "no_damage_building_prob": float(cfg["no_damage_building_prob"]),
        "hard_background_prob": float(cfg["hard_background_prob"]),
        "random_prob": float(cfg["random_prob"]),

        "rare_damage_classes": list(cfg["rare_damage_classes"]),
        "no_damage_classes": list(cfg["no_damage_classes"]),

        "hard_background_max_building_frac": float(cfg["hard_background_max_building_frac"]),
        "hard_background_tries": int(cfg["hard_background_tries"]),

        "patch_size": int(cfg["patch_size"]),
        "train_epoch_multiplier": int(cfg["train_epoch_multiplier"]),
    }
    with open(output_dir / "patch_sampling_config.json", "w") as f:
        json.dump(sample_cfg, f, indent=2)

    print_header("Class weights")
    print(json.dumps(weight_info, indent=2))
    print_header("Patch-sampling config")
    print(json.dumps(sample_cfg, indent=2))

    train_ds = BalancedPatchSamplingDataset(
        df=train_df,
        crop_size=int(cfg["patch_size"]),
        epoch_multiplier=int(cfg["train_epoch_multiplier"]),

        rare_damage_prob=float(cfg["rare_damage_prob"]),
        no_damage_building_prob=float(cfg["no_damage_building_prob"]),
        hard_background_prob=float(cfg["hard_background_prob"]),
        random_prob=float(cfg["random_prob"]),

        rare_damage_classes=tuple(cfg["rare_damage_classes"]),
        no_damage_classes=tuple(cfg["no_damage_classes"]),
        hard_background_max_building_frac=float(cfg["hard_background_max_building_frac"]),
        hard_background_tries=int(cfg["hard_background_tries"]),

        use_train_aug=True,
        seed=int(cfg["seed"]) + fold_index,
    )
    val_ds = FullImageDataset(df=val_df, image_size=int(cfg["image_size"]))

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["epochs"]), eta_min=1e-6
    )
    
    history = []
    best_value = -1.0
    best_epoch = -1
    best_path = ckpt_dir / "best_unet_patch_sampling.pt"
    metric_name = str(cfg["best_metric"])

    for epoch in range(1, int(cfg["epochs"]) + 1):
        print_header(f"Epoch {epoch}/{cfg['epochs']}")
        # train
        train_loss, train_cm = run_epoch(model, train_loader, optimizer, device, int(cfg["num_classes"]), class_weights, train=True)
        # val
        val_loss, val_cm = run_epoch(model, val_loader, None, device, int(cfg["num_classes"]), None, train=False)

        row = {"fold": fold_dir.name, "epoch": epoch}
        row.update(build_metric_row("train", train_loss, train_cm))
        row.update(build_metric_row("val", val_loss, val_cm))
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{cfg['epochs']} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_mIoU={row['train_mIoU']:.4f} val_mIoU={row['val_mIoU']:.4f} | "
            f"train_mIoU_fg={row['train_mIoU_fg']:.4f} val_mIoU_fg={row['val_mIoU_fg']:.4f} | "
            f"val_balanced_mIoU={row['val_balanced_mIoU']:.4f}"
        )
        pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)

        scheduler.step()
        metric_value = float(row[metric_name])
        if metric_value > best_value:
            best_value = metric_value
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "fold_dir": str(fold_dir),
                    "best_metric_name": metric_name,
                    "best_metric_value": best_value,
                    "class_weights": weight_info,
                    "patch_sampling_config": sample_cfg,
                },
                best_path,
            )
            print(f"  [saved] best checkpoint by {metric_name}: {best_value:.4f}")

    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
    with open(output_dir / "sampling_mode_counts.json", "w") as f:
        json.dump(train_ds.mode_counter, f, indent=2)

    print_header("Training complete")
    print(f"Best epoch by {metric_name}: {best_epoch} | best value={best_value:.4f}")
    print("Saved history to:", output_dir / "train_history.csv")
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
            "unet_patch_val",
        )

    # Save training summary
    pd.DataFrame([{
        "fold": fold_dir.name,
        "best_epoch": best_epoch,
        "best_metric": metric_name,
        "best_value": best_value,
        "checkpoint": str(best_path),
    }]).to_csv(output_dir / "training_summary.csv", index=False)

    # Evaluate on seen and unseen test splits using best checkpoint
    print_header("Evaluating on test splits")
    test_rows = []
    for split_name, split_df in [("seen_test", seen_test_df), ("unseen_test", unseen_test_df)]:
        print(f"\n  [{split_name}] n={len(split_df)}")
        result, _ = evaluate_split(model, split_df, cfg, device, split_name, output_dir)
        test_rows.append({
            "fold": fold_dir.name,
            "split": split_name,
            **{k: v for k, v in result.items() if k not in ("metrics_all", "metrics_fg", "confusion_matrix")},
        })
        print(
            f"  [{split_name}] mIoU={result['mIoU']:.4f} mIoU_fg={result['mIoU_fg']:.4f} "
            f"balanced_mIoU={result['balanced_mIoU']:.4f}"
        )

    return pd.DataFrame(test_rows)


def main():
    cfg = CONFIG
    fold_dirs = find_fold_dirs(cfg)

    print_header("U-Net patch-sampling fold plan")
    for i, fd in enumerate(fold_dirs):
        print(f"{i}: {fd}")

    all_test_rows = []
    for fold_index, fold_dir in enumerate(fold_dirs):
        fold_results = run_one_fold(cfg, fold_dir, fold_index)
        all_test_rows.append(fold_results)

    project_root = Path(cfg["project_root"])
    output_root = project_root / Path(cfg["output_subdir"])
    output_root.mkdir(parents=True, exist_ok=True)

    if all_test_rows:
        all_results = pd.concat(all_test_rows, ignore_index=True)
        all_results.to_csv(output_root / "all_folds_test_summary.csv", index=False)

        agg = all_results.groupby("split").agg(
            mIoU_mean=("mIoU", "mean"),
            mIoU_std=("mIoU", "std"),
            macroF1_mean=("macroF1", "mean"),
            macroF1_std=("macroF1", "std"),
            mIoU_fg_mean=("mIoU_fg", "mean"),
            mIoU_fg_std=("mIoU_fg", "std"),
            macroF1_fg_mean=("macroF1_fg", "mean"),
            macroF1_fg_std=("macroF1_fg", "std"),
            balanced_mIoU_mean=("balanced_mIoU", "mean"),
            balanced_mIoU_std=("balanced_mIoU", "std"),
        ).reset_index()
        agg.to_csv(output_root / "all_folds_test_summary_agg.csv", index=False)

        gap = make_seen_unseen_gap_table(all_results)
        gap.to_csv(output_root / "all_folds_seen_unseen_gap_by_fold.csv", index=False)
        if len(gap) > 0:
            gap_metrics = [c for c in gap.columns if c.startswith("drop_")]
            gap_agg = gap[gap_metrics].agg(["mean", "std", "min", "max"]).T.reset_index()
            gap_agg = gap_agg.rename(columns={"index": "metric"})
            gap_agg.to_csv(output_root / "all_folds_seen_unseen_gap_agg.csv", index=False)

        print_header("All-fold aggregate")
        print(agg)
        print("Saved:", output_root / "all_folds_test_summary.csv")
        print("Saved:", output_root / "all_folds_test_summary_agg.csv")
        print("Saved:", output_root / "all_folds_seen_unseen_gap_by_fold.csv")


if __name__ == "__main__":
    main()
