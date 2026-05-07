
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from torch.utils.data import DataLoader
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from damage_mapping_utils import (
    DamageDataset,
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

    # Week 4 training output folder from:
    #   week4_segformer_patch_sampling_seen_unseen.py
    "model_output_subdir": r"Outputs/week4_train_segformer_patch_sampling",
    "eval_output_subdir": r"Outputs/week4_test_segformer_patch_sampling",

    # Expected checkpoint per fold:
    #   model_output_subdir/fold_name/checkpoints/best_segformer_patch_sampling.pt
    "checkpoint_name": "best_segformer_patch_sampling.pt",

    # "all", "single", or "list"
    "fold_mode": "single",
    # "fold_name": "fold_00_holdout_socal-fire__guatemala-volcano",
    "fold_name": "fold_04_holdout_midwest-flooding__hurricane-matthew",
    "fold_names": [],

    "seen_test_csv_name": "seen_test.csv",
    "unseen_test_csv_name": "unseen_test.csv",

    # ------------------------------------------------------------
    # Evaluation setup
    # ------------------------------------------------------------
    "image_size": 512,
    "batch_size": 4,
    "num_workers": 8,
    "pin_memory": True,
    "seed": 42,
    "num_classes": 5,

    # If True, skip folds that have not been trained yet.
    "skip_missing_checkpoints": True,

    # Print full 5x5 confusion matrices in runtime log.
    "print_confusion_matrices": True,

    # Save qualitative examples. Set to 0 if you only want metrics.
    "save_prediction_examples": 4,

    # ------------------------------------------------------------
    # Fallback SegFormer-B0 architecture config.
    # Checkpoint config is used if available.
    # ------------------------------------------------------------
    "model_name": "SegFormer-B0",
    "num_encoder_blocks": 4,
    "depths": [2, 2, 2, 2],
    "hidden_sizes": [32, 64, 160, 256],
    "num_attention_heads": [1, 2, 5, 8],
    "patch_sizes": [7, 3, 3, 3],
    "strides": [4, 2, 2, 2],
    "sr_ratios": [8, 4, 2, 1],
    "mlp_ratios": [4, 4, 4, 4],
    "decoder_hidden_size": 256,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "classifier_dropout_prob": 0.1,
}


CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
BUILDING_CLASS_IDS = [1, 2, 3, 4]

ID2LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}

CLASS_MAP = {
    "background": 0,
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": -1,  # skip from building-level evaluation
}


def build_transforms(cfg: Dict):
    size = int(cfg["image_size"])
    return A.Compose([
        A.Resize(size, size, interpolation=1, mask_interpolation=0),
    ])


class SegFormerDenseWrapper(nn.Module):
    """
    HuggingFace SegFormer returns lower-resolution logits.
    This wrapper upsamples logits to input image size so evaluation is the
    same style as U-Net:
        logits = model(images)
    """

    def __init__(self, segformer: SegformerForSemanticSegmentation):
        super().__init__()
        self.segformer = segformer

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.segformer(pixel_values=images)
        logits = outputs.logits
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits


def cfg_get(cfg: Dict, key: str):
    return cfg[key] if key in cfg else CONFIG[key]


def build_segformer_config(cfg: Dict) -> SegformerConfig:
    return SegformerConfig(
        num_labels=int(cfg_get(cfg, "num_classes")),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        num_encoder_blocks=int(cfg_get(cfg, "num_encoder_blocks")),
        depths=list(cfg_get(cfg, "depths")),
        hidden_sizes=list(cfg_get(cfg, "hidden_sizes")),
        num_attention_heads=list(cfg_get(cfg, "num_attention_heads")),
        patch_sizes=list(cfg_get(cfg, "patch_sizes")),
        strides=list(cfg_get(cfg, "strides")),
        sr_ratios=list(cfg_get(cfg, "sr_ratios")),
        mlp_ratios=list(cfg_get(cfg, "mlp_ratios")),
        decoder_hidden_size=int(cfg_get(cfg, "decoder_hidden_size")),
        hidden_dropout_prob=float(cfg_get(cfg, "hidden_dropout_prob")),
        attention_probs_dropout_prob=float(cfg_get(cfg, "attention_probs_dropout_prob")),
        classifier_dropout_prob=float(cfg_get(cfg, "classifier_dropout_prob")),
    )


def build_model(cfg: Dict):
    hf_cfg = build_segformer_config(cfg)
    base = SegformerForSemanticSegmentation(hf_cfg)
    return SegFormerDenseWrapper(base)


def load_state_dict_flexible(model: nn.Module, state_dict: Dict):
    """
    Supports checkpoints saved from:
      1. SegFormerDenseWrapper: keys start with 'segformer.'
      2. Raw HuggingFace SegFormer: keys do not include wrapper prefix.
    """
    try:
        model.load_state_dict(state_dict)
        return "wrapper_state_dict"
    except RuntimeError as wrapper_err:
        try:
            model.segformer.load_state_dict(state_dict)
            return "raw_hf_state_dict"
        except RuntimeError:
            raise wrapper_err


def balanced_bg_fg_miou(cm: np.ndarray) -> float:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    true = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    denom = true + pred - tp
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(0.5 * iou[0] + 0.5 * np.mean(iou[1:]))


def pixel_metrics_from_cm(cm: np.ndarray) -> Dict:
    metrics_all = metrics_from_cm(cm, exclude_background=False)
    metrics_fg = metrics_from_cm(cm, exclude_background=True)

    out = {
        "mIoU": float(metrics_all["mIoU"]),
        "macroF1": float(metrics_all["macroF1"]),
        "mIoU_fg": float(metrics_fg["mIoU"]),
        "macroF1_fg": float(metrics_fg["macroF1"]),
        "balanced_mIoU": balanced_bg_fg_miou(cm),
        "per_class_iou": metrics_all.get("per_class_iou", {}),
        "per_class_f1": metrics_all.get("per_class_f1", {}),
        "metrics_all": metrics_all,
        "metrics_fg": metrics_fg,
        "confusion_matrix": cm.astype(int).tolist(),
    }
    return out


def building_metrics_from_cm(cm: np.ndarray) -> Dict:
    """
    Rows = true building class, columns = predicted class.
    True class usually excludes background. Predicted class may be background
    if the model misses the building region.
    """
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    true = cm.sum(axis=1)
    pred = cm.sum(axis=0)

    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    total = cm.sum()
    overall_acc = float(tp.sum() / total) if total > 0 else 0.0

    fg = BUILDING_CLASS_IDS
    macro_f1_fg = float(np.mean(f1[fg])) if len(fg) else 0.0
    macro_acc_fg = float(np.mean(recall[fg])) if len(fg) else 0.0

    return {
        "num_buildings": int(total),
        "building_accuracy_overall": overall_acc,
        "building_macroF1_fg": macro_f1_fg,
        "building_macro_accuracy_fg": macro_acc_fg,
        "per_class_building_accuracy": {CLASS_NAMES[i]: float(recall[i]) for i in fg},
        "per_class_building_F1": {CLASS_NAMES[i]: float(f1[i]) for i in fg},
        "per_class_building_precision": {CLASS_NAMES[i]: float(precision[i]) for i in fg},
        "confusion_matrix": cm.astype(int).tolist(),
    }



def flatten_damage_class_metrics(prefix: str, pixel_metrics: Dict, building_metrics: Dict) -> Dict:
    """
    Add per-damage-class metrics to the summary CSV.
    Classes:
      no-damage, minor-damage, major-damage, destroyed
    """
    row = {}
    for cls_name in CLASS_NAMES[1:]:
        safe = cls_name.replace("-", "_")

        row[f"{prefix}_{safe}_pixel_IoU"] = float(pixel_metrics["per_class_iou"].get(cls_name, np.nan))
        row[f"{prefix}_{safe}_pixel_F1"] = float(pixel_metrics["per_class_f1"].get(cls_name, np.nan))

        row[f"{prefix}_{safe}_building_accuracy"] = float(
            building_metrics["per_class_building_accuracy"].get(cls_name, np.nan)
        )
        row[f"{prefix}_{safe}_building_F1"] = float(
            building_metrics["per_class_building_F1"].get(cls_name, np.nan)
        )
        row[f"{prefix}_{safe}_building_precision"] = float(
            building_metrics["per_class_building_precision"].get(cls_name, np.nan)
        )
    return row


def make_damage_class_long_rows(
    fold_name: str,
    split_name: str,
    pixel_metrics: Dict,
    building_metrics: Dict,
) -> List[Dict]:
    rows = []
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        rows.append({
            "fold": fold_name,
            "split": split_name,
            "class_id": cls_id,
            "class_name": cls_name,
            "pixel_IoU": float(pixel_metrics["per_class_iou"].get(cls_name, np.nan)),
            "pixel_F1": float(pixel_metrics["per_class_f1"].get(cls_name, np.nan)),
            "building_accuracy": float(building_metrics["per_class_building_accuracy"].get(cls_name, np.nan)),
            "building_F1": float(building_metrics["per_class_building_F1"].get(cls_name, np.nan)),
            "building_precision": float(building_metrics["per_class_building_precision"].get(cls_name, np.nan)),
        })
    return rows


def print_pixel_report(split_name: str, metrics: Dict, cm: np.ndarray):
    print_header(f"Pixel-level evaluation | {split_name}")
    print(f"mIoU        : {metrics['mIoU']:.4f}")
    print(f"macro F1    : {metrics['macroF1']:.4f}")
    print(f"mIoU_fg     : {metrics['mIoU_fg']:.4f}")
    print(f"macro F1_fg : {metrics['macroF1_fg']:.4f}")
    print(f"balanced mIoU(bg/fg): {metrics['balanced_mIoU']:.4f}")

    print("\nPer-class pixel IoU / F1:")
    for name in CLASS_NAMES:
        iou_val = metrics["per_class_iou"].get(name, np.nan)
        f1_val = metrics["per_class_f1"].get(name, np.nan)
        print(f"  {name:13s}: IoU={iou_val:.4f}, F1={f1_val:.4f}")

    if bool(CONFIG["print_confusion_matrices"]):
        print("\nPixel confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))


def print_building_report(split_name: str, metrics: Dict, cm: np.ndarray):
    print_header(f"Building-level evaluation | {split_name}")
    print(f"num buildings          : {metrics['num_buildings']}")
    print(f"building overall acc   : {metrics['building_accuracy_overall']:.4f}")
    print(f"building macro F1      : {metrics['building_macroF1_fg']:.4f}")
    print(f"building macro accuracy: {metrics['building_macro_accuracy_fg']:.4f}")

    print("\nPer-class building accuracy / F1:")
    for name in CLASS_NAMES[1:]:
        acc = metrics["per_class_building_accuracy"].get(name, np.nan)
        f1 = metrics["per_class_building_F1"].get(name, np.nan)
        print(f"  {name:13s}: acc={acc:.4f}, F1={f1:.4f}")

    if bool(CONFIG["print_confusion_matrices"]):
        print("\nBuilding confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))


def scale_polygon_to_pred(poly: Polygon, orig_w: int, orig_h: int, pred_w: int, pred_h: int) -> np.ndarray:
    sx = pred_w / float(orig_w)
    sy = pred_h / float(orig_h)
    coords = np.array(poly.exterior.coords, dtype=np.float32)
    coords[:, 0] *= sx
    coords[:, 1] *= sy
    coords = np.round(coords).astype(np.int32)
    return coords.reshape((-1, 1, 2))


def fill_geom(mask: np.ndarray, geom, orig_w: int, orig_h: int, pred_w: int, pred_h: int):
    if isinstance(geom, Polygon):
        pts = scale_polygon_to_pred(geom, orig_w, orig_h, pred_w, pred_h)
        cv2.fillPoly(mask, [pts], 1)
        for interior in geom.interiors:
            coords = np.array(interior.coords, dtype=np.float32)
            coords[:, 0] *= pred_w / float(orig_w)
            coords[:, 1] *= pred_h / float(orig_h)
            hole = np.round(coords).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [hole], 0)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            fill_geom(mask, poly, orig_w, orig_h, pred_w, pred_h)


def update_building_cm_for_image(pred_mask: np.ndarray, row: pd.Series, building_cm: np.ndarray) -> int:
    """
    Building-level protocol:
      1. For each annotated building polygon, collect predicted labels inside it.
      2. Assign majority predicted class.
      3. Compare against ground-truth building damage label.
    """
    label_path = Path(row["label_path"])
    image_path = Path(row["image_path"])

    if not label_path.exists():
        raise FileNotFoundError(f"Missing label JSON: {label_path}")

    orig_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if orig_img is None:
        orig_h, orig_w = 1024, 1024
    else:
        orig_h, orig_w = orig_img.shape[:2]

    pred_h, pred_w = pred_mask.shape

    with open(label_path, "r") as f:
        data = json.load(f)

    count = 0
    for feat in data.get("features", {}).get("xy", []):
        props = feat.get("properties", {})
        subtype = props.get("subtype", "un-classified")
        true_cls = CLASS_MAP.get(subtype, -1)

        if true_cls not in BUILDING_CLASS_IDS:
            continue

        geom_wkt = feat.get("wkt", None)
        if geom_wkt is None:
            continue

        try:
            geom = wkt.loads(geom_wkt)
        except Exception:
            continue

        poly_mask = np.zeros((pred_h, pred_w), dtype=np.uint8)
        fill_geom(poly_mask, geom, orig_w, orig_h, pred_w, pred_h)

        vals = pred_mask[poly_mask > 0]
        if vals.size == 0:
            continue

        pred_cls = int(np.bincount(vals.astype(np.int64), minlength=len(CLASS_NAMES)).argmax())
        building_cm[true_cls, pred_cls] += 1
        count += 1

    return count


@torch.no_grad()
def evaluate_split(model, df: pd.DataFrame, cfg: Dict, device, split_name: str, fold_name: str, output_dir: Path):
    ds = DamageDataset(df, augment=build_transforms(cfg))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]) and device.type == "cuda",
        persistent_workers=int(cfg["num_workers"]) > 0,
        prefetch_factor=2 if int(cfg["num_workers"]) > 0 else None,
    )

    model.eval()
    num_classes = int(cfg["num_classes"])
    pixel_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    building_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    total_buildings = 0
    n_batches = len(loader)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        pixel_cm += confusion_matrix_from_tensors(
            preds.detach().cpu(),
            masks.detach().cpu(),
            num_classes,
        )

        preds_np = preds.detach().cpu().numpy().astype(np.int64)
        bs = preds_np.shape[0]
        start_idx = (step - 1) * int(cfg["batch_size"])

        for i in range(bs):
            row_idx = start_idx + i
            if row_idx >= len(df):
                continue
            total_buildings += update_building_cm_for_image(
                pred_mask=preds_np[i],
                row=df.iloc[row_idx],
                building_cm=building_cm,
            )

        if step == 1 or step % 25 == 0 or step == n_batches:
            print(f"  [{split_name}] batch {step}/{n_batches} | buildings so far={total_buildings}")

    pixel_metrics = pixel_metrics_from_cm(pixel_cm)
    building_metrics = building_metrics_from_cm(building_cm)

    with open(output_dir / f"{split_name}_pixel_metrics.json", "w") as f:
        json.dump(pixel_metrics, f, indent=2)
    pd.DataFrame(pixel_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / f"{split_name}_pixel_confusion_matrix.csv"
    )
    pd.DataFrame(
        [{"class_name": k, "IoU": v} for k, v in pixel_metrics["per_class_iou"].items()]
    ).to_csv(output_dir / f"{split_name}_pixel_per_class_iou.csv", index=False)

    with open(output_dir / f"{split_name}_building_metrics.json", "w") as f:
        json.dump(building_metrics, f, indent=2)
    pd.DataFrame(building_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / f"{split_name}_building_confusion_matrix.csv"
    )
    pd.DataFrame(
        [
            {
                "class_name": name,
                "building_accuracy": building_metrics["per_class_building_accuracy"].get(name, np.nan),
                "building_F1": building_metrics["per_class_building_F1"].get(name, np.nan),
                "building_precision": building_metrics["per_class_building_precision"].get(name, np.nan),
            }
            for name in CLASS_NAMES[1:]
        ]
    ).to_csv(output_dir / f"{split_name}_building_per_class_metrics.csv", index=False)

    print_pixel_report(split_name, pixel_metrics, pixel_cm)
    print_building_report(split_name, building_metrics, building_cm)

    if int(cfg["save_prediction_examples"]) > 0:
        save_prediction_examples(
            model,
            ds,
            device,
            output_dir / f"predictions_{split_name}",
            int(cfg["save_prediction_examples"]),
            f"segformer_patch_{split_name}",
        )

    # Long-format per-damage-class table for this fold/split.
    damage_class_rows = make_damage_class_long_rows(
        fold_name=fold_name,
        split_name=split_name,
        pixel_metrics=pixel_metrics,
        building_metrics=building_metrics,
    )
    pd.DataFrame(damage_class_rows).to_csv(
        output_dir / f"{split_name}_damage_class_metrics.csv",
        index=False,
    )

    summary_row = {
        "fold": fold_name,
        "split": split_name,
        "num_samples": int(len(df)),

        "pixel_mIoU": pixel_metrics["mIoU"],
        "pixel_macroF1": pixel_metrics["macroF1"],
        "pixel_mIoU_fg": pixel_metrics["mIoU_fg"],
        "pixel_macroF1_fg": pixel_metrics["macroF1_fg"],
        "pixel_balanced_mIoU": pixel_metrics["balanced_mIoU"],

        "building_num_buildings": building_metrics["num_buildings"],
        "building_accuracy_overall": building_metrics["building_accuracy_overall"],
        "building_macroF1": building_metrics["building_macroF1_fg"],
        "building_macro_accuracy": building_metrics["building_macro_accuracy_fg"],
    }
    summary_row.update(flatten_damage_class_metrics("", pixel_metrics, building_metrics))

    return summary_row, pd.DataFrame(damage_class_rows)


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
        fold_dirs = sorted(
            [p for p in splits_root.iterdir() if p.is_dir() and p.name.startswith("fold_")],
            key=fold_sort_key,
        )
        if not fold_dirs:
            raise RuntimeError(f"No fold directories found under {splits_root}")
        return fold_dirs

    raise ValueError(f"Unknown fold_mode: {cfg['fold_mode']}")


def read_test_data(fold_dir: Path, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seen_path = fold_dir / cfg["seen_test_csv_name"]
    unseen_path = fold_dir / cfg["unseen_test_csv_name"]
    for p in [seen_path, unseen_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing test CSV: {p}")

    seen_df = pd.read_csv(seen_path)
    unseen_df = pd.read_csv(unseen_path)

    for name, df in [("seen_test", seen_df), ("unseen_test", unseen_df)]:
        missing = {"image_path", "label_path", "damage_mask_path"} - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing required columns for full evaluation: {missing}")

    return seen_df, unseen_df


def print_event_distribution(name: str, df: pd.DataFrame):
    print(f"\n{name}: {len(df)} samples")
    if "event_name" in df.columns:
        counts = df["event_name"].value_counts().sort_index()
        for ev, cnt in counts.items():
            print(f"  - {ev}: {cnt}")


def load_checkpoint_and_model(cfg: Dict, fold_dir: Path, device):
    project_root = Path(cfg["project_root"])
    ckpt_path = (
        project_root
        / Path(cfg["model_output_subdir"])
        / fold_dir.name
        / "checkpoints"
        / cfg["checkpoint_name"]
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    train_cfg = ckpt.get("config", {})

    model_cfg = dict(CONFIG)
    model_cfg.update(train_cfg)

    # Evaluation config overrides.
    for key in [
        "project_root",
        "splits_root",
        "model_output_subdir",
        "eval_output_subdir",
        "checkpoint_name",
        "fold_mode",
        "fold_name",
        "fold_names",
        "seen_test_csv_name",
        "unseen_test_csv_name",
        "batch_size",
        "num_workers",
        "pin_memory",
        "skip_missing_checkpoints",
        "print_confusion_matrices",
        "save_prediction_examples",
    ]:
        model_cfg[key] = cfg[key]

    model = build_model(model_cfg).to(device)
    load_style = load_state_dict_flexible(model, ckpt["model_state_dict"])
    model.eval()

    return model, model_cfg, ckpt_path, load_style


def make_seen_unseen_gap_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "pixel_mIoU",
        "pixel_macroF1",
        "pixel_mIoU_fg",
        "pixel_macroF1_fg",
        "pixel_balanced_mIoU",
        "building_accuracy_overall",
        "building_macroF1",
        "building_macro_accuracy",
    ]

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


def make_aggregate_tables(results_df: pd.DataFrame, output_root: Path):
    agg = results_df.groupby("split").agg(
        num_folds=("fold", "count"),
        samples_mean=("num_samples", "mean"),

        pixel_mIoU_mean=("pixel_mIoU", "mean"),
        pixel_mIoU_std=("pixel_mIoU", "std"),
        pixel_macroF1_mean=("pixel_macroF1", "mean"),
        pixel_macroF1_std=("pixel_macroF1", "std"),
        pixel_mIoU_fg_mean=("pixel_mIoU_fg", "mean"),
        pixel_mIoU_fg_std=("pixel_mIoU_fg", "std"),
        pixel_balanced_mIoU_mean=("pixel_balanced_mIoU", "mean"),
        pixel_balanced_mIoU_std=("pixel_balanced_mIoU", "std"),

        building_num_buildings_sum=("building_num_buildings", "sum"),
        building_accuracy_overall_mean=("building_accuracy_overall", "mean"),
        building_accuracy_overall_std=("building_accuracy_overall", "std"),
        building_macroF1_mean=("building_macroF1", "mean"),
        building_macroF1_std=("building_macroF1", "std"),
        building_macro_accuracy_mean=("building_macro_accuracy", "mean"),
        building_macro_accuracy_std=("building_macro_accuracy", "std"),
    ).reset_index()

    agg.to_csv(output_root / "all_folds_seen_unseen_pixel_building_summary_agg.csv", index=False)

    gap = make_seen_unseen_gap_table(results_df)
    gap.to_csv(output_root / "all_folds_seen_unseen_pixel_building_gap_by_fold.csv", index=False)

    if len(gap) > 0:
        gap_metrics = [c for c in gap.columns if c.startswith("drop_")]
        gap_agg = gap[gap_metrics].agg(["mean", "std", "min", "max"]).T.reset_index()
        gap_agg = gap_agg.rename(columns={"index": "metric"})
        gap_agg.to_csv(output_root / "all_folds_seen_unseen_pixel_building_gap_agg.csv", index=False)
    else:
        gap_agg = pd.DataFrame()

    return agg, gap, gap_agg


def evaluate_one_fold(cfg: Dict, fold_dir: Path, fold_index: int):
    set_seed(int(cfg["seed"]) + fold_index)

    project_root = Path(cfg["project_root"])
    output_root = project_root / Path(cfg["eval_output_subdir"])
    output_dir = output_root / fold_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_header(f"Week 4 SegFormer patch-sampling full eval | {fold_dir.name}")
    print("fold dir  :", fold_dir)
    print("output dir:", output_dir)
    print("device    :", device)

    seen_df, unseen_df = read_test_data(fold_dir, cfg)
    print_event_distribution("seen_test", seen_df)
    print_event_distribution("unseen_test", unseen_df)

    try:
        model, model_cfg, ckpt_path, load_style = load_checkpoint_and_model(cfg, fold_dir, device)
    except FileNotFoundError as e:
        if bool(cfg.get("skip_missing_checkpoints", True)):
            print(f"[SKIP] {e}")
            return pd.DataFrame(), pd.DataFrame()
        raise

    print("checkpoint:", ckpt_path)
    print("checkpoint load style:", load_style)

    with open(output_dir / "eval_config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)

    fold_rows = []
    damage_class_rows = []

    for split_name, split_df in [
        ("seen_test", seen_df),
        ("unseen_test", unseen_df),
    ]:
        print_header(f"Evaluate {split_name}")
        summary_row, damage_class_df = evaluate_split(
            model=model,
            df=split_df,
            cfg=model_cfg,
            device=device,
            split_name=split_name,
            fold_name=fold_dir.name,
            output_dir=output_dir,
        )
        fold_rows.append(summary_row)
        damage_class_rows.append(damage_class_df)

    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(output_dir / "fold_seen_unseen_pixel_building_summary.csv", index=False)

    fold_damage_class_summary = pd.concat(damage_class_rows, ignore_index=True) if damage_class_rows else pd.DataFrame()
    fold_damage_class_summary.to_csv(output_dir / "fold_seen_unseen_damage_class_summary.csv", index=False)

    gap = make_seen_unseen_gap_table(fold_summary)
    gap.to_csv(output_dir / "fold_seen_unseen_pixel_building_gap.csv", index=False)

    if len(gap) > 0:
        print_header("Seen-to-unseen drop for this fold")
        print(gap)

    return fold_summary, fold_damage_class_summary


def main():
    cfg = CONFIG
    fold_dirs = find_fold_dirs(cfg)

    print_header("Week 4 SegFormer patch-sampling full pixel + building evaluation plan")
    for i, fd in enumerate(fold_dirs):
        print(f"{i}: {fd}")

    all_summary_rows = []
    all_damage_class_rows = []

    for fold_index, fold_dir in enumerate(fold_dirs):
        fold_summary, fold_damage_class_summary = evaluate_one_fold(cfg, fold_dir, fold_index)
        if fold_summary is not None and len(fold_summary) > 0:
            all_summary_rows.append(fold_summary)
        if fold_damage_class_summary is not None and len(fold_damage_class_summary) > 0:
            all_damage_class_rows.append(fold_damage_class_summary)

    project_root = Path(cfg["project_root"])
    output_root = project_root / Path(cfg["eval_output_subdir"])
    output_root.mkdir(parents=True, exist_ok=True)

    if not all_summary_rows:
        raise RuntimeError(
            "No folds were evaluated. Train at least one fold first, or set fold_mode='single' "
            "for a completed fold."
        )

    results_df = pd.concat(all_summary_rows, ignore_index=True)
    results_df.to_csv(output_root / "all_folds_seen_unseen_pixel_building_summary.csv", index=False)

    damage_class_df = pd.concat(all_damage_class_rows, ignore_index=True) if all_damage_class_rows else pd.DataFrame()
    damage_class_df.to_csv(output_root / "all_folds_seen_unseen_damage_class_summary.csv", index=False)

    if len(damage_class_df) > 0:
        damage_class_agg = damage_class_df.groupby(["split", "class_id", "class_name"]).agg(
            pixel_IoU_mean=("pixel_IoU", "mean"),
            pixel_IoU_std=("pixel_IoU", "std"),
            pixel_F1_mean=("pixel_F1", "mean"),
            pixel_F1_std=("pixel_F1", "std"),
            building_accuracy_mean=("building_accuracy", "mean"),
            building_accuracy_std=("building_accuracy", "std"),
            building_F1_mean=("building_F1", "mean"),
            building_F1_std=("building_F1", "std"),
            building_precision_mean=("building_precision", "mean"),
            building_precision_std=("building_precision", "std"),
        ).reset_index()
        damage_class_agg.to_csv(output_root / "all_folds_seen_unseen_damage_class_summary_agg.csv", index=False)
    else:
        damage_class_agg = pd.DataFrame()

    agg, gap, gap_agg = make_aggregate_tables(results_df, output_root)

    print_header("All-fold seen/unseen pixel + building summary")
    print(results_df)

    print_header("Aggregate by split")
    print(agg)

    print_header("Seen-to-unseen gap by fold")
    print(gap)

    if len(gap_agg) > 0:
        print_header("Seen-to-unseen gap aggregate")
        print(gap_agg)

    if len(damage_class_agg) > 0:
        print_header("Per-damage-class aggregate")
        print(damage_class_agg)

    print_header("Saved outputs")
    print("summary:", output_root / "all_folds_seen_unseen_pixel_building_summary.csv")
    print("summary agg:", output_root / "all_folds_seen_unseen_pixel_building_summary_agg.csv")
    print("gap by fold:", output_root / "all_folds_seen_unseen_pixel_building_gap_by_fold.csv")
    print("gap agg:", output_root / "all_folds_seen_unseen_pixel_building_gap_agg.csv")
    print("damage class summary:", output_root / "all_folds_seen_unseen_damage_class_summary.csv")
    print("damage class summary agg:", output_root / "all_folds_seen_unseen_damage_class_summary_agg.csv")
    print("\nPer-fold folders also contain:")
    print("  *_pixel_metrics.json")
    print("  *_pixel_per_class_iou.csv")
    print("  *_pixel_confusion_matrix.csv")
    print("  *_building_metrics.json")
    print("  *_building_per_class_metrics.csv")
    print("  *_building_confusion_matrix.csv")


if __name__ == "__main__":
    main()
