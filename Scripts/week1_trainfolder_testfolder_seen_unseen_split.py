from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from sklearn.model_selection import train_test_split

CONFIG = {
    "project_root": "/workspace/ECE6364",

    # Train/val samples come only from this folder.
    "train_subdir": "Data/train",

    # Seen/unseen test samples come only from this separate folder.
    "test_subdir": "Data/test",

    "images_subdir": "images",
    "labels_subdir": "labels",
    "binary_targets_subdir": "targets",
    "damage_masks_subdir": "damage_masks",
    "outputs_subdir": "Outputs/week1_trainfolder_testfolder_seen_unseen_split",

    "make_damage_masks": True,
    "overwrite_damage_masks": True,
    "image_size": (1024, 1024),

    "seed": 42,
    "val_ratio": 0.20,
    "split_mode": "train_folder_val_and_test_folder_seen_unseen_eval",
    "use_only_post_disaster": True,

    # Five folds, two unseen events per fold.
    # For each fold:
    #   - train/val: Data/train excluding these events
    #   - unseen_test: Data/test samples from these events
    #   - seen_test: Data/test samples from all other events
    "fold_holdout_events": [
        ["socal-fire", "guatemala-volcano"],
        ["hurricane-michael", "palu-tsunami"],
        ["hurricane-harvey", "mexico-earthquake"],
        ["hurricane-florence", "santa-rosa-wildfire"],
        ["midwest-flooding", "hurricane-matthew"],
    ],

    "num_visual_checks": 8,
}

CLASS_MAP = {
    "background": 0,
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 0,
}


def print_header(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def infer_event_name(stem: str) -> str:
    stem = stem.replace("_pre_disaster", "").replace("_post_disaster", "")
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def polygon_to_cv2_pts(poly: Polygon):
    coords = np.array(poly.exterior.coords, dtype=np.float32)
    coords = np.round(coords).astype(np.int32)
    return coords.reshape((-1, 1, 2))


def fill_geometry_on_mask(mask: np.ndarray, geom, class_id: int):
    if isinstance(geom, Polygon):
        pts = polygon_to_cv2_pts(geom)
        cv2.fillPoly(mask, [pts], int(class_id))
        for interior in geom.interiors:
            hole = np.array(interior.coords, dtype=np.float32)
            hole = np.round(hole).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [hole], 0)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            fill_geometry_on_mask(mask, poly, class_id)


def make_mask_from_json(json_path: Path, image_shape=(1024, 1024)) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(json_path, "r") as f:
        data = json.load(f)

    xy_features = data.get("features", {}).get("xy", [])
    for feat in xy_features:
        props = feat.get("properties", {})
        subtype = props.get("subtype", "background")
        geom_wkt = feat.get("wkt", None)
        if geom_wkt is None:
            continue

        class_id = CLASS_MAP.get(subtype, 0)
        try:
            geom = wkt.loads(geom_wkt)
            fill_geometry_on_mask(mask, geom, class_id)
        except Exception as e:
            print(f"[WARN] Failed on {json_path.name}: {e}")

    return mask


def inspect_json_schema(label_path: Path) -> dict:
    with open(label_path, "r") as f:
        data = json.load(f)

    features = data.get("features", {})
    out = {
        "top_level_keys": list(data.keys()),
        "feature_keys": list(features.keys()) if isinstance(features, dict) else [],
        "xy_count": len(features.get("xy", [])) if isinstance(features, dict) else 0,
        "lng_lat_count": len(features.get("lng_lat", [])) if isinstance(features, dict) else 0,
        "sample_xy_feature": features.get("xy", [None])[0] if isinstance(features, dict) and features.get("xy") else None,
    }
    return out


def build_or_verify_masks(split_root: Path, cfg: dict, split_name: str):
    lbl_dir = split_root / cfg["labels_subdir"]
    damage_dir = split_root / cfg["damage_masks_subdir"]
    damage_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(lbl_dir.glob("*.json")) if lbl_dir.exists() else []

    generated = 0
    skipped = 0
    if cfg["make_damage_masks"]:
        print_header(f"Build / verify 5-class damage masks for {split_name}")
        print("label dir   :", lbl_dir)
        print("damage masks:", damage_dir)
        print("num labels  :", len(label_files))

        for json_path in label_files:
            stem = json_path.stem
            out_path = damage_dir / f"{stem}_damage_mask.png"
            if out_path.exists() and not cfg["overwrite_damage_masks"]:
                skipped += 1
                continue
            mask = make_mask_from_json(json_path, image_shape=cfg["image_size"])
            cv2.imwrite(str(out_path), mask)
            generated += 1

        print("generated:", generated)
        print("skipped  :", skipped)

    return generated, skipped


def build_manifest(split_root: Path, cfg: dict, split_source: str) -> pd.DataFrame:
    img_dir = split_root / cfg["images_subdir"]
    lbl_dir = split_root / cfg["labels_subdir"]
    damage_dir = split_root / cfg["damage_masks_subdir"]

    rows = []
    pattern = "*post_disaster*.png" if cfg["use_only_post_disaster"] else "*.png"

    for img_path in sorted(img_dir.glob(pattern)):
        stem = img_path.stem
        label_path = lbl_dir / f"{stem}.json"
        damage_mask_path = damage_dir / f"{stem}_damage_mask.png"

        if not label_path.exists():
            continue
        if not damage_mask_path.exists():
            continue

        event_name = infer_event_name(stem)
        rows.append({
            "image_path": str(img_path),
            "label_path": str(label_path),
            "damage_mask_path": str(damage_mask_path),
            "event_name": event_name,
            "image_id": stem,
            "source_folder": split_source,
        })

    return pd.DataFrame(rows)


def stratified_train_val_split(df: pd.DataFrame, val_ratio: float, seed: int):
    if len(df) == 0:
        raise RuntimeError("Cannot split an empty train/val dataframe.")

    indices = np.arange(len(df))
    if df["event_name"].nunique() > 1:
        try:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=seed,
                shuffle=True,
                stratify=df["event_name"].astype(str),
            )
        except Exception as e:
            print(f"[WARN] Stratified train/val split failed, using random split instead: {e}")
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=seed,
                shuffle=True,
            )
    else:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    return train_df, val_df


def validate_fold_design(fold_holdout_events: list[list[str]], train_events: list[str], test_events: list[str]):
    all_events = sorted(set(train_events) | set(test_events))
    flat = [ev for pair in fold_holdout_events for ev in pair]

    duplicates = sorted([ev for ev in set(flat) if flat.count(ev) > 1])
    missing = sorted(list(set(all_events) - set(flat)))
    unknown = sorted(list(set(flat) - set(all_events)))

    if duplicates:
        print(f"[WARN] Some events appear in more than one holdout fold: {duplicates}")
    if missing:
        print(f"[WARN] Some available events are never held out as unseen: {missing}")
    if unknown:
        raise ValueError(f"Configured holdout events not found in train/test data: {unknown}")


def colorize_mask(mask):
    palette = np.array([
        [0, 0, 0],        # 0 background -> black
        [0, 255, 0],      # 1 no-damage -> green
        [255, 255, 0],    # 2 minor-damage -> yellow
        [255, 165, 0],    # 3 major-damage -> orange
        [255, 0, 0],      # 4 destroyed -> red
    ], dtype=np.uint8)

    mask = np.clip(mask.astype(np.int64), 0, 4)
    return palette[mask]

def save_visual_checks(df: pd.DataFrame, out_dir: Path, num_samples: int = 8):
    import matplotlib.pyplot as plt

    vis_dir = out_dir / "visual_checks"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        return 0

    sample_df = df.sample(min(num_samples, len(df)), random_state=CONFIG["seed"]).reset_index(drop=True)

    saved = 0
    for i, row in sample_df.iterrows():
        img = cv2.imread(str(Path(row["image_path"])), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(Path(row["damage_mask_path"])), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(img)
        axes[0].set_title(f"image ({row['source_folder']})")
        axes[0].axis("off")

        axes[1].imshow(colorize_mask(mask))
        axes[1].set_title("damage mask")
        axes[1].axis("off")

        overlay = img.copy()
        color_mask = np.zeros_like(img)
        palette = {
            1: [0, 255, 0],
            2: [255, 255, 0],
            3: [255, 165, 0],
            4: [255, 0, 0],
        }
        for cls_id, color in palette.items():
            color_mask[mask == cls_id] = color
        alpha = 0.35
        overlay = np.where(color_mask > 0, (1 - alpha) * overlay + alpha * color_mask, overlay).astype(np.uint8)

        axes[2].imshow(overlay)
        axes[2].set_title("overlay")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = vis_dir / f"sample_{i:02d}_{row['source_folder']}_{Path(row['image_path']).stem}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    return saved


def main():
    cfg = CONFIG

    project_root = Path(cfg["project_root"])
    train_root = project_root / Path(cfg["train_subdir"])
    test_root = project_root / Path(cfg["test_subdir"])
    out_dir = project_root / Path(cfg["outputs_subdir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print_header("ECE 6364 - Week 1 pipeline (train-folder + test-folder seen/unseen split)")
    print("project root :", project_root)
    print("train root   :", train_root)
    print("test root    :", test_root)
    print("outputs dir  :", out_dir)
    print("split mode   :", cfg["split_mode"])
    print("val ratio    :", cfg["val_ratio"])
    print("seed         :", cfg["seed"])

    for name, root in [("train", train_root), ("test", test_root)]:
        print_header(f"Inspect {name} dataset")
        img_dir = root / cfg["images_subdir"]
        lbl_dir = root / cfg["labels_subdir"]
        tgt_dir = root / cfg["binary_targets_subdir"]
        print("root exists  :", root.exists())
        print("images exists:", img_dir.exists())
        print("labels exists:", lbl_dir.exists())
        print("targets exists:", tgt_dir.exists())
        print("num images        :", len(sorted(img_dir.glob("*"))) if img_dir.exists() else 0)
        print("num labels        :", len(sorted(lbl_dir.glob("*.json"))) if lbl_dir.exists() else 0)
        print("num binary targets:", len(sorted(tgt_dir.glob("*"))) if tgt_dir.exists() else 0)

    schema_info = {}
    train_label_files = sorted((train_root / cfg["labels_subdir"]).glob("*.json"))
    if train_label_files:
        print_header("Inspect JSON schema")
        schema_info = inspect_json_schema(train_label_files[0])
        print("top-level keys:", schema_info["top_level_keys"])
        print("feature keys  :", schema_info["feature_keys"])
        print("xy count      :", schema_info["xy_count"])
        print("lng_lat count :", schema_info["lng_lat_count"])
        if schema_info["sample_xy_feature"] is not None:
            print("sample xy feature:")
            print(json.dumps(schema_info["sample_xy_feature"], indent=2)[:1200])

    train_generated, train_skipped = build_or_verify_masks(train_root, cfg, "Data/train")
    test_generated, test_skipped = build_or_verify_masks(test_root, cfg, "Data/test")

    train_manifest = build_manifest(train_root, cfg, "train_folder")
    test_manifest = build_manifest(test_root, cfg, "test_folder")

    print_header("Manifest summaries")
    print("train-folder post-disaster samples:", len(train_manifest))
    if len(train_manifest):
        print("train-folder events:", train_manifest["event_name"].nunique())
        print(train_manifest["event_name"].value_counts().sort_index())
    print("\ntest-folder post-disaster samples:", len(test_manifest))
    if len(test_manifest):
        print("test-folder events:", test_manifest["event_name"].nunique())
        print(test_manifest["event_name"].value_counts().sort_index())

    if len(train_manifest) == 0:
        raise RuntimeError("Train manifest is empty. Check Data/train paths, labels, and masks.")
    if len(test_manifest) == 0:
        raise RuntimeError("Test manifest is empty. Check Data/test paths, labels, and masks.")

    train_events = sorted(train_manifest["event_name"].unique().tolist())
    test_events = sorted(test_manifest["event_name"].unique().tolist())
    validate_fold_design(cfg["fold_holdout_events"], train_events, test_events)

    combined_for_visuals = pd.concat([train_manifest, test_manifest], ignore_index=True)
    saved_visuals = save_visual_checks(combined_for_visuals, out_dir, num_samples=cfg["num_visual_checks"])
    print_header("Visual checks")
    print("saved visual checks:", saved_visuals)

    print_header("5-fold train-folder/test-folder seen-unseen summary")
    all_fold_dfs = []
    split_ids_by_fold = {}
    summary_folds = {}

    for fold_idx, holdout_events in enumerate(cfg["fold_holdout_events"]):
        holdout_set = set(holdout_events)
        fold_name = f"fold_{fold_idx:02d}_holdout_" + "__".join(holdout_events)
        fold_dir = out_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Train and validation are from Data/train only, excluding holdout events.
        trainval_pool = train_manifest[~train_manifest["event_name"].isin(holdout_set)].copy().reset_index(drop=True)
        removed_from_train = train_manifest[train_manifest["event_name"].isin(holdout_set)].copy().reset_index(drop=True)
        train_df, val_df = stratified_train_val_split(trainval_pool, cfg["val_ratio"], cfg["seed"] + fold_idx)

        # Evaluation is from Data/test only.
        unseen_test_df = test_manifest[test_manifest["event_name"].isin(holdout_set)].copy().reset_index(drop=True)
        seen_test_df = test_manifest[~test_manifest["event_name"].isin(holdout_set)].copy().reset_index(drop=True)

        if len(unseen_test_df) == 0:
            print(f"[WARN] Fold {fold_idx} unseen_test is empty. Check whether {holdout_events} exist in Data/test.")
        if len(seen_test_df) == 0:
            print(f"[WARN] Fold {fold_idx} seen_test is empty. Check Data/test event coverage.")

        train_df["split"] = "train"
        val_df["split"] = "val"
        seen_test_df["split"] = "seen_test"
        unseen_test_df["split"] = "unseen_test"
        removed_from_train["split"] = "excluded_from_train_pool"

        for part in [train_df, val_df, seen_test_df, unseen_test_df, removed_from_train]:
            part["fold"] = fold_idx
            part["holdout_events"] = ",".join(holdout_events)

        fold_df = pd.concat([train_df, val_df, seen_test_df, unseen_test_df, removed_from_train], ignore_index=True)
        all_fold_dfs.append(fold_df)

        train_csv = fold_dir / "train.csv"
        val_csv = fold_dir / "val.csv"
        seen_test_csv = fold_dir / "seen_test.csv"
        unseen_test_csv = fold_dir / "unseen_test.csv"
        excluded_train_csv = fold_dir / "excluded_train_pool_events.csv"
        combined_test_csv = fold_dir / "test_combined_seen_unseen.csv"
        manifest_csv = fold_dir / "manifest.csv"

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        seen_test_df.to_csv(seen_test_csv, index=False)
        unseen_test_df.to_csv(unseen_test_csv, index=False)
        removed_from_train.to_csv(excluded_train_csv, index=False)
        pd.concat([seen_test_df, unseen_test_df], ignore_index=True).to_csv(combined_test_csv, index=False)
        fold_df.to_csv(manifest_csv, index=False)

        print(f"{fold_name}")
        print(f"  holdout events removed from train/val: {holdout_events}")
        print(f"  train={len(train_df)}, val={len(val_df)}, seen_test={len(seen_test_df)}, unseen_test={len(unseen_test_df)}")
        print(f"  excluded train-pool samples={len(removed_from_train)}")
        print(f"  unseen_test event counts:\n{unseen_test_df['event_name'].value_counts().sort_index() if len(unseen_test_df) else 'EMPTY'}")

        key = fold_name
        split_ids_by_fold[key] = {
            "train": sorted(train_df["image_id"].tolist()),
            "val": sorted(val_df["image_id"].tolist()),
            "seen_test": sorted(seen_test_df["image_id"].tolist()),
            "unseen_test": sorted(unseen_test_df["image_id"].tolist()),
            "excluded_from_train_pool": sorted(removed_from_train["image_id"].tolist()),
        }

        summary_folds[key] = {
            "fold": fold_idx,
            "holdout_events": holdout_events,
            "num_train": int(len(train_df)),
            "num_val": int(len(val_df)),
            "num_seen_test": int(len(seen_test_df)),
            "num_unseen_test": int(len(unseen_test_df)),
            "num_excluded_from_train_pool": int(len(removed_from_train)),
            "train_events": sorted(train_df["event_name"].unique().tolist()),
            "val_events": sorted(val_df["event_name"].unique().tolist()),
            "seen_test_events": sorted(seen_test_df["event_name"].unique().tolist()),
            "unseen_test_events": sorted(unseen_test_df["event_name"].unique().tolist()),
            "excluded_train_pool_events": sorted(removed_from_train["event_name"].unique().tolist()),
            "train_event_counts": train_df["event_name"].value_counts().sort_index().to_dict(),
            "val_event_counts": val_df["event_name"].value_counts().sort_index().to_dict(),
            "seen_test_event_counts": seen_test_df["event_name"].value_counts().sort_index().to_dict(),
            "unseen_test_event_counts": unseen_test_df["event_name"].value_counts().sort_index().to_dict(),
            "excluded_train_pool_event_counts": removed_from_train["event_name"].value_counts().sort_index().to_dict(),
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "seen_test_csv": str(seen_test_csv),
            "unseen_test_csv": str(unseen_test_csv),
            "combined_test_csv": str(combined_test_csv),
            "excluded_train_pool_csv": str(excluded_train_csv),
            "manifest_csv": str(manifest_csv),
        }

    all_folds_manifest = pd.concat(all_fold_dfs, ignore_index=True)
    all_folds_manifest_path = out_dir / "week1_trainfolder_testfolder_seen_unseen_all_folds_manifest.csv"
    all_folds_manifest.to_csv(all_folds_manifest_path, index=False)

    with open(out_dir / "split_image_ids_by_fold.json", "w") as f:
        json.dump(split_ids_by_fold, f, indent=2)

    summary = {
        "config": cfg,
        "train_masks_generated": train_generated,
        "train_masks_skipped": train_skipped,
        "test_masks_generated": test_generated,
        "test_masks_skipped": test_skipped,
        "num_train_folder_post_samples": int(len(train_manifest)),
        "num_test_folder_post_samples": int(len(test_manifest)),
        "train_folder_event_counts": train_manifest["event_name"].value_counts().sort_index().to_dict(),
        "test_folder_event_counts": test_manifest["event_name"].value_counts().sort_index().to_dict(),
        "json_schema": schema_info,
        "visual_checks_saved": saved_visuals,
        "folds": summary_folds,
        "notes": [
            "Train/validation samples are drawn only from Data/train.",
            "Test samples are drawn only from Data/test.",
            "For each fold, the two holdout events are removed completely from train/validation.",
            "seen_test contains Data/test samples from events present in training.",
            "unseen_test contains Data/test samples from held-out events absent from training/validation.",
            "Evaluate seen_test and unseen_test separately to quantify event-level generalization."
        ]
    }

    summary_path = out_dir / "week1_trainfolder_testfolder_seen_unseen_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_header("Saved outputs")
    print("summary              :", summary_path)
    print("all folds manifest   :", all_folds_manifest_path)
    print("split ids by fold    :", out_dir / "split_image_ids_by_fold.json")
    print("fold-specific folders:", out_dir / "fold_XX_holdout_eventA__eventB")
    print("\nDone. Train once per fold using train.csv and val.csv. Evaluate seen_test.csv and unseen_test.csv separately.")


if __name__ == "__main__":
    main()
