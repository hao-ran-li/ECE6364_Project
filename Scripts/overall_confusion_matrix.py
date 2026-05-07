from pathlib import Path
import pandas as pd
import numpy as np

CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]

# Change this depending on model
EVAL_ROOT = Path(
    "/workspace/ECE6364/Outputs/week3_segformer_weighted_ce_seen_unseen_eval_full"
)

# Options:
# U-Net weighted CE:
# /workspace/ECE6364/Outputs/week2_unet_weighted_ce_seen_unseen_eval_full
#
# SegFormer weighted CE:
# /workspace/ECE6364/Outputs/week3_segformer_weighted_ce_seen_unseen_eval_full
#
# U-Net patch sampling:
# /workspace/ECE6364/Outputs/week4_unet_patch_sampling_seen_unseen_eval_full
#
# SegFormer patch sampling:
# /workspace/ECE6364/Outputs/week4_segformer_patch_sampling_seen_unseen_eval_full


def aggregate_cm(split_name, level):
    """
    level = "pixel" or "building"
    """
    total_cm = pd.DataFrame(0, index=CLASS_NAMES, columns=CLASS_NAMES, dtype=np.int64)

    pattern = f"fold_*/{split_name}_{level}_confusion_matrix.csv"
    paths = sorted(EVAL_ROOT.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No files found for pattern: {EVAL_ROOT / pattern}")

    print(f"\nFound {len(paths)} {level} confusion matrices for {split_name}")

    for p in paths:
        cm = pd.read_csv(p, index_col=0)
        cm = cm.reindex(index=CLASS_NAMES, columns=CLASS_NAMES).fillna(0).astype(np.int64)
        total_cm = total_cm.add(cm, fill_value=0).astype(np.int64)

    # Raw all-fold count matrix
    raw_out = EVAL_ROOT / f"all_folds_{split_name}_{level}_confusion_matrix_raw.csv"
    total_cm.to_csv(raw_out)

    # Row-normalized matrix
    row_sums = total_cm.sum(axis=1).replace(0, np.nan)
    norm_cm = total_cm.div(row_sums, axis=0).fillna(0)

    norm_out = EVAL_ROOT / f"all_folds_{split_name}_{level}_confusion_matrix_row_normalized.csv"
    norm_cm.to_csv(norm_out)

    print(f"\n{split_name} {level} raw confusion matrix:")
    print(total_cm)

    print(f"\n{split_name} {level} row-normalized confusion matrix:")
    print(norm_cm.round(3))

    print("\nSaved:")
    print(raw_out)
    print(norm_out)


def main():
    for split in ["seen_test", "unseen_test"]:
        for level in ["pixel", "building"]:
            aggregate_cm(split, level)


if __name__ == "__main__":
    main()