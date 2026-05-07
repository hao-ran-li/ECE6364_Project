"""
Plot building confusion matrices per model (fold_04), seen vs unseen side by side.
Background class excluded. Produces one row per model — no aggregation across models.
"""

import os
import glob
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.join(os.path.dirname(__file__), "..", "Outputs")

TEST_FOLDERS = [
    "week2_test_unet_weighted_ce",
    "week3_test_segformer_weighted_ce",
    "week4_test_unet_patch_sampling",
    "week4_test_segformer_patch_sampling",
]

MODEL_LABELS = {
    "week2_test_unet_weighted_ce":        "UNet (Weighted CE)",
    "week3_test_segformer_weighted_ce":   "SegFormer (Weighted CE)",
    "week4_test_unet_patch_sampling":     "UNet (Patch Sampling)",
    "week4_test_segformer_patch_sampling": "SegFormer (Patch Sampling)",
}

DMG_CLASS_LABELS = ["No-damage", "Minor-damage", "Major-damage", "Destroyed"]


def load_cm(folder, split_prefix):
    """Load raw building confusion matrix from fold_04 of a single test folder."""
    fold_dirs = glob.glob(os.path.join(BASE, folder, "fold_04_*"))
    if not fold_dirs:
        print(f"[WARN] no fold_04 dir in {folder}")
        return None
    cm_path = os.path.join(fold_dirs[0], f"{split_prefix}_building_confusion_matrix.csv")
    if not os.path.isfile(cm_path):
        print(f"[WARN] missing: {cm_path}")
        return None
    return pd.read_csv(cm_path, index_col=0).values.astype(float)


def plot_cm(ax, cm_raw, title):
    cm_dmg = cm_raw[1:, 1:]  # drop background row/col → 4×4 damage classes
    row_sums = cm_dmg.sum(axis=1, keepdims=True)
    cm_norm = cm_dmg / np.where(row_sums == 0, 1, row_sums)

    cmap = LinearSegmentedColormap.from_list("wb", ["#ffffff", "#1a4f8a"])
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap=cmap, aspect="equal")

    n = len(DMG_CLASS_LABELS)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(DMG_CLASS_LABELS, rotation=35, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(DMG_CLASS_LABELS, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
    ax.set_ylabel("True", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold")

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
    return im


def main():
    n_models = len(TEST_FOLDERS)
    # One row per model, columns for Seen and Unseen Test
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 5.5 * n_models))
    # fig.suptitle("Normalized Building Confusion Matrices (fold_04, per model, No BG)",
    #              fontsize=13, fontweight="bold", y=1.01)

    for row, folder in enumerate(TEST_FOLDERS):
        model_label = MODEL_LABELS[folder]
        seen_cm   = load_cm(folder, "seen_test")
        unseen_cm = load_cm(folder, "unseen_test")

        ax_s = axes[row, 0]
        ax_u = axes[row, 1]

        if seen_cm is not None:
            im_s = plot_cm(ax_s, seen_cm, f"{model_label}\nSeen Test")
            plt.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04, label="Recall")
        else:
            ax_s.set_visible(False)

        if unseen_cm is not None:
            im_u = plot_cm(ax_u, unseen_cm, f"{model_label}\nUnseen Test")
            plt.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04, label="Recall")
        else:
            ax_u.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(BASE, "building_confusion_matrices_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
