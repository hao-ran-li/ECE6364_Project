"""
Aggregate all_folds_seen_unseen_damage_class_summary.csv from each test folder
and plot pixel mIoU and building F1 for seen vs unseen splits.
"""

import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "Outputs")

TEST_FOLDERS = {
    "UNet\nWeighted CE":        "week2_test_unet_weighted_ce",
    "SegFormer\nWeighted CE":   "week3_test_segformer_weighted_ce",
    "UNet\nPatch Sampling":     "week4_test_unet_patch_sampling",
    "SegFormer\nPatch Sampling":"week4_test_segformer_patch_sampling",
}

CSV_NAME = "all_folds_seen_unseen_damage_class_summary.csv"


def load_metrics(folder_path: str) -> dict:
    """Return mean pixel mIoU and building F1 for seen/unseen across all folds."""
    csv_path = os.path.join(folder_path, CSV_NAME)
    df = pd.read_csv(csv_path)

    results = {}
    for split in ("seen_test", "unseen_test"):
        sub = df[df["split"] == split]

        # pixel mIoU: mean IoU over all classes (including background) per fold, then mean across folds
        miou_per_fold = sub.groupby("fold")["pixel_IoU"].mean()

        # building F1: exclude background (class_id == 0), mean over damage classes per fold, then across folds
        dmg = sub[sub["class_id"] != 0]
        f1_per_fold = dmg.groupby("fold")["building_F1"].mean()

        key = split.replace("_test", "")
        results[f"{key}_miou_mean"] = miou_per_fold.mean()
        results[f"{key}_miou_std"]  = miou_per_fold.std()
        results[f"{key}_f1_mean"]   = f1_per_fold.mean()
        results[f"{key}_f1_std"]    = f1_per_fold.std()
        results[f"{key}_n_folds"]   = len(miou_per_fold)

    return results


def main():
    records = {}
    for label, folder in TEST_FOLDERS.items():
        path = os.path.join(BASE, folder)
        if not os.path.isdir(path):
            print(f"[WARN] folder not found: {path}")
            continue
        metrics = load_metrics(path)
        records[label] = metrics

    labels = list(records.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Seen vs Unseen Performance Across Models", fontsize=14, fontweight="bold")

    model_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]  # blue, green, red, purple

    plot_specs = [
        (axes[0], "miou", "Pixel mIoU (incl. background)", "Mean IoU"),
        (axes[1], "f1",   "Building F1 (damage classes only)", "Mean F1"),
    ]

    # collect all values to set a shared y-axis
    all_means = []
    for _, metric, _, _ in plot_specs:
        for l in labels:
            all_means += [records[l][f"seen_{metric}_mean"], records[l][f"unseen_{metric}_mean"]]
    y_max = min(1.0, max(all_means) * 1.4)

    for ax, metric, title, ylabel in plot_specs:
        seen_means   = [records[l][f"seen_{metric}_mean"]   for l in labels]
        seen_stds    = [records[l][f"seen_{metric}_std"]    for l in labels]
        unseen_means = [records[l][f"unseen_{metric}_mean"] for l in labels]
        unseen_stds  = [records[l][f"unseen_{metric}_std"]  for l in labels]

        for i, (color, l) in enumerate(zip(model_colors, labels)):
            bar_s = ax.bar(x[i] - width/2, seen_means[i],   width,
                           color=color, alpha=0.9,
                           yerr=seen_stds[i],   capsize=4, error_kw={"linewidth": 1.2},
                           label=f"{l.replace(chr(10), ' ')} – Seen")
            bar_u = ax.bar(x[i] + width/2, unseen_means[i], width,
                           color=color, alpha=0.45, hatch="///",
                           yerr=unseen_stds[i], capsize=4, error_kw={"linewidth": 1.2},
                           label=f"{l.replace(chr(10), ' ')} – Unseen")
            for bar in [bar_s[0], bar_u[0]]:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, y_max)
        ax.legend(fontsize=7.5, ncol=2, loc="upper right")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for i, l in enumerate(labels):
            n = records[l][f"seen_n_folds"]
            ax.text(x[i], -0.06, f"n={n}", ha="center", va="top",
                    fontsize=7, color="gray", transform=ax.get_xaxis_transform())

    plt.tight_layout()
    out_path = os.path.join(BASE, "results_summary_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
