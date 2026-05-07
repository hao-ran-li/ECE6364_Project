"""
Pixel-level metrics across all test folders.
Layout:
  top row:    Mean IoU (mIoU) | Macro F1
  bottom row: Per-class IoU (spanning full width)
Seen = solid bar, Unseen = same color + hatch.
"""

import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "Outputs")

TEST_FOLDERS = {
    "UNet\nWeighted CE":         "week2_test_unet_weighted_ce",
    "SegFormer\nWeighted CE":    "week3_test_segformer_weighted_ce",
    "UNet\nPatch Sampling":      "week4_test_unet_patch_sampling",
    "SegFormer\nPatch Sampling": "week4_test_segformer_patch_sampling",
}

CSV_NAME = "all_folds_seen_unseen_damage_class_summary.csv"
CLASS_ORDER  = [0, 1, 2, 3, 4]
CLASS_LABELS = ["Background", "No-damage", "Minor-damage", "Major-damage", "Destroyed"]
MODEL_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def load_df(folder_path):
    return pd.read_csv(os.path.join(folder_path, CSV_NAME))


def summary_metric(df, col, split):
    per_fold = df[df["split"] == split].groupby("fold")[col].mean()
    return per_fold.mean(), per_fold.std()


def per_class_metric(df, col, split):
    sub = df[df["split"] == split]
    return {cid: (sub[sub["class_id"] == cid].groupby("fold")[col].mean().mean(),
                  sub[sub["class_id"] == cid].groupby("fold")[col].mean().std())
            for cid in CLASS_ORDER}


def plot_summary_bars(ax, records, labels, col, title, ylabel):
    x = np.arange(len(labels))
    w = 0.35
    for i, (label, color) in enumerate(zip(labels, MODEL_COLORS)):
        sm, ss = summary_metric(records[label], col, "seen_test")
        um, us = summary_metric(records[label], col, "unseen_test")
        bs = ax.bar(x[i] - w/2, sm, w, color=color, alpha=0.9,
                    yerr=ss, capsize=4, error_kw={"linewidth": 1.1})
        bu = ax.bar(x[i] + w/2, um, w, color=color, alpha=0.45, hatch="///",
                    yerr=us, capsize=4, error_kw={"linewidth": 1.1})
        for bar, val in [(bs[0], sm), (bu[0], um)]:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.004,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    handles = (
        [mpatches.Patch(facecolor=c, alpha=0.9,     label=l.replace("\n", " ") + " – Seen")
         for c, l in zip(MODEL_COLORS, labels)] +
        [mpatches.Patch(facecolor=c, alpha=0.45, hatch="///", label=l.replace("\n", " ") + " – Unseen")
         for c, l in zip(MODEL_COLORS, labels)]
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=handles, fontsize=6.5, ncol=2, loc="upper right")


def plot_perclass_bars(ax, records, labels, col, title, ylabel):
    bar_w = 0.08
    model_gap = 0.05
    total_w = len(labels) * 2 * bar_w + (len(labels) - 1) * model_gap
    pos = -total_w / 2
    offsets = []
    for _ in labels:
        offsets.append((pos + bar_w/2, pos + bar_w + bar_w/2))
        pos += 2 * bar_w + model_gap

    x = np.arange(len(CLASS_ORDER)) * 1.3

    for m_idx, (label, color) in enumerate(zip(labels, MODEL_COLORS)):
        pc_seen   = per_class_metric(records[label], col, "seen_test")
        pc_unseen = per_class_metric(records[label], col, "unseen_test")
        so, uo = offsets[m_idx]
        lbl = label.replace("\n", " ")
        ax.bar(x + so, [pc_seen[c][0]   for c in CLASS_ORDER], bar_w,
               color=color, alpha=0.9,
               yerr=[pc_seen[c][1]   for c in CLASS_ORDER], capsize=2, error_kw={"linewidth": 0.8},
               label=f"{lbl} – Seen")
        ax.bar(x + uo, [pc_unseen[c][0] for c in CLASS_ORDER], bar_w,
               color=color, alpha=0.45, hatch="///",
               yerr=[pc_unseen[c][1] for c in CLASS_ORDER], capsize=2, error_kw={"linewidth": 0.8},
               label=f"{lbl} – Unseen")

    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS, fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")


def main():
    records = {}
    for label, folder in TEST_FOLDERS.items():
        path = os.path.join(BASE, folder)
        if not os.path.isdir(path):
            print(f"[WARN] not found: {path}")
            continue
        records[label] = load_df(path)

    labels = list(records.keys())

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Pixel-Level Metrics: Seen vs Unseen", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax_miou = fig.add_subplot(gs[0, 0])
    ax_f1   = fig.add_subplot(gs[0, 1])
    ax_iou  = fig.add_subplot(gs[1, :])   # spans both columns

    plot_summary_bars(ax_miou, records, labels, "pixel_IoU",
                      "Mean IoU (mIoU, all classes incl. background)", "Mean IoU")
    plot_summary_bars(ax_f1,   records, labels, "pixel_F1",
                      "Macro F1 (all classes incl. background)", "Mean F1")
    plot_perclass_bars(ax_iou,  records, labels, "pixel_IoU",
                       "Per-class IoU", "IoU")

    # shared y-axis
    y_max = min(1.0, max(ax.get_ylim()[1] for ax in [ax_miou, ax_f1, ax_iou]) * 1.05)
    for ax in [ax_miou, ax_f1, ax_iou]:
        ax.set_ylim(0, y_max)

    out_path = os.path.join(BASE, "pixel_metrics_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
