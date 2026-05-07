"""
Building-level metrics across all test folders.
Layout:
  top row:    Macro F1 (summary bars per model)
  bottom row: Per-class F1 (spanning full width)
Seen = solid bar, Unseen = same color + hatch.
Uses building_F1 from all_folds_seen_unseen_damage_class_summary.csv (class_id 1-4).
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
DMG_CLASS_IDS    = [1, 2, 3, 4]
DMG_CLASS_LABELS = ["No-damage", "Minor-damage", "Major-damage", "Destroyed"]
MODEL_COLORS     = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def load_df(folder_path):
    df = pd.read_csv(os.path.join(folder_path, CSV_NAME))
    # Filter to fold 04 only
    df = df[df["fold"].str.contains("fold_04")]
    return df


def macro_f1(df, split):
    """Mean and std of per-fold macro building F1 (damage classes only)."""
    sub = df[(df["split"] == split) & (df["class_id"] != 0)]
    per_fold = sub.groupby("fold")["building_F1"].mean()
    return per_fold.mean(), per_fold.std()


def per_class_f1(df, split):
    """Mean and std across folds for each damage class."""
    sub = df[(df["split"] == split) & (df["class_id"] != 0)]
    return {cid: (sub[sub["class_id"] == cid].groupby("fold")["building_F1"].mean().mean(),
                  sub[sub["class_id"] == cid].groupby("fold")["building_F1"].mean().std())
            for cid in DMG_CLASS_IDS}


def plot_summary_bars(ax, records, labels, title):
    x = np.arange(len(labels))
    w = 0.35
    for i, (label, color) in enumerate(zip(labels, MODEL_COLORS)):
        sm, ss = macro_f1(records[label], "seen_test")
        um, us = macro_f1(records[label], "unseen_test")
        bs = ax.bar(x[i] - w/2, sm, w, color=color, alpha=0.9,
                    yerr=ss, capsize=4, error_kw={"linewidth": 1.1})
        bu = ax.bar(x[i] + w/2, um, w, color=color, alpha=0.45, hatch="///",
                    yerr=us, capsize=4, error_kw={"linewidth": 1.1})
        for bar, val in [(bs[0], sm), (bu[0], um)]:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.004,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    handles = (
        [mpatches.Patch(facecolor=c, alpha=0.9,           label=l.replace("\n", " ") + " – Seen")
         for c, l in zip(MODEL_COLORS, labels)] +
        [mpatches.Patch(facecolor=c, alpha=0.45, hatch="///", label=l.replace("\n", " ") + " – Unseen")
         for c, l in zip(MODEL_COLORS, labels)]
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Macro F1", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=handles, fontsize=6.5, ncol=2, loc="upper right")


def plot_perclass_bars(ax, records, labels, title):
    bar_w = 0.08
    model_gap = 0.05
    total_w = len(labels) * 2 * bar_w + (len(labels) - 1) * model_gap
    pos = -total_w / 2
    offsets = []
    for _ in labels:
        offsets.append((pos + bar_w/2, pos + bar_w + bar_w/2))
        pos += 2 * bar_w + model_gap

    x = np.arange(len(DMG_CLASS_IDS)) * 1.3

    for m_idx, (label, color) in enumerate(zip(labels, MODEL_COLORS)):
        pc_seen   = per_class_f1(records[label], "seen_test")
        pc_unseen = per_class_f1(records[label], "unseen_test")
        so, uo = offsets[m_idx]
        lbl = label.replace("\n", " ")
        ax.bar(x + so, [pc_seen[c][0]   for c in DMG_CLASS_IDS], bar_w,
               color=color, alpha=0.9,
               yerr=[pc_seen[c][1]   for c in DMG_CLASS_IDS], capsize=2,
               error_kw={"linewidth": 0.8}, label=f"{lbl} – Seen")
        ax.bar(x + uo, [pc_unseen[c][0] for c in DMG_CLASS_IDS], bar_w,
               color=color, alpha=0.45, hatch="///",
               yerr=[pc_unseen[c][1] for c in DMG_CLASS_IDS], capsize=2,
               error_kw={"linewidth": 0.8}, label=f"{lbl} – Unseen")

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("F1", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(DMG_CLASS_LABELS, fontsize=9)
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

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Building-Level Metrics (No BG): Seen vs Unseen", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45)

    ax_macro = fig.add_subplot(gs[0])
    ax_pc    = fig.add_subplot(gs[1])

    plot_summary_bars(ax_macro, records, labels,
                      "Macro F1 (damage classes: no-damage / minor / major / destroyed)")
    plot_perclass_bars(ax_pc,   records, labels,
                       "Per-class Building F1")

    y_max = min(1.0, max(ax.get_ylim()[1] for ax in [ax_macro, ax_pc]) * 1.05)
    for ax in [ax_macro, ax_pc]:
        ax.set_ylim(0, y_max)

    out_path = os.path.join(BASE, "building_metrics_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
