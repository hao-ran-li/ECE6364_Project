"""
Summarizes pixel-level IoU and building-level F1 from the same data paths
used by plot_pixel_metrics.py and plot_building_metrics.py.
Saves output to Outputs/building_metrics_summary.txt — no plots generated.

Pixel metrics:  all folds averaged  (matches plot_pixel_metrics.py)
Building metrics: fold_04 only      (matches plot_building_metrics.py)
"""

import os
import pandas as pd
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..', 'Outputs')

TEST_FOLDERS = {
    'UNet Weighted CE':         'week2_test_unet_weighted_ce',
    'SegFormer Weighted CE':    'week3_test_segformer_weighted_ce',
    'UNet Patch Sampling':      'week4_test_unet_patch_sampling',
    'SegFormer Patch Sampling': 'week4_test_segformer_patch_sampling',
}

CSV_NAME = 'all_folds_seen_unseen_damage_class_summary.csv'

ALL_CLASS_IDS    = [0, 1, 2, 3, 4]
ALL_CLASS_LABELS = {0: 'Background', 1: 'No-damage', 2: 'Minor-damage', 3: 'Major-damage', 4: 'Destroyed'}

DMG_CLASS_IDS    = [1, 2, 3, 4]
DMG_CLASS_LABELS = {1: 'No-damage', 2: 'Minor-damage', 3: 'Major-damage', 4: 'Destroyed'}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pixel_df(folder):
    """All folds — matches plot_pixel_metrics.py behaviour."""
    return pd.read_csv(os.path.join(BASE, folder, CSV_NAME))


def load_building_df(folder):
    """fold_04 only — matches plot_building_metrics.py behaviour."""
    df = pd.read_csv(os.path.join(BASE, folder, CSV_NAME))
    return df[df['fold'].str.contains('fold_04')]


# ---------------------------------------------------------------------------
# Pixel helpers
# ---------------------------------------------------------------------------

def pixel_summary(df, col, split):
    """Mean ± std of per-fold mean across all classes."""
    per_fold = df[df['split'] == split].groupby('fold')[col].mean()
    return per_fold.mean(), per_fold.std()


def pixel_per_class(df, col, split):
    sub = df[df['split'] == split]
    return {cid: (sub[sub['class_id'] == cid].groupby('fold')[col].mean().mean(),
                  sub[sub['class_id'] == cid].groupby('fold')[col].mean().std())
            for cid in ALL_CLASS_IDS}


# ---------------------------------------------------------------------------
# Building helpers
# ---------------------------------------------------------------------------

def building_macro_f1(df, split):
    sub = df[(df['split'] == split) & (df['class_id'] != 0)]
    per_fold = sub.groupby('fold')['building_F1'].mean()
    return per_fold.mean(), per_fold.std()


def building_per_class_f1(df, split):
    sub = df[(df['split'] == split) & (df['class_id'] != 0)]
    return {cid: (sub[sub['class_id'] == cid].groupby('fold')['building_F1'].mean().mean(),
                  sub[sub['class_id'] == cid].groupby('fold')['building_F1'].mean().std())
            for cid in DMG_CLASS_IDS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pixel_records    = {}
    building_records = {}

    for label, folder in TEST_FOLDERS.items():
        csv_path = os.path.join(BASE, folder, CSV_NAME)
        if not os.path.isfile(csv_path):
            print(f'[WARN] not found: {csv_path}')
            continue
        pixel_records[label]    = load_pixel_df(folder)
        building_records[label] = load_building_df(folder)

    lines = []

    # -----------------------------------------------------------------------
    # PIXEL METRICS
    # -----------------------------------------------------------------------
    lines.append('=' * 76)
    lines.append('PIXEL-LEVEL METRICS  (all folds averaged, all classes incl. background)')
    lines.append('=' * 76)

    lines.append('')
    lines.append('mIoU  (mean across all 5 classes)')
    lines.append(f"  {'Model':<26}  {'Seen':>8}  {'Unseen':>8}  {'Delta':>8}")
    lines.append('  ' + '-' * 56)
    for label, df in pixel_records.items():
        sm, _ = pixel_summary(df, 'pixel_IoU', 'seen_test')
        um, _ = pixel_summary(df, 'pixel_IoU', 'unseen_test')
        lines.append(f'  {label:<26}  {sm:.4f}     {um:.4f}     {um-sm:+.4f}')

    for split_name, split_key in [('SEEN TEST', 'seen_test'), ('UNSEEN TEST', 'unseen_test')]:
        lines.append('')
        lines.append(f'PER-CLASS IoU  — {split_name}')
        lines.append(
            f"  {'Model':<26}" +
            ''.join(f'  {ALL_CLASS_LABELS[c]:>13}' for c in ALL_CLASS_IDS)
        )
        lines.append('  ' + '-' * (26 + 15 * len(ALL_CLASS_IDS)))
        for label, df in pixel_records.items():
            pc = pixel_per_class(df, 'pixel_IoU', split_key)
            row = f'  {label:<26}' + ''.join(f'  {pc[c][0]:>13.4f}' for c in ALL_CLASS_IDS)
            lines.append(row)

    for split_name, split_key in [('SEEN TEST', 'seen_test'), ('UNSEEN TEST', 'unseen_test')]:
        lines.append('')
        lines.append(f'PER-CLASS IoU (mean ± std)  — {split_name}')
        lines.append(
            f"  {'Model':<26}" +
            ''.join(f'  {ALL_CLASS_LABELS[c]:>20}' for c in ALL_CLASS_IDS)
        )
        lines.append('  ' + '-' * (26 + 22 * len(ALL_CLASS_IDS)))
        for label, df in pixel_records.items():
            pc = pixel_per_class(df, 'pixel_IoU', split_key)
            row = f'  {label:<26}' + ''.join(f'  {pc[c][0]:>7.4f} ± {pc[c][1]:>6.4f}' for c in ALL_CLASS_IDS)
            lines.append(row)

    # -----------------------------------------------------------------------
    # BUILDING METRICS
    # -----------------------------------------------------------------------
    lines.append('')
    lines.append('=' * 76)
    lines.append('BUILDING-LEVEL METRICS  (fold_04 only, damage classes only, no BG)')
    lines.append('=' * 76)

    lines.append('')
    lines.append('MACRO F1')
    lines.append(f"  {'Model':<26}  {'Seen':>8}  {'Unseen':>8}  {'Delta':>8}")
    lines.append('  ' + '-' * 56)
    for label, df in building_records.items():
        sm, _ = building_macro_f1(df, 'seen_test')
        um, _ = building_macro_f1(df, 'unseen_test')
        lines.append(f'  {label:<26}  {sm:.4f}     {um:.4f}     {um-sm:+.4f}')

    for split_name, split_key in [('SEEN TEST', 'seen_test'), ('UNSEEN TEST', 'unseen_test')]:
        lines.append('')
        lines.append(f'PER-CLASS F1  — {split_name}')
        lines.append(
            f"  {'Model':<26}" +
            ''.join(f'  {DMG_CLASS_LABELS[c]:>13}' for c in DMG_CLASS_IDS)
        )
        lines.append('  ' + '-' * (26 + 15 * len(DMG_CLASS_IDS)))
        for label, df in building_records.items():
            pc = building_per_class_f1(df, split_key)
            row = f'  {label:<26}' + ''.join(f'  {pc[c][0]:>13.4f}' for c in DMG_CLASS_IDS)
            lines.append(row)

    for split_name, split_key in [('SEEN TEST', 'seen_test'), ('UNSEEN TEST', 'unseen_test')]:
        lines.append('')
        lines.append(f'PER-CLASS F1 (mean ± std)  — {split_name}')
        lines.append(
            f"  {'Model':<26}" +
            ''.join(f'  {DMG_CLASS_LABELS[c]:>20}' for c in DMG_CLASS_IDS)
        )
        lines.append('  ' + '-' * (26 + 22 * len(DMG_CLASS_IDS)))
        for label, df in building_records.items():
            pc = building_per_class_f1(df, split_key)
            row = f'  {label:<26}' + ''.join(f'  {pc[c][0]:>7.4f} ± {pc[c][1]:>6.4f}' for c in DMG_CLASS_IDS)
            lines.append(row)

    out = '\n'.join(lines)
    print(out)

    out_path = os.path.join(BASE, 'metrics_summary.txt')
    with open(out_path, 'w') as f:
        f.write(out + '\n')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
