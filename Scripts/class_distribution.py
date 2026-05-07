from pathlib import Path
import pandas as pd
import cv2
import numpy as np

csv_path = "/workspace/ECE6364/Outputs/week1_trainfolder_testfolder_seen_unseen_split/fold_00_holdout_socal-fire__guatemala-volcano/train.csv"

df = pd.read_csv(csv_path)

class_names = {
    0: "background",
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}

counts = np.zeros(5, dtype=np.int64)

for p in df["damage_mask_path"]:
    mask = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    counts += np.bincount(mask.reshape(-1), minlength=5)[:5]

total = counts.sum()

print("Pixel-level class distribution:")
for i, c in enumerate(counts):
    print(f"{i} {class_names[i]:13s}: {c:12d}  {100*c/total:.4f}%")

rare = counts[2] + counts[3] + counts[4]
print(f"\nRare damage classes 2/3/4 total: {rare}  {100*rare/total:.4f}%")

minor_major = counts[2] + counts[3]
print(f"Minor+major classes 2/3 total:  {minor_major}  {100*minor_major/total:.4f}%")