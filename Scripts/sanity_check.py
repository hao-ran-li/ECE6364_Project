import pandas as pd
from pathlib import Path

fold = Path("/workspace/ECE6364/Outputs/week1_trainfolder_testfolder_seen_unseen_split/fold_00_holdout_socal-fire__guatemala-volcano")

train = pd.read_csv(fold / "train.csv")
val = pd.read_csv(fold / "val.csv")
seen = pd.read_csv(fold / "seen_test.csv")
unseen = pd.read_csv(fold / "unseen_test.csv")

for a_name, a in [("train", train), ("val", val)]:
    for b_name, b in [("seen", seen), ("unseen", unseen)]:
        overlap = set(a["image_path"]) & set(b["image_path"])
        print(a_name, b_name, len(overlap))