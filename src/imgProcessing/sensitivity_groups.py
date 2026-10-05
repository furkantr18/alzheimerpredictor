"""Sensitivity analysis (protocol section 7): stricter patient groups.

Two patient groups are merged (union-find) when any slice of one has a correlation >= 0.98 with any
slice of the other (64x64, zero-mean, unit-norm: same measure as data_checks.py). Writes
output/cv/groups_override.csv (pixel_md5 -> strict group) and output/cv/folds_strict.csv.
Usage: python src/imgProcessing/sensitivity_groups.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_core as cv  # noqa: E402
from mri_data import pixel_features  # noqa: E402

THRESH = 0.98


def main() -> None:
    df = cv.slices()
    X = pixel_features(df, 64)
    X = X - X.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
    g = df["group"].to_numpy()
    parent = {x: x for x in np.unique(g)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = 0
    for i in range(0, len(X), 512):
        S = X[i:i + 512] @ X.T
        for a, b in zip(*np.nonzero(S >= THRESH)):
            ga, gb = g[i + a], g[b]
            if ga != gb:
                pairs += 1
                ra, rb = find(ga), find(gb)
                if ra != rb:
                    parent[rb] = ra
    strict = np.array([find(x) for x in g])
    cross_class = pd.DataFrame({"s": strict, "c": df["stage4"]}).groupby("s")["c"].nunique()
    out = pd.DataFrame({"pixel_md5": df["pixel_md5"], "group": strict})
    out.to_csv(cv.CV_DIR / "groups_override.csv", index=False)
    cv.slices.use_override = True
    df2 = cv.slices()
    cv.slices.use_override = False
    # stratify by the majority stage of the merged group
    df2 = df2.copy()
    maj = df2.groupby("group")["stage4"].agg(lambda s: s.value_counts().idxmax())
    df2["stage4"] = df2["group"].map(maj)
    cv.build_folds(df2, cv.CV_DIR / "folds_strict.csv")
    rep = {"threshold": THRESH, "cross_group_slice_pairs": int(pairs // 2), "groups_before": int(len(np.unique(g))),
           "groups_after": int(len(np.unique(strict))), "merged_groups_with_mixed_labels": int((cross_class > 1).sum()),
           "largest_group_slices": int(pd.Series(strict).value_counts().max())}
    (cv.CV_DIR / "results").mkdir(parents=True, exist_ok=True)
    (cv.CV_DIR / "results" / "sensitivity_groups.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
