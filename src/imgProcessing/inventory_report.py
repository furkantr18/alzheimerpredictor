"""Phase 1 report from output/inventory/images.csv (run dataset_inventory.py first).

Prints and saves (output/inventory/inventory_summary.json):
counts per dataset/split/class, sizes/modes, name patterns, exact duplicates inside and
across datasets and splits, perceptual-hash overlap, and file-name overlap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main() -> None:
    inv = C.OUTPUT_DIR / "inventory"
    df = pd.read_csv(inv / "images.csv")
    df = df[df["readable"]].copy()
    df["ds_split"] = df["dataset"] + "/" + df["split"]
    out: dict = {}

    out["counts"] = df.groupby(["ds_split", "class_name"]).size().unstack(fill_value=0).to_dict(orient="index")
    out["sizes"] = df.groupby("ds_split").apply(lambda g: (g["width"].astype(str) + "x" + g["height"].astype(str)).value_counts().head(5).to_dict()).to_dict()
    out["modes"] = df.groupby("ds_split")["mode"].value_counts().unstack(fill_value=0).to_dict(orient="index")
    out["name_patterns"] = df.groupby(["ds_split", "name_pattern"]).size().unstack(fill_value=0).to_dict(orient="index")
    out["mean_intensity_by_class"] = df.groupby(["ds_split", "class_name"])["mean_intensity"].mean().round(2).unstack().to_dict(orient="index")
    out["frac_black_by_class"] = df.groupby(["ds_split", "class_name"])["frac_black"].mean().round(3).unstack().to_dict(orient="index")

    # exact duplicates (decoded pixels) inside each dataset/split, and with conflicting labels
    dup = {}
    for key, g in df.groupby("ds_split"):
        vc = g["pixel_md5"].value_counts()
        multi = vc[vc > 1]
        labels = g[g["pixel_md5"].isin(multi.index)].groupby("pixel_md5")["class_name"].nunique()
        dup[key] = {"images": int(len(g)), "unique_pixels": int(g["pixel_md5"].nunique()),
                    "dup_groups": int(len(multi)), "images_in_dup_groups": int(multi.sum()),
                    "dup_groups_with_conflicting_labels": int((labels > 1).sum())}
    out["exact_duplicates_within"] = dup

    # cross-split / cross-dataset exact overlap
    sets = {k: set(g["pixel_md5"]) for k, g in df.groupby("ds_split")}
    keys = sorted(sets)
    out["exact_overlap_pixels"] = {f"{a} & {b}": len(sets[a] & sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}
    names = {k: set(g["stem"]) for k, g in df.groupby("ds_split")}
    out["filename_overlap"] = {f"{a} & {b}": len(names[a] & names[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}

    # same file name in both datasets: are the pixels identical?
    a = df[df["dataset"] == "archive"].set_index(["class_name", "stem"])["pixel_md5"]
    b = df[df["dataset"] == "archive (1)"].set_index(["class_name", "stem"])["pixel_md5"]
    common = a.index.intersection(b.index)
    out["same_class_and_name_identical_pixels"] = {"pairs": int(len(common)),
                                                   "identical": int((a.loc[common].values == b.loc[common].values).sum())}

    # archive (combined) images that are NOT in archive (1) at all (exact pixels)
    in_uran = set(df[df["dataset"] == "archive (1)"]["pixel_md5"])
    comb = df[df["dataset"] == "archive"]
    extra = comb[~comb["pixel_md5"].isin(in_uran)]
    out["archive_only_images"] = {"total": int(len(extra)),
                                  "by_class": extra["class_name"].value_counts().to_dict(),
                                  "by_pattern": extra["name_pattern"].value_counts().to_dict()}

    # perceptual near-duplicates: archive (1) train vs val with identical pHash (or mirrored pHash)
    tr = df[df["ds_split"] == "archive (1)/train"]
    va = df[df["ds_split"] == "archive (1)/val"]
    va_ph = set(va["phash"]) | set(va["phash_flip"])
    out["uraninjo_train_with_exact_phash_in_val"] = int(tr["phash"].isin(va_ph).sum())

    (inv / "inventory_summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
