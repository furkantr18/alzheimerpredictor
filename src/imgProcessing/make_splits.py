"""Phase 3: fixed train/val/test splits (saved as CSV of file paths, seed recorded).

Three split files, all from output/splits/groups.csv:

* split_subject.csv  (HONEST, used for all reported results)
  Units = pseudo-subject groups, per class 70/15/15 (seeded). ModerateDemented has only
  2 subjects: 1 -> train, 1 -> test, none in val (documented limitation).
  Val/test contain ONLY original (non-augmented) images. Train = originals of train
  subjects + augmented images confidently matched to a train subject. Augmented images of
  val/test subjects and uncertain matches are excluded.
* split_image.csv  (augmentation-aware only, subjects ignored)
  Units = original image (augmented images follow their nearest original). Same rules.
  Shows how much slice-level (same patient) leakage remains when only copies are grouped.
* split_naive.csv  (what Yasemin's code effectively did)
  datasets/archive/combined_images (44,000), stratified random 70/15/15 per image.

Usage: python src/imgProcessing/make_splits.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402

FRACTIONS = (0.70, 0.15, 0.15)


def assign_units(units: pd.DataFrame, unit_col: str, seed: int) -> dict[str, str]:
    """units: one row per unit with class_name. Returns unit -> split, stratified by class."""
    rng = np.random.default_rng(seed)
    out = {}
    for cls, g in units.groupby("class_name"):
        u = g[unit_col].to_numpy().copy()
        rng.shuffle(u)
        n = len(u)
        if n == 2:  # ModerateDemented: 2 subjects
            out.update({u[0]: "train", u[1]: "test"})
            continue
        n_test = max(1, int(round(FRACTIONS[2] * n)))
        n_val = max(1, int(round(FRACTIONS[1] * n)))
        for i, unit in enumerate(u):
            out[unit] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return out


def grouped_split(groups: pd.DataFrame, unit_col: str, seed: int) -> pd.DataFrame:
    orig = groups[groups["kind"] == "original"]
    units = orig.drop_duplicates(unit_col)[[unit_col, "class_name"]]
    unit_split = assign_units(units, unit_col, seed)
    df = groups.copy()
    df["split"] = df[unit_col].map(unit_split)
    aug = df["kind"] != "original"
    df.loc[aug & ~df["confident"], "split"] = "excluded_uncertain"
    df.loc[aug & df["confident"] & df["split"].isin(["val", "test"]), "split"] = "excluded_heldout_copy"
    df["split"] = df["split"].fillna("excluded_uncertain")
    return df


def naive_split(seed: int) -> pd.DataFrame:
    inv = pd.read_csv(C.OUTPUT_DIR / "inventory" / "images.csv")
    comb = inv[(inv["dataset"] == "archive") & inv["readable"]][["pixel_md5", "path", "class_name", "stem", "name_pattern"]].reset_index(drop=True)
    from build_cache import open_cache
    _, row_of = open_cache(224)
    comb["cache_row"] = comb["pixel_md5"].map(row_of)
    tr, rest = train_test_split(comb.index, test_size=0.30, random_state=seed, stratify=comb["class_name"])
    va, te = train_test_split(rest, test_size=0.50, random_state=seed, stratify=comb.loc[rest, "class_name"])
    comb["split"] = "train"
    comb.loc[va, "split"] = "val"
    comb.loc[te, "split"] = "test"
    return comb


def summary(df: pd.DataFrame, unit_col: str | None) -> dict:
    s = {"images": df.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_dict(orient="index")}
    if unit_col:
        s["units"] = df[df["kind"] == "original"].groupby(["split", "class_name"])[unit_col].nunique().unstack(fill_value=0).to_dict(orient="index")
        tr = set(df.loc[df["split"] == "train", unit_col])
        held = set(df.loc[df["split"].isin(["val", "test"]), unit_col])
        s["units_shared_between_train_and_heldout"] = len(tr & held)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    C.ensure_dirs()
    groups = pd.read_csv(C.SPLITS_DIR / "groups.csv")

    subj = grouped_split(groups, "subject_group", args.seed)
    img = grouped_split(groups, "image_group", args.seed)
    naive = naive_split(args.seed)
    for name, df in [("split_subject", subj), ("split_image", img), ("split_naive", naive)]:
        df.assign(seed=args.seed).to_csv(C.SPLITS_DIR / f"{name}.csv", index=False)

    # exact-pixel overlap check between train and held-out sets for every split file
    def overlap(df):
        tr = set(df.loc[df["split"] == "train", "pixel_md5"])
        return {k: len(tr & set(df.loc[df["split"] == k, "pixel_md5"])) for k in ("val", "test")}

    rep = {"seed": args.seed,
           "split_subject": {**summary(subj, "subject_group"), "exact_pixel_overlap_train_vs": overlap(subj)},
           "split_image": {**summary(img, "image_group"), "exact_pixel_overlap_train_vs": overlap(img)},
           "split_naive": {**summary(naive.assign(kind="original"), None), "exact_pixel_overlap_train_vs": overlap(naive)}}
    (C.REPORTS_DIR / "splits_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
