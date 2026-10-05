"""Phase 3: fixed train/val/test splits (saved as CSV of file paths, seed recorded).

Decision (see IMAGE_INTEGRATION.md "Decisions"): the dataset's own augmented copies are NOT
used in the honest setting. Their source image cannot be recovered reliably (best matcher:
93.5% subject accuracy on the 46 copies whose source is named in the file name), so any of
them could be a copy of a val/test patient. Training uses ORIGINAL images of training
subjects plus our own on-the-fly augmentation instead.

* split_subject.csv  (HONEST, used for all reported results)
  6,400 originals, units = pseudo-subject (patient) groups, per class 70/15/15 (seeded).
  ModerateDemented has only 2 subjects: 1 -> train, 1 -> test, none in val (limitation).
* split_image.csv    (originals only, random per IMAGE, subjects ignored)
  No copies, but neighbouring slices of one patient can sit in train and test.
* split_naive.csv    (what Yasemin's code effectively did)
  datasets/archive/combined_images (44,000 incl. copies), stratified random 70/15/15 per image.

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
from build_cache import open_cache  # noqa: E402

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


def originals() -> pd.DataFrame:
    g = pd.read_csv(C.SPLITS_DIR / "groups.csv")
    o = g[g["kind"] == "original"][["pixel_md5", "path", "class_name", "stem", "name_pattern", "cache_row", "subject_group", "kind"]]
    return o.reset_index(drop=True)


def subject_split(seed: int) -> pd.DataFrame:
    o = originals()
    s = assign_units(o.drop_duplicates("subject_group")[["subject_group", "class_name"]], "subject_group", seed)
    return o.assign(split=o["subject_group"].map(s))


def image_split(seed: int) -> pd.DataFrame:
    o = originals()
    s = assign_units(o[["pixel_md5", "class_name"]], "pixel_md5", seed)
    return o.assign(split=o["pixel_md5"].map(s))


def naive_split(seed: int) -> pd.DataFrame:
    inv = pd.read_csv(C.OUTPUT_DIR / "inventory" / "images.csv")
    comb = inv[(inv["dataset"] == "archive") & inv["readable"]][["pixel_md5", "path", "class_name", "stem", "name_pattern"]].reset_index(drop=True)
    _, row_of = open_cache(224)
    comb["cache_row"] = comb["pixel_md5"].map(row_of)
    comb["kind"] = "naive"
    tr, rest = train_test_split(comb.index, test_size=0.30, random_state=seed, stratify=comb["class_name"])
    va, te = train_test_split(rest, test_size=0.50, random_state=seed, stratify=comb.loc[rest, "class_name"])
    comb["split"] = "train"
    comb.loc[va, "split"] = "val"
    comb.loc[te, "split"] = "test"
    return comb


def report(df: pd.DataFrame, subject_known: bool) -> dict:
    r = {"images": df.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_dict(orient="index")}
    tr = df[df["split"] == "train"]
    for k in ("val", "test"):
        h = df[df["split"] == k]
        r[f"exact_pixel_overlap_train_{k}"] = len(set(tr["pixel_md5"]) & set(h["pixel_md5"]))
        if subject_known:
            r[f"subjects_shared_train_{k}"] = len(set(tr["subject_group"]) & set(h["subject_group"]))
            r[f"subjects_{k}"] = h.groupby("class_name")["subject_group"].nunique().to_dict()
    if subject_known:
        r["subjects_train"] = tr.groupby("class_name")["subject_group"].nunique().to_dict()
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    C.ensure_dirs()
    subj, img, naive = subject_split(args.seed), image_split(args.seed), naive_split(args.seed)
    for name, df in [("split_subject", subj), ("split_image", img), ("split_naive", naive)]:
        df.assign(seed=args.seed).to_csv(C.SPLITS_DIR / f"{name}.csv", index=False)
    rep = {"seed": args.seed, "split_subject": report(subj, True), "split_image": report(img, True),
           "split_naive": report(naive, False)}
    (C.REPORTS_DIR / "splits_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
