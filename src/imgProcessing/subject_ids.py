"""Reconstruct pseudo-subject (scan) IDs for the 6,400 ORIGINAL images.

The dataset ships no patient IDs. Phase 3 found that the file names encode them:

* `<class>Dem<i>.jpg` (e.g. mildDem717): files are ordered slice-major, so
  slice = i // P and subject = i % P, with P = subjects in that class
  (MildDemented 28, ModerateDemented 2, NonDemented 100, VeryMildDemented 70).
  Evidence: image i and i+P are near-identical neighbouring slices (mean correlation 0.99
  versus 0.89 for i and i+1), and 200 subjects x 32 slices = 6,400.
* `<a> (<b>).jpg` / `<a>.jpg`: slice a (26..32), subject b (1-based; no "(b)" means b = 1).

`verify()` re-checks this on the pixels: every subject's slices must form one smooth
sequence, and the `a (b)` images must continue the matching `classDemN` subject.
Run:  python src/imgProcessing/subject_ids.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402

SUBJECTS_PER_CLASS = {"MildDemented": 28, "ModerateDemented": 2, "NonDemented": 100, "VeryMildDemented": 70}
_CLASSDEM = re.compile(r"^(mild|moderate|non|verymild)Dem(\d+)$")
_SLICE_SUBJ = re.compile(r"^(\d+)(?: \((\d+)\))?$")


def parse(class_name: str, stem: str) -> tuple[int, int] | None:
    """Return (subject_index 0-based, slice_index) or None if the name has no known scheme."""
    p = SUBJECTS_PER_CLASS[class_name]
    m = _CLASSDEM.match(stem)
    if m:
        i = int(m.group(2))
        return i % p, i // p
    m = _SLICE_SUBJ.match(stem)
    if m:
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else 1
        return b - 1, a
    return None


def subject_id(class_name: str, stem: str) -> str | None:
    r = parse(class_name, stem)
    return None if r is None else f"{class_name}::S{r[0]:03d}"


def verify(originals: pd.DataFrame) -> dict:
    """originals: rows with path, class_name, stem. Returns check statistics."""
    from PIL import Image

    def vec(p):
        a = np.asarray(Image.open(p).convert("L").resize((64, 64)), dtype=np.float32).ravel()
        a -= a.mean()
        return a / (np.linalg.norm(a) + 1e-6)

    df = originals.copy()
    parsed = [parse(c, s) for c, s in zip(df["class_name"], df["stem"])]
    df["subj"] = [None if r is None else r[0] for r in parsed]
    df["slice"] = [None if r is None else r[1] for r in parsed]
    out = {"unparsed": int(df["subj"].isna().sum())}
    df = df.dropna(subset=["subj"])
    df["vec"] = [vec(p) for p in df["path"]]
    sizes = df.groupby(["class_name", "subj"]).size()
    out["subjects"] = int(len(sizes))
    out["slices_per_subject"] = sizes.value_counts().to_dict()
    out["duplicate_subject_slice"] = int(df.duplicated(["class_name", "subj", "slice"]).sum())

    # neighbouring slices of the same subject vs. same slice of a different subject
    same, other = [], []
    for cls, g in df.groupby("class_name"):
        idx = {(r.subj, r.slice): r.vec for r in g.itertuples()}
        for (s, k), v in idx.items():
            if (s, k + 1) in idx:
                same.append(float(v @ idx[(s, k + 1)]))
            alt = (s + 1) % SUBJECTS_PER_CLASS[cls]
            if alt != s and (alt, k + 1) in idx:
                other.append(float(v @ idx[(alt, k + 1)]))
    same, other = np.array(same), np.array(other)
    out["corr_next_slice_same_subject"] = {"median": float(np.median(same)), "p01": float(np.percentile(same, 1)), "n": int(len(same))}
    out["corr_next_slice_other_subject"] = {"median": float(np.median(other)), "p99": float(np.percentile(other, 99)), "n": int(len(other))}
    out["same_subject_pairs_below_other_p99"] = float((same < np.percentile(other, 99)).mean())
    return out


def main() -> None:
    import json

    inv = pd.read_csv(C.OUTPUT_DIR / "inventory" / "images.csv")
    orig = inv[(inv["dataset"] == "archive (1)") & (inv["split"] == "val")]
    res = verify(orig)
    print(json.dumps(res, indent=2))
    (C.OUTPUT_DIR / "inventory" / "subject_id_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
