"""Reconstruct pseudo-subject (scan) IDs for the 6,400 ORIGINAL images.

The dataset ships no patient IDs. Phase 3 found that the file names encode them:

* `<class>Dem<i>.jpg` (e.g. mildDem717): files are ordered slice-major, so
  slice = i // P and subject = i % P, with P = subjects in that class
  (MildDemented 28, ModerateDemented 2, NonDemented 100, VeryMildDemented 70).
  Evidence: image i and i+P are near-identical neighbouring slices (mean correlation 0.99
  versus 0.89 for i and i+1), and 200 subjects x 32 slices = 6,400.
* `<a> (<b>).jpg` / `<a>.jpg`: slice a (26..32) of one subject. FIX 2026-10-05: the b numbers do
  NOT follow the classDemN subject order (they partly follow a string sort of file names), so the
  tail sequence of each subject is re-linked to its head sequence by slice continuity
  (`build_tail_remap`, Hungarian assignment per class, stored in output/inventory/tail_remap.json).
  The old identity mapping attached the last ~7 slices of most patients to another patient of the
  same class. The old `verify()` missed it because it only compared positions k and k+1, and the
  head/tail boundary always skips a position (24 -> 26 or 25 -> 27).

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


REMAP_PATH = C.OUTPUT_DIR / "inventory" / "tail_remap.json"
_REMAP: dict | None = None
ROW25 = 25  # the last, partial classDemN row (only some subjects have a slice 25)


def _remap() -> dict:
    """{"tail": {class: {tail_subject: head_subject}}, "row25": {class: {named_subject: head_subject}}} or {}."""
    global _REMAP
    if _REMAP is None:
        import json
        raw = json.loads(REMAP_PATH.read_text()) if REMAP_PATH.exists() else {}
        _REMAP = {part: {c: {int(k): v for k, v in d.items()} for c, d in raw.get(part, {}).items()} for part in ("tail", "row25")} if raw else {}
    return _REMAP


def subject_id(class_name: str, stem: str) -> str | None:
    """Patient ID with the 2026-10-05 fix: the slice-25 row of the classDemN files and the `a (b)` tail files
    do NOT follow the subject order of the full classDemN rows; both are re-linked by slice continuity."""
    r = parse(class_name, stem)
    if r is None:
        return None
    subj, sl = r
    rm = _remap()
    if rm:
        if not _CLASSDEM.match(stem):
            subj = rm["tail"].get(class_name, {}).get(subj, subj)
        elif sl == ROW25:
            subj = rm["row25"].get(class_name, {}).get(subj, subj)
    return f"{class_name}::S{subj:03d}"


def match_tails(head_last: np.ndarray, head_prev: np.ndarray, tail_first: np.ndarray, tail_next: np.ndarray) -> np.ndarray:
    """One-to-one assignment of tail sequences to head sequences by slice continuity.

    Inputs are unit-norm vectors, one row per head (or tail) sequence. Score(h, t) =
    corr(head_last_h, tail_first_t) + 0.5 * corr(head_prev_h, tail_first_t) + 0.5 * corr(head_last_h, tail_next_t).
    Returns tail index per head (Hungarian, maximal total score)."""
    from scipy.optimize import linear_sum_assignment
    S = head_last @ tail_first.T + 0.5 * (head_prev @ tail_first.T) + 0.5 * (head_last @ tail_next.T)
    rows, cols = linear_sum_assignment(-S)
    out = np.empty(len(head_last), dtype=int)
    out[rows] = cols
    return out


def build_tail_remap() -> dict:
    """Re-link the slice-25 row and the tail sequences of every class by slice continuity.

    Each true patient = A (slices 0..24, full classDemN rows, verified by the i / i+P similarity)
    + optionally B (one slice 25 from the partial last row) + C (tail: starts at 27 if B exists,
    at 26 otherwise; 32 slices in total). Step 1 matches every B slice to an A sequence, step 2
    matches the tails to the extended heads within the two length-compatible blocks."""
    import json

    import cv2
    from scipy.optimize import linear_sum_assignment

    from build_cache import open_cache
    inv = pd.read_csv(C.OUTPUT_DIR / "inventory" / "images.csv")
    o = inv[(inv["dataset"] == "archive (1)") & (inv["split"] == "val")].copy()
    arr, row_of = open_cache(224)

    def vec(md5):
        a = cv2.resize(np.asarray(arr[row_of[md5]]), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        a -= a.mean()
        return a / (np.linalg.norm(a) + 1e-6)

    parsed = [parse(c, s) for c, s in zip(o["class_name"], o["stem"])]
    o["subj"], o["slice"] = [p[0] for p in parsed], [p[1] for p in parsed]
    o["tail"] = ~o["stem"].str.match(_CLASSDEM)
    remap, report = {"tail": {}, "row25": {}}, {}
    for cls, d in o.groupby("class_name"):
        h = d[~d["tail"]]
        A = {s: g.sort_values("slice") for s, g in h[h["slice"] < ROW25].groupby("subj")}
        if any(len(g) != ROW25 or g["slice"].iloc[-1] != ROW25 - 1 for g in A.values()):
            raise RuntimeError(f"{cls}: unexpected head layout")
        Bdf = h[h["slice"] == ROW25].sort_values("subj")
        tails = {s: g.sort_values("slice") for s, g in d[d["tail"]].groupby("subj")}
        a_ids = sorted(A)
        a24 = np.stack([vec(A[s]["pixel_md5"].iloc[-1]) for s in a_ids])
        a23 = np.stack([vec(A[s]["pixel_md5"].iloc[-2]) for s in a_ids])
        # step 1: B slices -> A sequences
        b_named = Bdf["subj"].tolist()
        b_vec = np.stack([vec(m) for m in Bdf["pixel_md5"]]) if len(Bdf) else np.zeros((0, a24.shape[1]))
        owner = {}
        if len(b_named):
            S = b_vec @ a24.T + 0.5 * (b_vec @ a23.T)
            r_, c_ = linear_sum_assignment(-S)
            owner = {b_named[i]: a_ids[j] for i, j in zip(r_, c_)}
        remap["row25"][cls] = {int(k): int(v) for k, v in owner.items()}
        b_of = {v: b_vec[b_named.index(k)] for k, v in owner.items()}
        # step 2: tails -> extended heads, in the two length-compatible blocks
        remap["tail"][cls], new_c, ident = {}, [], []
        for with_b, t_start in ((True, ROW25 + 2), (False, ROW25 + 1)):
            hs = [s for s in a_ids if (s in b_of) == with_b]
            ts = sorted(t for t in tails if tails[t]["slice"].iloc[0] == t_start)
            if not hs:
                continue
            if len(hs) != len(ts):
                raise RuntimeError(f"{cls}: {len(hs)} heads {'with' if with_b else 'without'} slice 25 but {len(ts)} tails start at {t_start}")
            hl = np.stack([b_of[s] if with_b else a24[a_ids.index(s)] for s in hs])
            hp = np.stack([a24[a_ids.index(s)] if with_b else a23[a_ids.index(s)] for s in hs])
            tf = np.stack([vec(tails[t]["pixel_md5"].iloc[0]) for t in ts])
            tn = np.stack([vec(tails[t]["pixel_md5"].iloc[1]) for t in ts])
            asg = match_tails(hl, hp, tf, tn)
            remap["tail"][cls].update({int(ts[asg[i]]): int(s) for i, s in enumerate(hs)})
            new_c += [float(hl[i] @ tf[asg[i]]) for i in range(len(hs))]
            ident += [ts[asg[i]] == s for i, s in enumerate(hs)]
        report[cls] = {"subjects": len(a_ids), "row25_slices": len(b_named),
                       "row25_unchanged": float(np.mean([owner[k] == k for k in owner])) if owner else None,
                       "tail_unchanged": float(np.mean(ident)), "head_tail_corr_new": float(np.mean(new_c))}
    REMAP_PATH.write_text(json.dumps(remap, indent=2))
    global _REMAP
    _REMAP = None
    return report


def verify(originals: pd.DataFrame) -> dict:
    """originals: rows with path, class_name, stem. Returns check statistics."""
    from PIL import Image

    def vec(p):
        a = np.asarray(Image.open(p).convert("L").resize((64, 64)), dtype=np.float32).ravel()
        a -= a.mean()
        return a / (np.linalg.norm(a) + 1e-6)

    df = originals.copy()
    parsed = [parse(c, s) for c, s in zip(df["class_name"], df["stem"])]
    df["subj"] = [subject_id(c, s) for c, s in zip(df["class_name"], df["stem"])]  # with the tail remap if built
    df["slice"] = [None if r is None else r[1] for r in parsed]
    df["tail"] = ~df["stem"].str.match(_CLASSDEM)
    out = {"unparsed": int(df["subj"].isna().sum()), "tail_remap_used": bool(_remap())}
    df = df.dropna(subset=["subj"])
    df["vec"] = [vec(p) for p in df["path"]]
    sizes = df.groupby("subj").size()
    out["subjects"] = int(len(sizes))
    out["slices_per_subject"] = {int(k): int(v) for k, v in sizes.value_counts().items()}
    out["duplicate_subject_slice"] = int(df.duplicated(["subj", "slice"]).sum())

    # consecutive slices (in sorted order, so gaps such as 25 -> 27 are included) of the same subject,
    # separately for the head->tail boundary, vs. the same consecutive positions of another subject of that class
    same, boundary, other = [], [], []
    for cls, g in df.groupby("class_name"):
        subs = {s: part.sort_values("slice") for s, part in g.groupby("subj")}
        keys = sorted(subs)
        for i, s in enumerate(keys):
            part = subs[s]
            v, t = np.stack(part["vec"].to_list()), part["tail"].to_numpy()
            c = (v[:-1] * v[1:]).sum(1)
            same.extend(c.tolist())
            boundary.extend(c[(~t[:-1]) & t[1:]].tolist())
            alt = subs[keys[(i + 1) % len(keys)]]
            if alt is not part:
                va = np.stack(alt["vec"].to_list())
                n = min(len(v), len(va)) - 1
                other.extend((v[:n] * va[1:n + 1]).sum(1).tolist())
    same, boundary, other = np.array(same), np.array(boundary), np.array(other)
    p99 = np.percentile(other, 99)
    out["corr_consecutive_same_subject"] = {"median": float(np.median(same)), "p01": float(np.percentile(same, 1)), "n": int(len(same))}
    out["corr_head_tail_boundary"] = {"median": float(np.median(boundary)), "p05": float(np.percentile(boundary, 5)), "n": int(len(boundary)),
                                      "share_below_other_p99": float((boundary < p99).mean())}
    out["corr_other_subject"] = {"median": float(np.median(other)), "p99": float(p99), "n": int(len(other))}
    out["same_subject_pairs_below_other_p99"] = float((same < p99).mean())
    return out


def main() -> None:
    import json

    inv = pd.read_csv(C.OUTPUT_DIR / "inventory" / "images.csv")
    orig = inv[(inv["dataset"] == "archive (1)") & (inv["split"] == "val")]
    import sys as _s
    if "--rebuild-remap" in _s.argv or not REMAP_PATH.exists():
        print("tail remap:", json.dumps(build_tail_remap(), indent=2))
    res = verify(orig)
    print(json.dumps(res, indent=2))
    (C.OUTPUT_DIR / "inventory" / "subject_id_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
