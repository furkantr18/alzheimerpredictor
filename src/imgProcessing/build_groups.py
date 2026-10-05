"""Phase 3: leakage-aware groups.

Every image gets a `subject_group`:
* ORIGINAL images (archive (1)/val, 6,400): pseudo-subject from the file name
  (subject_ids.py, verified on pixels). The 3 ambiguous subject pairs (31/33 slices) are
  merged into one group each, so an uncertain slice can never sit on both sides of a split.
* AUGMENTED images (archive (1)/train, UUID names, no source info): matched to their source
  original with an ImageNet ResNet-50 embedding of the shared-preprocessed image
  (cosine similarity, same class only, originals also mirrored). We need the SUBJECT, so the
  decision uses the subject margin = best similarity to the winning subject minus best
  similarity to any other subject. Only matches with margin >= threshold are "confident";
  the rest are never used (excluded from every split).
* archive-only extras: `aug_<k>_<uuid>` inherit the subject of that UUID image,
  `aug_<k>_<original name>` take the subject of the named original (ground truth).

The threshold is chosen on images whose source is known independently:
(1) augmented images whose pHash equals an original's pHash, (2) `aug_<k>_<original name>`.

Usage: python src/imgProcessing/build_groups.py [--min-precision 0.995]
Output: output/splits/groups.csv, output/reports/grouping_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402
import subject_ids  # noqa: E402
from build_cache import open_cache  # noqa: E402


def merged_subject_groups(orig: pd.DataFrame) -> pd.Series:
    """subject id per original; per class, the 33-slice and 31-slice subjects become one group."""
    sid = pd.Series([subject_ids.subject_id(c, s) for c, s in zip(orig["class_name"], orig["stem"])], index=orig.index)
    sizes = sid.value_counts()
    remap = {}
    for cls in orig["class_name"].unique():
        odd = [s for s in sizes.index if s.startswith(cls + "::") and sizes[s] != 32]
        if odd:
            merged = "+".join(sorted(odd))
            remap.update({s: merged for s in odd})
    return sid.map(lambda s: remap.get(s, s))


def embed(rows: np.ndarray, cache: np.memmap, batch: int = 128, flip: bool = False) -> np.ndarray:
    import torch
    from torchvision import models

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        for i in range(0, len(rows), batch):
            x = torch.from_numpy(np.asarray(cache[np.sort(rows[i:i + batch])])).to(dev)
            # restore original order inside the batch (memmap fancy indexing wants sorted rows)
            order = np.argsort(np.argsort(rows[i:i + batch]))
            x = x[torch.from_numpy(order).to(dev)]
            x = x.float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
            if flip:
                x = torch.flip(x, dims=[3])
            f = net((x - mean) / std).float()
            out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-precision", type=float, default=0.995)
    args = ap.parse_args()
    C.ensure_dirs()

    inv = pd.read_csv(C.OUTPUT_DIR / "inventory" / "images.csv")
    inv = inv[inv["readable"]].copy()
    cache, row_of = open_cache(224)
    inv["cache_row"] = inv["pixel_md5"].map(row_of)

    orig = inv[(inv["dataset"] == "archive (1)") & (inv["split"] == "val")].copy()
    orig["subject_group"] = merged_subject_groups(orig)
    aug = inv[(inv["dataset"] == "archive (1)") & (inv["split"] == "train")].drop_duplicates("pixel_md5").copy()
    extra = inv[(inv["dataset"] == "archive") & ~inv["pixel_md5"].isin(set(inv.loc[inv["dataset"] == "archive (1)", "pixel_md5"]))].copy()

    print(f"[groups] originals={len(orig)} augmented(unique)={len(aug)} archive-only extras={len(extra)}")
    e_orig = embed(orig["cache_row"].to_numpy(), cache)
    e_orig_f = embed(orig["cache_row"].to_numpy(), cache, flip=True)

    def match(df: pd.DataFrame) -> pd.DataFrame:
        e = embed(df["cache_row"].to_numpy(), cache)
        res = []
        for cls in df["class_name"].unique():
            mi = np.where(df["class_name"].to_numpy() == cls)[0]
            oi = np.where(orig["class_name"].to_numpy() == cls)[0]
            sims = np.maximum(e[mi] @ e_orig[oi].T, e[mi] @ e_orig_f[oi].T)  # (n_aug, n_orig)
            subj = orig["subject_group"].to_numpy()[oi]
            usubj, inv_idx = np.unique(subj, return_inverse=True)
            per_subj = np.full((len(mi), len(usubj)), -1.0, dtype=np.float32)
            for k in range(len(usubj)):
                per_subj[:, k] = sims[:, inv_idx == k].max(1)
            top = np.argsort(-per_subj, axis=1)
            best = per_subj[np.arange(len(mi)), top[:, 0]]
            second = per_subj[np.arange(len(mi)), top[:, 1]] if len(usubj) > 1 else np.full(len(mi), -1.0)
            nearest = oi[sims.argmax(1)]
            for j, m in enumerate(mi):
                res.append((m, usubj[top[j, 0]], float(best[j]), float(best[j] - second[j]), orig["pixel_md5"].iat[nearest[j]]))
        r = pd.DataFrame(res, columns=["pos", "match_subject", "match_sim", "match_margin", "nearest_original"]).set_index("pos").sort_index()
        out = df.reset_index(drop=True).join(r)
        return out

    aug = match(aug)

    # ---- independent ground truth for choosing the threshold ----
    ph = {}
    for r in orig.itertuples():
        ph.setdefault(r.phash, set()).add(r.subject_group)
        ph.setdefault(r.phash_flip, set()).add(r.subject_group)
    gt1 = aug["phash"].map(lambda h: next(iter(ph[h])) if h in ph and len(ph[h]) == 1 else None)
    known = aug[gt1.notna()].copy()
    known["truth"] = gt1[gt1.notna()]

    name_to_subject = dict(zip(orig["class_name"] + "::" + orig["stem"], orig["subject_group"]))
    extra["source"] = extra["stem"].str.replace(r"^aug_\d+_", "", regex=True)
    is_named = ~extra["source"].str.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-")
    named = extra[is_named].copy()
    named["truth"] = (named["class_name"] + "::" + named["source"]).map(name_to_subject)
    named = match(named[named["truth"].notna()])

    gt = pd.concat([known[["match_subject", "match_margin", "truth"]].assign(src="phash"),
                    named[["match_subject", "match_margin", "truth"]].assign(src="named")])
    gt["correct"] = gt["match_subject"] == gt["truth"]
    thr_grid = np.round(np.arange(0.0, 0.2001, 0.0025), 4)
    curve = []
    for t in thr_grid:
        sel = gt[gt["match_margin"] >= t]
        prec = float(sel["correct"].mean()) if len(sel) else 1.0
        curve.append({"threshold": float(t), "precision": prec, "kept_gt": int(len(sel)),
                      "kept_all_aug": int((aug["match_margin"] >= t).sum())})
    ok = [c for c in curve if c["precision"] >= args.min_precision]
    threshold = ok[0]["threshold"] if ok else float(thr_grid[-1])
    print(f"[groups] ground-truth pairs={len(gt)} accuracy@0={gt['correct'].mean():.4f}; threshold={threshold}")

    aug["confident"] = aug["match_margin"] >= threshold
    aug["subject_group"] = np.where(aug["confident"], aug["match_subject"], "UNCERTAIN")

    # archive-only extras: inherit (uuid source) or named source
    uuid_subject = dict(zip(aug["stem"], aug["subject_group"]))
    uuid_match = dict(zip(aug["stem"], aug["match_subject"]))
    extra["match_subject"] = np.where(is_named, (extra["class_name"] + "::" + extra["source"]).map(name_to_subject),
                                      extra["source"].map(uuid_match))
    extra["subject_group"] = np.where(is_named, (extra["class_name"] + "::" + extra["source"]).map(name_to_subject),
                                      extra["source"].map(uuid_subject))
    extra["subject_group"] = extra["subject_group"].fillna("UNCERTAIN")
    extra["confident"] = extra["subject_group"] != "UNCERTAIN"

    cols = ["pixel_md5", "path", "class_name", "stem", "name_pattern", "cache_row", "subject_group", "confident"]
    orig["kind"], orig["confident"] = "original", True
    aug["kind"], extra["kind"] = "augmented", "augmented_extra"
    groups = pd.concat([orig[cols + ["kind"]],
                        aug[cols + ["kind", "match_subject", "match_sim", "match_margin", "nearest_original"]],
                        extra[cols + ["kind", "match_subject"]]], ignore_index=True)
    orig_nearest = dict(zip(orig["pixel_md5"], orig["pixel_md5"]))
    groups["image_group"] = groups["nearest_original"].fillna(groups["pixel_md5"].map(orig_nearest))
    groups.to_csv(C.SPLITS_DIR / "groups.csv", index=False)

    sizes = groups[groups["subject_group"] != "UNCERTAIN"].groupby("subject_group").size()
    rep = {
        "threshold_subject_margin": threshold, "min_precision_target": args.min_precision,
        "ground_truth_pairs": {"phash": int((gt["src"] == "phash").sum()), "named": int((gt["src"] == "named").sum())},
        "ground_truth_accuracy_no_threshold": float(gt["correct"].mean()),
        "ground_truth_accuracy_by_source": gt.groupby("src")["correct"].mean().to_dict(),
        "precision_curve": curve,
        "augmented_confident": int(aug["confident"].sum()), "augmented_uncertain": int((~aug["confident"]).sum()),
        "extras_confident": int(extra["confident"].sum()), "extras_uncertain": int((~extra["confident"]).sum()),
        "subject_groups": int(len(sizes)),
        "subject_groups_per_class": groups[groups["kind"] == "original"].groupby("class_name")["subject_group"].nunique().to_dict(),
        "group_size_images": {"min": int(sizes.min()), "median": float(sizes.median()), "max": int(sizes.max())},
        "uncertain_by_class": aug[~aug["confident"]]["class_name"].value_counts().to_dict(),
    }
    (C.REPORTS_DIR / "grouping_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "precision_curve"}, indent=2))


if __name__ == "__main__":
    main()
