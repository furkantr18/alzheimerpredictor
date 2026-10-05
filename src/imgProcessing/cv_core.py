"""Shared infrastructure for the nested, patient-grouped CV (docs/EXPERIMENT_PROTOCOL.md).

* slices()                -> DataFrame of the 6,400 originals (row order = embedding order), with group
* build_folds()/folds()   -> outer 5-fold x 3 repeats, stratified by 4-class patient label, grouped
* inner_folds()           -> grouped stratified inner folds inside an outer-training part
* task_labels()           -> labels for T3 / T2 / T4
* patient_metrics()       -> macro-F1, balanced acc, QWK, within-one, ROC-AUC, accuracy
* summarize() / paired()  -> repeat mean +/- SD, patient-cluster bootstrap CI, paired difference CI
* log_run()               -> append one row to experiments_registry.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402

CV_DIR = C.OUTPUT_DIR / "cv"
REGISTRY = CV_DIR / "experiments_registry.csv"
REPEAT_SEEDS = (101, 202, 303)
N_OUTER = 5
CLASS4 = C.CLASS_NAMES  # Mild, Moderate, Non, VeryMild (alphabetical, as everywhere else)

# ordered stage of each original class: Non 0, VeryMild 1, Mild 2, Moderate 3
STAGE4 = {"NonDemented": 0, "VeryMildDemented": 1, "MildDemented": 2, "ModerateDemented": 3}
TASKS = {
    "T3": {"names": ["Non", "VeryMild", "Mild+Moderate"], "map": {0: 0, 1: 1, 2: 2, 3: 2}, "ordered": True},
    "T2": {"names": ["Non", "Demented"], "map": {0: 0, 1: 1, 2: 1, 3: 1}, "ordered": True},
    "T4": {"names": ["Non", "VeryMild", "Mild", "Moderate"], "map": {0: 0, 1: 1, 2: 2, 3: 3}, "ordered": True},
}


def slices() -> pd.DataFrame:
    """The 6,400 originals in a fixed order (by pixel_md5) with group, stage and slice position."""
    from subject_ids import parse
    g = pd.read_csv(C.SPLITS_DIR / "groups.csv")
    o = g[g["kind"] == "original"].sort_values("pixel_md5").reset_index(drop=True)
    o["stage4"] = o["class_name"].map(STAGE4).astype(int)
    o["slice_pos"] = [parse(c, s)[1] for c, s in zip(o["class_name"], o["stem"])]
    o["group"] = o["subject_group"]
    gp = Path(CV_DIR / "groups_override.csv")  # sensitivity analysis: stricter groups
    if gp.exists() and getattr(slices, "use_override", False):
        ov = pd.read_csv(gp)
        o["group"] = o["pixel_md5"].map(dict(zip(ov["pixel_md5"], ov["group"])))
    return o[["pixel_md5", "path", "cache_row", "class_name", "stem", "group", "stage4", "slice_pos"]]


def task_labels(stage4: np.ndarray, task: str) -> np.ndarray:
    m = TASKS[task]["map"]
    return np.vectorize(m.get)(np.asarray(stage4)).astype(int)


def build_folds(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    pats = df.groupby("group")["stage4"].first().reset_index()
    rows = []
    for r, seed in enumerate(REPEAT_SEEDS):
        sgkf = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
        for k, (_, te) in enumerate(sgkf.split(pats, pats["stage4"], pats["group"])):
            for gname in pats["group"].iloc[te]:
                rows.append({"group": gname, "repeat": r, "outer_fold": k})
    f = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(path, index=False)
    return f


def folds(name: str = "folds") -> pd.DataFrame:
    p = CV_DIR / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else build_folds(slices(), p)


def outer_splits(df: pd.DataFrame, fold_df: pd.DataFrame):
    """Yield (repeat, fold, train_idx, test_idx) as slice indices into df."""
    for (r, k), part in fold_df.groupby(["repeat", "outer_fold"]):
        te_groups = set(part["group"])
        te = np.where(df["group"].isin(te_groups))[0]
        tr = np.where(~df["group"].isin(te_groups))[0]
        yield int(r), int(k), tr, te


def inner_folds(df: pd.DataFrame, tr_idx: np.ndarray, n: int = 3, seed: int = 0):
    """Grouped, stratified inner folds over the outer-training slices. Yields (itr, iva) as indices into df."""
    sub = df.iloc[tr_idx]
    pats = sub.groupby("group")["stage4"].first().reset_index()
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    for _, va in sgkf.split(pats, pats["stage4"], pats["group"]):
        vg = set(pats["group"].iloc[va])
        m = sub["group"].isin(vg).to_numpy()
        yield tr_idx[~m], tr_idx[m]


def inner_holdout(df: pd.DataFrame, tr_idx: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One grouped stratified ~80/20 inner split (first fold of a 5-fold split)."""
    return next(inner_folds(df, tr_idx, n=5, seed=seed))


# ---------------------------------------------------------------- aggregation
def aggregate(prob: np.ndarray, groups: np.ndarray, how: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Slice probabilities -> patient probabilities. Returns (patient_ids, P)."""
    d = pd.DataFrame(prob)
    d["g"] = groups
    out_ids, out = [], []
    for g, part in d.groupby("g", sort=True):
        p = part.drop(columns="g").to_numpy()
        if how == "mean":
            v = p.mean(0)
        elif how == "median":
            v = np.median(p, 0)
        elif how == "trimmed":
            q = np.sort(p, 0)
            k = int(0.2 * len(q))
            v = q[k:len(q) - k].mean(0) if len(q) - 2 * k > 0 else q.mean(0)
        elif how == "top8":
            conf = p.max(1)
            v = p[np.argsort(-conf)[:8]].mean(0)
        elif how == "logmean":
            v = np.exp(np.log(np.clip(p, 1e-6, 1)).mean(0))
        else:
            raise ValueError(how)
        out_ids.append(g)
        out.append(v / v.sum())
    return np.array(out_ids), np.vstack(out)


def patient_stats_features(prob: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-patient features for the learned aggregator: per-class mean, max, std of slice probs."""
    d = pd.DataFrame(prob)
    d["g"] = groups
    agg = d.groupby("g", sort=True).agg(["mean", "max", "std"])
    return agg.index.to_numpy(), agg.fillna(0).to_numpy()


# ---------------------------------------------------------------- metrics
def patient_metrics(y: np.ndarray, P: np.ndarray, n_classes: int) -> dict:
    y = np.asarray(y)
    pred = P.argmax(1)
    labels = list(range(n_classes))
    present = [c for c in labels if (y == c).any()]
    m = {"macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
         "balanced_acc": float(balanced_accuracy_score(y, pred)),
         "accuracy": float(accuracy_score(y, pred)),
         "qwk": float(cohen_kappa_score(y, pred, weights="quadratic", labels=labels)) if len(set(y)) > 1 else float("nan"),
         "within_one": float((np.abs(pred - y) <= 1).mean())}
    try:
        if n_classes == 2:
            m["roc_auc"] = float(roc_auc_score(y, P[:, 1]))
        else:
            aucs = [roc_auc_score((y == c).astype(int), P[:, c]) for c in present if 0 < (y == c).sum() < len(y)]
            m["roc_auc"] = float(np.mean(aucs))
    except ValueError:
        m["roc_auc"] = float("nan")
    return m


def _auc_binary(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mann-Whitney AUC with tie handling (same value as sklearn roc_auc_score)."""
    s = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(s) + 1, dtype=float)
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks[order] = r
    n1, n0 = len(pos_scores), len(neg_scores)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_metrics(y: np.ndarray, P: np.ndarray, n_classes: int) -> dict:
    """Numpy re-implementation of patient_metrics (used inside the bootstrap; equal to sklearn, see tests)."""
    y = np.asarray(y)
    pred = P.argmax(1)
    cm = np.zeros((n_classes, n_classes))
    np.add.at(cm, (y, pred), 1)
    present = cm.sum(1) > 0
    tp = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(cm.sum(0) > 0, tp / cm.sum(0), 0.0)
        rec = np.where(cm.sum(1) > 0, tp / cm.sum(1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    n = len(y)
    m = {"macro_f1": float(f1[present].mean()), "balanced_acc": float(rec[present].mean()),
         "accuracy": float(tp.sum() / n), "within_one": float((np.abs(pred - y) <= 1).mean())}
    if present.sum() > 1:
        w = (np.subtract.outer(np.arange(n_classes), np.arange(n_classes)) ** 2).astype(float)
        E = np.outer(cm.sum(1), cm.sum(0)) / n
        den = (w * E).sum()
        m["qwk"] = float(1 - (w * cm).sum() / den) if den > 0 else float("nan")
    else:
        m["qwk"] = float("nan")
    aucs = []
    cls = [1] if n_classes == 2 else [c for c in range(n_classes) if present[c]]
    for c in cls:
        pos = y == c
        if 0 < pos.sum() < n:
            aucs.append(_auc_binary(P[pos, c], P[~pos, c]))
    m["roc_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    return m


def ece(y: np.ndarray, P: np.ndarray, bins: int = 10) -> float:
    conf, pred = P.max(1), P.argmax(1)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


METRICS = ("macro_f1", "balanced_acc", "qwk", "within_one", "roc_auc", "accuracy")


def summarize(oof: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], n_classes: int,
              n_boot: int = 2000, seed: int = 0, metrics=METRICS) -> dict:
    """oof: repeat -> (unit_ids, y, P) with the same units in every repeat.
    Returns per-metric mean, sd over repeats and patient-cluster bootstrap CI of the repeat mean."""
    reps = sorted(oof)
    ids0 = oof[reps[0]][0]
    order = {r: pd.Series(np.arange(len(oof[r][0])), index=oof[r][0]).loc[ids0].to_numpy() for r in reps}
    Y = {r: oof[r][1][order[r]] for r in reps}
    PP = {r: oof[r][2][order[r]] for r in reps}
    per_rep = [patient_metrics(Y[r], PP[r], n_classes) for r in reps]
    out = {k: {"mean": float(np.nanmean([p[k] for p in per_rep])), "sd": float(np.nanstd([p[k] for p in per_rep], ddof=1)) if len(reps) > 1 else 0.0}
           for k in metrics}
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in metrics}
    n = len(ids0)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals = [fast_metrics(Y[r][idx], PP[r][idx], n_classes) for r in reps]
        for k in metrics:
            boots[k].append(np.nanmean([v[k] for v in vals]))
    for k in metrics:
        out[k]["ci95"] = [float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))]
    out["n_units"] = int(n)
    return out


def paired(oof_a: dict, oof_b: dict, n_classes: int, metric: str = "macro_f1", n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired patient-cluster bootstrap of (A - B) on the same units and folds."""
    reps = sorted(set(oof_a) & set(oof_b))
    ids0 = oof_a[reps[0]][0]

    def aligned(oof, r):
        s = pd.Series(np.arange(len(oof[r][0])), index=oof[r][0]).loc[ids0].to_numpy()
        return oof[r][1][s], oof[r][2][s]

    A = {r: aligned(oof_a, r) for r in reps}
    B = {r: aligned(oof_b, r) for r in reps}
    obs = np.mean([patient_metrics(*A[r], n_classes)[metric] - patient_metrics(*B[r], n_classes)[metric] for r in reps])
    rng = np.random.default_rng(seed)
    n = len(ids0)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean([fast_metrics(A[r][0][idx], A[r][1][idx], n_classes)[metric]
                              - fast_metrics(B[r][0][idx], B[r][1][idx], n_classes)[metric] for r in reps]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"metric": metric, "diff": float(obs), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
            "distinguishable": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------- registry and OOF storage
def log_run(**row) -> None:
    CV_DIR.mkdir(parents=True, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    for k, v in list(row.items()):
        if isinstance(v, (dict, list)):
            row[k] = json.dumps(v)
    new = not REGISTRY.exists()
    fields = ["time", "stage", "method", "config", "task", "repeat", "outer_fold", "seed", "inner_score",
              "outer_macro_f1", "outer_balanced_acc", "seconds", "exploratory", "note"]
    with open(REGISTRY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def save_oof(name: str, task: str, oof: dict, extra: dict | None = None) -> Path:
    """oof: repeat -> (ids, y, P[, slice-level arrays...]). Stored as npz under output/cv/oof/."""
    d = CV_DIR / "oof"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}__{task}.npz"
    arrays = {}
    for r, tup in oof.items():
        arrays[f"ids_{r}"], arrays[f"y_{r}"], arrays[f"P_{r}"] = tup[0].astype(str), tup[1], tup[2]
    for k, v in (extra or {}).items():
        arrays[k] = v
    np.savez_compressed(p, **arrays)
    return p


def load_oof(name: str, task: str) -> dict:
    z = np.load(CV_DIR / "oof" / f"{name}__{task}.npz", allow_pickle=False)
    reps = sorted({int(k.split("_")[1]) for k in z.files if k.startswith("ids_")})
    return {r: (z[f"ids_{r}"], z[f"y_{r}"], z[f"P_{r}"]) for r in reps}
