"""Stage A: frozen embeddings -> slice head -> patient aggregation, nested patient-grouped CV.

Per outer fold: inner 3-fold grouped CV picks the head hyper-parameter and the aggregation
(patient macro-F1 of inner out-of-fold predictions); then the head is refit on the whole outer
training part and the held-out patients are predicted. Scaler/PCA are fit on training data only.

Usage: python src/imgProcessing/stage_a_heads.py --emb resnet50 --head logreg --task T3
       python src/imgProcessing/stage_a_heads.py --all --task T3     (every backbone x head)
Output: output/cv/oof/A_<emb>_<head>__<task>.npz, output/cv/results/A_<emb>_<head>__<task>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_core as cv  # noqa: E402

warnings.filterwarnings("ignore")
RES_DIR = cv.CV_DIR / "results"
HEADS = {
    "logreg": [{"C": c} for c in (1e-3, 1e-2, 1e-1, 1.0)],
    "linsvm": [{"C": c} for c in (1e-4, 1e-3, 1e-2)],
    "ridge": [{"alpha": a} for a in (10.0, 100.0, 1000.0)],
    "lgbm": [{"num_leaves": n} for n in (7, 15)],
    "mlp": [{"alpha": a} for a in (1e-1, 1.0)],
}
AGGS = ["mean", "median", "trimmed", "top8", "logmean", "learned"]
# Stage D (T3-ord): ordinal Frank-Hall logistic regression; only run explicitly (--head ordlogreg)
EXTRA_HEADS = {"ordlogreg": [{"C": c} for c in (1e-3, 1e-2, 1e-1, 1.0)]}
HEADS_ALL = {**HEADS, **EXTRA_HEADS}


class FrankHallLR:
    """Ordinal classifier (Frank & Hall 2001): one balanced LogReg per threshold P(y > k)."""

    def __init__(self, C=1.0, seed=0):
        self.C, self.seed = C, seed

    def fit(self, X, y):
        self.classes_ = np.arange(int(y.max()) + 1)
        self.models_ = [LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced", random_state=self.seed).fit(X, (y > k).astype(int))
                        for k in range(len(self.classes_) - 1)]
        return self

    def predict_proba(self, X):
        g = np.stack([m.predict_proba(X)[:, 1] for m in self.models_], 1)
        g = np.minimum.accumulate(g, axis=1)  # enforce P(y>k) non-increasing
        K = len(self.classes_)
        p = np.zeros((len(X), K))
        p[:, 0] = 1 - g[:, 0]
        for k in range(1, K - 1):
            p[:, k] = g[:, k - 1] - g[:, k]
        p[:, K - 1] = g[:, K - 2]
        p = np.clip(p, 1e-6, None)
        return p / p.sum(1, keepdims=True)


def make_head(name: str, p: dict, seed: int):
    if name == "logreg":
        return LogisticRegression(C=p["C"], max_iter=3000, class_weight="balanced", random_state=seed)
    if name == "linsvm":
        return LinearSVC(C=p["C"], class_weight="balanced", max_iter=5000, random_state=seed)
    if name == "ridge":
        return RidgeClassifier(alpha=p["alpha"], class_weight="balanced")
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=p["num_leaves"], feature_fraction=0.5,
                              class_weight="balanced", n_jobs=1, random_state=seed, verbose=-1)
    if name == "ordlogreg":
        return FrankHallLR(C=p["C"], seed=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128,), alpha=p["alpha"], early_stopping=True, max_iter=300, random_state=seed)
    raise ValueError(name)


def proba(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
    else:
        d = model.decision_function(X)
        if d.ndim == 1:
            d = np.stack([-d, d], 1)
        d = d - d.max(1, keepdims=True)
        p = np.exp(d) / np.exp(d).sum(1, keepdims=True)
    full = np.zeros((len(X), n_classes))
    full[:, model.classes_] = p
    return full


def fit_transform(Xtr: np.ndarray, Xte_list: list[np.ndarray], seed: int, extra_tr=None, extra_te=None):
    sc = StandardScaler().fit(Xtr)
    k = min(256, Xtr.shape[1], Xtr.shape[0] - 1)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(sc.transform(Xtr))
    Ztr = pca.transform(sc.transform(Xtr))
    Zte = [pca.transform(sc.transform(x)) for x in Xte_list]
    if extra_tr is not None:  # flagged ablation features appended after PCA
        Ztr = np.hstack([Ztr, extra_tr])
        Zte = [np.hstack([z, e]) for z, e in zip(Zte, extra_te)]
    return Ztr, Zte


def learned_agg_fit(prob, groups, y_slice):
    ids, Fp = cv.patient_stats_features(prob, groups)
    yp = np.array([y_slice[np.where(groups == g)[0][0]] for g in ids])
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Fp, yp)


def learned_agg_apply(model, prob, groups, n_classes):
    ids, Fp = cv.patient_stats_features(prob, groups)
    P = np.zeros((len(ids), n_classes))
    P[:, model.classes_] = model.predict_proba(Fp)
    return ids, P


def patient_y(groups, y_slice, ids):
    first = {}
    for g, yy in zip(groups, y_slice):
        first.setdefault(g, yy)
    return np.array([first[g] for g in ids])


def run_fold(df, X, y, groups, posf, r, k, tr, te, heads, n_classes, seed):
    """One outer fold for several heads; Scaler/PCA transforms are fit once per (inner) training part and shared."""
    t0 = time.time()
    inner = list(cv.inner_folds(df, tr, n=3, seed=seed))
    pe = (lambda ix: None) if posf is None else (lambda ix: posf[ix])
    inner_Z = [fit_transform(X[itr], [X[iva]], seed, pe(itr), None if posf is None else [pe(iva)]) for itr, iva in inner]
    Ztr_full, (Zte_full,) = fit_transform(X[tr], [X[te]], seed, pe(tr), None if posf is None else [pe(te)])
    fold_of = np.full(len(y), -1)
    for fi, (_, iva) in enumerate(inner):
        fold_of[iva] = fi
    idx = np.where(fold_of >= 0)[0]
    out = {}
    for head in heads:
        th = time.time()
        grid = HEADS_ALL[head]
        inner_prob = {gi: np.zeros((len(y), n_classes)) for gi in range(len(grid))}
        for (itr, iva), (Ztr, (Zva,)) in zip(inner, inner_Z):
            for gi, p in enumerate(grid):
                m = make_head(head, p, seed).fit(Ztr, y[itr])
                inner_prob[gi][iva] = proba(m, Zva, n_classes)
        # choose grid value by patient macro-F1 (mean aggregation)
        scores = []
        for gi in range(len(grid)):
            ids, P = cv.aggregate(inner_prob[gi][idx], groups[idx], "mean")
            scores.append(cv.patient_metrics(patient_y(groups[idx], y[idx], ids), P, n_classes)["macro_f1"])
        gbest = int(np.argmax(scores))
        pb = inner_prob[gbest]
        # choose aggregation (learned aggregator evaluated leave-one-inner-fold-out)
        agg_scores = {}
        for a in AGGS:
            if a == "learned":
                Ps, ys = [], []
                for fi in range(len(inner)):
                    trm, tem = idx[fold_of[idx] != fi], idx[fold_of[idx] == fi]
                    lm = learned_agg_fit(pb[trm], groups[trm], y[trm])
                    ids, P = learned_agg_apply(lm, pb[tem], groups[tem], n_classes)
                    Ps.append(P); ys.append(patient_y(groups[tem], y[tem], ids))
                agg_scores[a] = cv.patient_metrics(np.concatenate(ys), np.vstack(Ps), n_classes)["macro_f1"]
            else:
                ids, P = cv.aggregate(pb[idx], groups[idx], a)
                agg_scores[a] = cv.patient_metrics(patient_y(groups[idx], y[idx], ids), P, n_classes)["macro_f1"]
        abest = max(agg_scores, key=agg_scores.get)
        # refit on the full outer-training part
        m = make_head(head, grid[gbest], seed).fit(Ztr_full, y[tr])
        ps = proba(m, Zte_full, n_classes)
        ps_tr = proba(m, Ztr_full, n_classes)
        if abest == "learned":
            lm = learned_agg_fit(pb[idx], groups[idx], y[idx])
            ids, P = learned_agg_apply(lm, ps, groups[te], n_classes)
        else:
            ids, P = cv.aggregate(ps, groups[te], abest)
        ids_tr, Ptr = cv.aggregate(ps_tr, groups[tr], "mean")
        out[head] = {"r": r, "k": k, "te": te, "slice_prob": ps, "ids": ids, "P": P, "y": patient_y(groups[te], y[te], ids),
                     "grid": grid[gbest], "agg": abest, "inner_score": float(max(agg_scores.values())), "inner_grid_scores": scores,
                     "inner_agg_scores": agg_scores,
                     "train_patient_macro_f1": cv.patient_metrics(patient_y(groups[tr], y[tr], ids_tr), Ptr, n_classes)["macro_f1"],
                     "seconds": time.time() - th + (th - t0) / len(heads)}
    return out


def run(emb: str, heads, task: str, positions: bool = False, n_jobs: int = 8, fold_name: str = "folds", tag: str = "") -> dict:
    heads = [heads] if isinstance(heads, str) else list(heads)
    df = cv.slices()
    X = np.load(cv.CV_DIR / "emb" / f"{emb}.npy")
    y = cv.task_labels(df["stage4"].to_numpy(), task)
    n_classes = len(cv.TASKS[task]["names"])
    groups = df["group"].to_numpy()
    posf = None
    if positions:  # flagged ablation: coarse slice position (8 bins) one-hot; raw positions leak (see protocol amendment 1)
        b = np.clip(df["slice_pos"].to_numpy() // 4, 0, 7)
        posf = np.eye(8)[b]
    f = cv.folds(fold_name)
    splits = list(cv.outer_splits(df, f))
    res_all = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_fold)(df[["group", "stage4"]], X, y, groups, posf, r, k, tr, te, heads, n_classes, seed=1000 * r + k) for r, k, tr, te in splits)
    return {h: _finish(df, y, n_classes, [x[h] for x in res_all], f"A_{emb}_{h}" + ("_pos" if positions else "") + tag, task, positions, tag)
            for h in heads}


def _finish(df, y, n_classes, res, name, task, positions, tag):
    oof, slice_oof = {}, {}
    for r in sorted({x["r"] for x in res}):
        parts = [x for x in res if x["r"] == r]
        oof[r] = (np.concatenate([x["ids"] for x in parts]), np.concatenate([x["y"] for x in parts]), np.vstack([x["P"] for x in parts]))
        sp = np.zeros((len(df), n_classes))
        for x in parts:
            sp[x["te"]] = x["slice_prob"]
        slice_oof[f"slice_P_{r}"] = sp
    cv.save_oof(name, task, oof, slice_oof)
    summ = cv.summarize(oof, n_classes)
    # slice-level metrics (per repeat, mean)
    sl = [cv.patient_metrics(y, slice_oof[f"slice_P_{r}"], n_classes) for r in sorted(oof)]
    summ["slice_level"] = {k2: float(np.mean([s[k2] for s in sl])) for k2 in ("macro_f1", "balanced_acc", "roc_auc")}
    summ["train_patient_macro_f1_mean"] = float(np.mean([x["train_patient_macro_f1"] for x in res]))
    summ["chosen"] = [{"r": x["r"], "k": x["k"], "grid": x["grid"], "agg": x["agg"], "inner_score": x["inner_score"]} for x in res]
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / f"{name}__{task}.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    for x in res:
        pm = cv.patient_metrics(x["y"], x["P"], n_classes)
        cv.log_run(stage="A" if not tag else "sens", method=name, config={"grid": x["grid"], "agg": x["agg"]}, task=task,
                   repeat=x["r"], outer_fold=x["k"], seed=1000 * x["r"] + x["k"], inner_score=round(x["inner_score"], 4),
                   outer_macro_f1=round(pm["macro_f1"], 4), outer_balanced_acc=round(pm["balanced_acc"], 4),
                   seconds=round(x["seconds"], 1), exploratory=positions, note="positions ablation" if positions else "")
    print(f"[A] {name} {task}: patient macro-F1 {summ['macro_f1']['mean']:.3f} ({summ['macro_f1']['ci95'][0]:.3f}-{summ['macro_f1']['ci95'][1]:.3f}) "
          f"bal-acc {summ['balanced_acc']['mean']:.3f} QWK {summ['qwk']['mean']:.3f} AUC {summ['roc_auc']['mean']:.3f} | "
          f"slice mF1 {summ['slice_level']['macro_f1']:.3f} | train patient mF1 {summ['train_patient_macro_f1_mean']:.3f}", flush=True)
    return summ


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emb")
    ap.add_argument("--head", default="logreg")
    ap.add_argument("--task", default="T3")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.all:
        st = json.loads((cv.CV_DIR / "emb" / "status.json").read_text())
        embs = [k for k, v in st.items() if v.get("ok") and not k.endswith("_25d") and k != "handcrafted"]
        for e in embs:
            todo = [h for h in HEADS if not (RES_DIR / f"A_{e}_{h}__{a.task}.json").exists()]
            if todo:
                run(e, todo, a.task, n_jobs=a.jobs)
            else:
                print(f"[A] skip {e} {a.task} (done)", flush=True)
    else:
        run(a.emb, a.head.split(","), a.task, positions=a.positions, n_jobs=a.jobs)


if __name__ == "__main__":
    main()
