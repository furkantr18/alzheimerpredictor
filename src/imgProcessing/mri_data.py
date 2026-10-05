"""Shared data access and metrics for MRI training/evaluation.

* load_split(name)    -> DataFrame of a saved split CSV (train/val/test rows only)
* pixel_features(df)  -> (N, size*size) float32 features from the preprocessed cache
* compute_metrics()   -> per-class P/R/F1, macro-F1, balanced accuracy, confusion matrix,
                         one-vs-rest ROC-AUC, plus a subject-cluster bootstrap 95% CI
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402
from build_cache import open_cache  # noqa: E402

LABEL_OF = {c: i for i, c in enumerate(C.CLASS_NAMES)}


def load_split(name: str = "split_subject") -> pd.DataFrame:
    df = pd.read_csv(C.SPLITS_DIR / f"{name}.csv")
    df = df[df["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    df["y"] = df["class_name"].map(LABEL_OF).astype(int)
    if "subject_group" not in df.columns:      # naive split: every image is its own unit
        df["subject_group"] = df["pixel_md5"]
    return df


def pixel_features(df: pd.DataFrame, size: int = 64) -> np.ndarray:
    """Downsample the shared 224 preprocessing to size x size and flatten to [0, 1] floats."""
    import cv2

    feat_path = C.CACHE_DIR / f"feat_{size}.npy"
    arr, row_of = open_cache(224)
    if feat_path.exists():
        allf = np.load(feat_path, mmap_mode="r")
    else:
        allf = np.empty((arr.shape[0], size * size), dtype=np.float16)
        for i in range(arr.shape[0]):
            allf[i] = cv2.resize(np.asarray(arr[i]), (size, size), interpolation=cv2.INTER_AREA).ravel() / 255.0
        np.save(feat_path, allf)
    return np.asarray(allf[df["cache_row"].to_numpy()], dtype=np.float32)


def _core(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(1)
    labels = list(range(len(C.CLASS_NAMES)))
    out = {"accuracy": float(accuracy_score(y, pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=[l for l in labels if l in set(y)], zero_division=0))}
    try:
        present = sorted(set(y))
        aucs = {C.CLASS_NAMES[c]: float(roc_auc_score((y == c).astype(int), prob[:, c])) for c in present if 0 < (y == c).sum() < len(y)}
        out["roc_auc_ovr_macro"] = float(np.mean(list(aucs.values()))) if aucs else None
        out["roc_auc_ovr_per_class"] = aucs
    except ValueError:
        out["roc_auc_ovr_macro"] = None
    return out


def compute_metrics(y: np.ndarray, prob: np.ndarray, groups: np.ndarray | None = None,
                    n_boot: int = 1000, seed: int = 0) -> dict:
    """y: int labels, prob: (N, 4) class probabilities, groups: cluster ids for the bootstrap."""
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=np.float64)
    pred = prob.argmax(1)
    labels = list(range(len(C.CLASS_NAMES)))
    m = _core(y, prob)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    m["per_class"] = {C.CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i in labels}
    m["confusion_matrix"] = confusion_matrix(y, pred, labels=labels).tolist()
    m["n"] = int(len(y))
    if groups is not None and n_boot > 0:
        rng = np.random.default_rng(seed)
        g = np.asarray(groups)
        ug = np.unique(g)
        members = {u: np.where(g == u)[0] for u in ug}
        boots = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
        for _ in range(n_boot):
            idx = np.concatenate([members[u] for u in rng.choice(ug, size=len(ug), replace=True)])
            b = _core(y[idx], prob[idx])
            for k in boots:
                boots[k].append(b[k])
        m["bootstrap_ci95_by_subject"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
        m["bootstrap_units"] = int(len(ug))
    return m


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def library_versions() -> dict:
    import platform

    import sklearn
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    for mod in ("torch", "torchvision", "cv2", "xgboost"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    return v
