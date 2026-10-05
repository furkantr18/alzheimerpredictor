"""Classical MRI pipeline: shared preprocessing -> pixels (64x64) -> StandardScaler -> PCA -> classifier.

Rewritten from Yasemin's version (ae3bce0). Same 6 model families, with these fixes:
* data come from a saved, leakage-aware split CSV (make_splits.py), not a random split;
* hyper-parameters are tuned with StratifiedGroupKFold on the TRAINING set only, grouped by
  pseudo-subject; the CV validation folds are scored on ORIGINAL images only;
* PCA is randomized with a fixed number of components (the old `svd_solver="full"` on
  ~30k x 16k pixels needed ~8 GB RAM); SVC has no `probability=True` 5-fold refit, its
  scores are calibrated by a softmax over decision values;
* unreadable images never reach this script (cache build skips and counts them);
* the test split is NOT used here (final_evaluation.py uses it once).

Usage:
    python src/imgProcessing/mri_model_trainer.py [--split split_subject] [--models all] [--seed 42]
Output: output/models/classical_<split>_<model>.joblib, output/reports/classical_<split>.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402
from mri_data import compute_metrics, library_versions, load_split, pixel_features, save_json  # noqa: E402
from mri_preprocess import PreprocessSpec  # noqa: E402

FEATURE_SIZE = 64
PCA_COMPONENTS = 128


class ConsecutiveLabelXGB(BaseEstimator, ClassifierMixin):
    """XGBoost needs labels 0..k-1. With only 2 ModerateDemented subjects, a grouped CV fold can lack
    that class entirely, so labels are re-indexed here and `classes_` keeps the real label ids."""

    def __init__(self, max_depth: int = 6, seed: int = 42):
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier
        self.classes_ = np.unique(y)
        self.model_ = XGBClassifier(n_estimators=300, max_depth=self.max_depth, learning_rate=0.1, subsample=0.9,
                                    colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
                                    tree_method="hist", device="cpu", random_state=self.seed)
        self.model_.fit(X, np.searchsorted(self.classes_, y), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# Pickle by module name, not "__main__", so artifacts load from any script (API, evaluation).
ConsecutiveLabelXGB.__module__ = "mri_model_trainer"


def load_artifact(path):
    """joblib.load that also reads artifacts pickled while this file ran as __main__."""
    main_mod = sys.modules["__main__"]
    if not hasattr(main_mod, "ConsecutiveLabelXGB"):
        main_mod.ConsecutiveLabelXGB = ConsecutiveLabelXGB
    return joblib.load(path)


def make_model(name: str, params: dict, seed: int):
    if name == "LogisticRegression":
        return LogisticRegression(C=params["C"], max_iter=3000, class_weight="balanced", random_state=seed)
    if name == "SVM_RBF":
        return SVC(C=params["C"], kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=params["k"], weights="distance", n_jobs=-1)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(256,), alpha=params["alpha"], early_stopping=True, max_iter=300, random_state=seed)
    if name == "XGBoost":
        return ConsecutiveLabelXGB(max_depth=params["max_depth"], seed=seed)
    raise ValueError(name)


GRIDS = {
    "LogisticRegression": {"C": [0.001, 0.01, 0.1]},
    "SVM_RBF": {"C": [1.0, 10.0]},
    "RandomForest": {"max_depth": [None, 16], "min_samples_leaf": [1, 5]},
    "KNN": {"k": [5, 15, 41]},
    "MLP": {"alpha": [1e-3, 1e-1]},
    "XGBoost": {"max_depth": [3, 6]},
}
NEEDS_SAMPLE_WEIGHT = {"MLP": False, "XGBoost": True, "KNN": False}



def predict_proba(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    clf = pipe[-1]
    if hasattr(clf, "predict_proba"):
        p = pipe.predict_proba(X)
    else:  # SVC without probability=True: softmax over one-vs-rest decision values
        d = pipe.decision_function(X)
        d = d - d.max(1, keepdims=True)
        p = np.exp(d) / np.exp(d).sum(1, keepdims=True)
    full = np.zeros((len(X), len(C.CLASS_NAMES)))
    full[:, clf.classes_] = p          # classes missing from training get probability 0
    return full


def fit_pipe(name: str, params: dict, X, y, seed: int) -> Pipeline:
    pipe = Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=seed)),
                     ("clf", make_model(name, params, seed))])
    kw = {}
    if NEEDS_SAMPLE_WEIGHT.get(name):
        kw["clf__sample_weight"] = compute_sample_weight("balanced", y)
    pipe.fit(X, y, **kw)
    return pipe


def grouped_cv_folds(train_df, n_splits: int, seed: int):
    """Folds over pseudo-subjects; validation indices = ORIGINAL images of the held-out subjects."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    is_orig = (train_df["kind"] == "original").to_numpy() if "kind" in train_df else np.ones(len(train_df), bool)
    for tr, va in sgkf.split(train_df, train_df["y"], train_df["subject_group"]):
        yield tr, va[is_orig[va]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="split_subject")
    ap.add_argument("--models", default="all")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    C.ensure_dirs()
    np.random.seed(args.seed)

    df = load_split(args.split)
    tr, va = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "val"].reset_index(drop=True)
    Xtr, ytr = pixel_features(tr, FEATURE_SIZE), tr["y"].to_numpy()
    Xva, yva = pixel_features(va, FEATURE_SIZE), va["y"].to_numpy()
    print(f"[classical] split={args.split} train={len(tr)} val={len(va)} (test untouched)", flush=True)

    names = list(GRIDS) if args.models == "all" else args.models.split(",")
    folds = list(grouped_cv_folds(tr, args.folds, args.seed))
    report_path = C.REPORTS_DIR / f"classical_{args.split}.json"
    report = {"split": args.split, "seed": args.seed, "feature_size": FEATURE_SIZE, "pca_components": PCA_COMPONENTS,
              "cv": f"StratifiedGroupKFold({args.folds}) by subject_group on train; scored on originals", "models": {}}
    if report_path.exists():  # re-running a subset of models keeps the others
        report["models"] = json.loads(report_path.read_text(encoding="utf-8")).get("models", {})

    for name in names:
        t0 = time.time()
        best = None
        cv_rows = []
        for combo in itertools.product(*GRIDS[name].values()):
            params = dict(zip(GRIDS[name].keys(), combo))
            scores = []
            for ftr, fva in folds:
                pipe = fit_pipe(name, params, Xtr[ftr], ytr[ftr], args.seed)
                scores.append(compute_metrics(ytr[fva], predict_proba(pipe, Xtr[fva]), n_boot=0)["macro_f1"])
            row = {"params": params, "cv_macro_f1_mean": float(np.mean(scores)), "cv_macro_f1_std": float(np.std(scores))}
            cv_rows.append(row)
            print(f"[classical] {name} {params} cv macro-F1 {row['cv_macro_f1_mean']:.3f} +/- {row['cv_macro_f1_std']:.3f}", flush=True)
            if best is None or row["cv_macro_f1_mean"] > best["cv_macro_f1_mean"]:
                best = row
        pipe = fit_pipe(name, best["params"], Xtr, ytr, args.seed)
        train_m = compute_metrics(ytr, predict_proba(pipe, Xtr), n_boot=0)
        val_m = compute_metrics(yva, predict_proba(pipe, Xva), va["subject_group"].to_numpy(), n_boot=200)
        artifact = {"kind": "classical", "model_name": name, "pipeline": pipe, "class_names": C.CLASS_NAMES,
                    "feature_size": FEATURE_SIZE, "preprocess": PreprocessSpec(size=224).to_dict(),
                    "split_file": f"{args.split}.csv", "seed": args.seed, "params": best["params"], "versions": library_versions()}
        joblib.dump(artifact, C.MODELS_DIR / f"classical_{args.split}_{name}.joblib")
        report["models"][name] = {"cv": cv_rows, "best_params": best["params"], "train": train_m, "val": val_m,
                                  "seconds": round(time.time() - t0, 1)}
        print(f"[classical] {name} best={best['params']} train macro-F1 {train_m['macro_f1']:.3f} | val macro-F1 {val_m['macro_f1']:.3f} "
              f"bal-acc {val_m['balanced_accuracy']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        save_json(report, report_path)


if __name__ == "__main__":
    main()
