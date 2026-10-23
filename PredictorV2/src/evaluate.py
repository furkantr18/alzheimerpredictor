"""Models and subject-level evaluation (one row per participant, so folds are subject-disjoint)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import adni_config as cfg


def models(seed: int = cfg.SEED) -> dict:
    """Imputer and scaler live inside the pipeline, so they are fit on training folds only.

    Hyperparameters are fixed in advance (logreg and SVM tune C by an inner CV on the training fold),
    so no test data is ever used to choose them.
    """
    def scaled(clf):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)

    def imputed(clf):
        return make_pipeline(SimpleImputer(strategy="median"), clf)

    out = {
        "logreg": scaled(LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", class_weight="balanced",
                                              max_iter=5000, random_state=seed)),
        "svm_rbf": scaled(GridSearchCV(SVC(kernel="rbf", class_weight="balanced", probability=True,
                                           random_state=seed),
                                       {"C": [0.1, 1, 10]}, cv=5, scoring="roc_auc")),
        "knn": scaled(KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "mlp": scaled(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                                    validation_fraction=0.15, max_iter=1000, random_state=seed)),
        "rf": imputed(RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced",
                                             n_jobs=-1, random_state=seed)),
        "extratrees": imputed(ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                                   class_weight="balanced", n_jobs=-1, random_state=seed)),
    }
    try:
        from xgboost import XGBClassifier
        out["xgb"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, eval_metric="logloss", random_state=seed,
                                   n_jobs=-1)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        out["lightgbm"] = LGBMClassifier(n_estimators=300, num_leaves=8, learning_rate=0.05, subsample=0.8,
                                         subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                                         random_state=seed, n_jobs=-1, verbose=-1)
    except ImportError:
        pass
    return out


def metrics(y, prob, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, prob),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "sensitivity": recall_score(y, pred, pos_label=1),
        "specificity": recall_score(y, pred, pos_label=0),
    }


def _fit(model, X, y):
    m = clone(model)
    if m.__class__.__name__ == "XGBClassifier":       # no class_weight: balance with scale_pos_weight
        m.set_params(scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
    return m.fit(X, y)


def repeated_cv(model, X: pd.DataFrame, y: pd.Series, n_splits: int = cfg.N_SPLITS,
                n_repeats: int = cfg.N_REPEATS, seed: int = cfg.SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """Metrics of each repeat (on that repeat's pooled out-of-fold predictions) and the
    out-of-fold probabilities, shape (n_repeats, n_participants)."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof = np.full((n_repeats, len(y)), np.nan)
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        m = _fit(model, X.iloc[tr], y.iloc[tr])
        oof[i // n_splits, te] = m.predict_proba(X.iloc[te])[:, 1]
    return pd.DataFrame([metrics(y.values, p) for p in oof]), oof


def external(model, X_train, y_train, X_test, y_test) -> tuple[dict, np.ndarray, object]:
    """Fit once on the training cohort, score once on the test cohort. Returns metrics, test
    probabilities and the fitted model."""
    m = _fit(model, X_train, y_train)
    prob = m.predict_proba(X_test)[:, 1]
    return metrics(y_test.values, prob), prob, m


def summarize(per_repeat: pd.DataFrame) -> dict:
    out = {}
    for c in per_repeat.columns:
        out[f"{c}_mean"] = per_repeat[c].mean()
        out[f"{c}_sd"] = per_repeat[c].std(ddof=1)
    return out
