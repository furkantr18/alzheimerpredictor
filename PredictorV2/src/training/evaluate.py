"""Subject-level evaluation: metrics, repeated CV, nested CV with tuning, external test.

One row per participant, so every split is subject-disjoint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from common import config as cfg
from training.models import SEARCH_SPACES

METRICS = ["roc_auc", "pr_auc", "balanced_acc", "accuracy", "precision", "recall", "specificity", "f1",
           "macro_f1", "fpr", "fnr", "brier", "log_loss"]


def metrics(y, prob, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),            # = sensitivity = TPR
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "fnr": fn / (fn + tp) if fn + tp else np.nan,
        "brier": brier_score_loss(y, prob),
        "log_loss": log_loss(y, prob, labels=[0, 1]),
    }


def _balance(model, y):
    """XGBoost has no class_weight: balance it with scale_pos_weight from the training labels."""
    if model.named_steps["clf"].__class__.__name__ == "XGBClassifier":
        model.set_params(clf__scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
    return model


def fit(model, X, y, tune: str | None = None, seed: int = cfg.SEED, n_jobs: int = 1):
    """Fit a copy of `model`. With tune=<model name>, run a random search (inner stratified CV on the
    given training data only) and return (fitted best model, best params); otherwise params is {}."""
    m = _balance(clone(model), y)
    if tune is None:
        return m.fit(X, y), {}
    search = RandomizedSearchCV(m, SEARCH_SPACES[tune], n_iter=cfg.TUNE_ITER, scoring="roc_auc", refit=True,
                                cv=StratifiedKFold(cfg.TUNE_INNER_SPLITS, shuffle=True, random_state=seed),
                                random_state=seed, n_jobs=n_jobs)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def repeated_cv(model, X: pd.DataFrame, y: pd.Series, n_splits: int = cfg.N_SPLITS,
                n_repeats: int = cfg.N_REPEATS, seed: int = cfg.SEED, tune: str | None = None,
                n_jobs: int = cfg.N_JOBS):
    """Metrics of each repeat (on that repeat's pooled out-of-fold predictions), the out-of-fold
    probabilities (n_repeats, n_participants), the best params of every outer fold (when tuning) and the
    share of outer folds in which each feature was kept (when the pipeline has a 'select' step).
    With tuning this is nested CV: the search only sees the outer training fold."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(rskf.split(X, y))

    def one_fold(i, tr, te):
        m, p = fit(model, X.iloc[tr], y.iloc[tr], tune=tune, seed=seed + i)
        support = m.named_steps["select"].get_support() if "select" in m.named_steps else None
        return m.predict_proba(X.iloc[te])[:, 1], p, support

    results = Parallel(n_jobs=n_jobs)(delayed(one_fold)(i, tr, te) for i, (tr, te) in enumerate(splits))
    oof = np.full((n_repeats, len(y)), np.nan)
    params, kept = [], []
    for i, ((tr, te), (prob, p, support)) in enumerate(zip(splits, results)):
        oof[i // n_splits, te] = prob
        if tune:
            params.append({"repeat": i // n_splits, "fold": i % n_splits, **p})
        if support is not None:
            kept.append(support)
    per_repeat = pd.DataFrame([metrics(y.values, p) for p in oof])
    frequency = pd.Series(np.mean(kept, axis=0), index=X.columns, name="selected_share") if kept else None
    return per_repeat, oof, pd.DataFrame(params), frequency


def external(model, X_train, y_train, X_test, y_test, tune: str | None = None):
    """Fit once on the training cohort (tuned on it if requested), score once on the test cohort.
    Returns metrics, test probabilities, the fitted model and its params."""
    m, p = fit(model, X_train, y_train, tune=tune, n_jobs=cfg.N_JOBS)
    prob = m.predict_proba(X_test)[:, 1]
    return metrics(y_test.values, prob), prob, m, p


def summarize(per_repeat: pd.DataFrame) -> dict:
    out = {}
    for c in per_repeat.columns:
        out[f"{c}_mean"] = per_repeat[c].mean()
        out[f"{c}_sd"] = per_repeat[c].std(ddof=1)
    return out
