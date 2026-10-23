"""The 8 classifiers, with fixed default hyperparameters and random-search spaces for tuning.

Every model is a Pipeline whose last step is called 'clf', so search spaces use the 'clf__' prefix.
Imputation and scaling are pipeline steps, so they are fit on training data only.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import config as cfg


class AnovaFdrSelector(BaseEstimator, TransformerMixin):
    """Keep features whose one-way ANOVA F-test (sklearn f_classif) is significant after Benjamini-Hochberg
    FDR correction at `alpha`. As a pipeline step it is fit on the training data of each fold only, so the
    test fold never influences which features are kept (no selection leakage). Keeps at least the single
    best feature so a fold never ends up empty."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.scores_, p = f_classif(X, y)
        p = np.nan_to_num(p, nan=1.0)
        order = np.argsort(p)
        m = len(p)
        passed = p[order] <= self.alpha * np.arange(1, m + 1) / m
        k = passed.nonzero()[0].max() + 1 if passed.any() else 0
        support = np.zeros(m, dtype=bool)
        support[order[:max(k, 1)]] = True
        self.pvalues_, self.support_ = p, support
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

    def get_support(self):
        return self.support_


def with_selection(model: Pipeline, alpha: float = 0.05) -> Pipeline:
    """Insert imputation (if missing) and the ANOVA-FDR selector before scaling / the classifier."""
    steps = list(model.steps)
    if steps[0][0] != "impute":
        steps.insert(0, ("impute", SimpleImputer(strategy="median")))
    steps.insert(1, ("select", AnovaFdrSelector(alpha)))
    return Pipeline(steps)


def _scaled(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clf)])


def _imputed(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])


def models(seed: int = cfg.SEED, select: bool = False) -> dict[str, Pipeline]:
    """Fixed hyperparameters, chosen before seeing any result (library defaults or common choices).
    Every model runs single-threaded: parallelism is across CV folds (see evaluate.repeated_cv).
    select=True adds the leakage-free ANOVA-FDR feature selection step to every model."""
    out = {
        "logreg": _scaled(LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)),
        "svm_rbf": _scaled(SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=seed)),
        "knn": _scaled(KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "mlp": _scaled(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                                     validation_fraction=0.15, max_iter=1000, random_state=seed)),
        "rf": _imputed(RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced",
                                              n_jobs=1, random_state=seed)),
        "extratrees": _imputed(ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced",
                                                    n_jobs=1, random_state=seed)),
    }
    try:
        from xgboost import XGBClassifier
        out["xgb"] = Pipeline([("clf", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                     subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                                     random_state=seed, n_jobs=1))])
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        out["lightgbm"] = Pipeline([("clf", LGBMClassifier(n_estimators=300, num_leaves=8, learning_rate=0.05,
                                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                           class_weight="balanced", random_state=seed, n_jobs=1,
                                                           verbose=-1))])
    except ImportError:
        pass
    return {k: with_selection(v) for k, v in out.items()} if select else out


SEARCH_SPACES = {
    "logreg": {"clf__C": loguniform(1e-3, 1e2)},
    "svm_rbf": {"clf__C": loguniform(1e-2, 1e2), "clf__gamma": loguniform(1e-4, 1e-1)},
    "knn": {"clf__n_neighbors": randint(5, 51), "clf__weights": ["uniform", "distance"]},
    "mlp": {"clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)], "clf__alpha": loguniform(1e-5, 1e-1),
            "clf__learning_rate_init": loguniform(1e-4, 1e-2)},
    "rf": {"clf__n_estimators": randint(200, 801), "clf__max_depth": [None, 3, 5, 8, 12],
           "clf__min_samples_leaf": randint(1, 11), "clf__max_features": ["sqrt", 0.3, 0.5]},
    "extratrees": {"clf__n_estimators": randint(200, 801), "clf__max_depth": [None, 3, 5, 8, 12],
                   "clf__min_samples_leaf": randint(1, 11), "clf__max_features": ["sqrt", 0.3, 0.5]},
    "xgb": {"clf__n_estimators": randint(100, 601), "clf__max_depth": randint(2, 7),
            "clf__learning_rate": loguniform(1e-2, 0.2), "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": randint(1, 11)},
    "lightgbm": {"clf__n_estimators": randint(100, 601), "clf__num_leaves": randint(4, 33),
                 "clf__learning_rate": loguniform(1e-2, 0.2), "clf__subsample": uniform(0.6, 0.4),
                 "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_samples": randint(5, 41)},
}
