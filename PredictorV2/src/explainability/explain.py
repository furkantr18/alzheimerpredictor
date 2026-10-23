"""Explain the CV-selected models: SHAP, LIME and permutation importance.

For each task and each feature set in EXPLAIN_SETS, the model selected by CV in the model comparison
(selected_models.csv) is refit on all of ADNI1 with the same hyperparameters and explained on the
external test participants (ADNIGO/2/3), who were never used for training or selection.

Writes to output/3_explainability/<task>/<mode>/<feature set>/:
    shap/                    fig_shap_beeswarm, fig_shap_bar, shap_values.csv
    lime/                    fig_lime_<case>, lime_<case>.csv for three participants:
                             a correct positive, a correct negative and the most confident error
    permutation_importance/  fig_permutation_importance, permutation_importance.csv
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from common import cohort as co
from common import config as cfg
from common import features as ft
from common import plots
from common.plots import plt
from training import evaluate as ev
from training.models import models

warnings.filterwarnings("ignore")

EXPLAIN_SETS = ["demo+mri_eng", "demo+mri_fs7"]
TOP = 15
N_SHAP = 200          # external participants explained with SHAP (all if fewer)
TREE_MODELS = {"rf", "extratrees", "xgb", "lightgbm"}
DISCRETE = {"PTGENDER", "APOE4"}           # treated as categorical by LIME


def load_selection(task: str, mode: str) -> pd.DataFrame:
    path = cfg.OUTPUT_DIR / cfg.STAGES["models"] / task / mode / "tables" / "selected_models.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the model comparison first (python PredictorV2/run.py models)")
    return pd.read_csv(path)


def rebuild(model_name: str, params_json: str, select: bool = False):
    """The selected model with its saved hyperparameters (only keys the estimator accepts)."""
    m = models(select=select)[model_name]
    clf = m.named_steps["clf"]
    params = {k: v for k, v in json.loads(params_json or "{}").items() if k in clf.get_params()}
    for k, v in params.items():
        if isinstance(v, list):
            params[k] = tuple(v)
        if isinstance(v, str) and v.lower() in ("none", "nan"):
            params[k] = None
    clf.set_params(**params)
    return m


def _preprocess(model, X: pd.DataFrame) -> pd.DataFrame:
    """Inputs after imputation (raw units, so explanations read in mm, ratios, years)."""
    if "impute" in model.named_steps:
        return pd.DataFrame(model.named_steps["impute"].transform(X), columns=X.columns, index=X.index)
    return X


def _predict_from_imputed(model):
    """Probability of the positive class for already-imputed inputs."""
    steps = [s for name, s in model.steps if name != "impute"]
    def f(X):
        X = np.asarray(X, dtype=float)
        for s in steps[:-1]:
            X = s.transform(X)
        return steps[-1].predict_proba(X)[:, 1]
    return f


def shap_explain(model, model_name, X_train, X_test, names, task, out):
    import shap
    Xtr, Xte = _preprocess(model, X_train), _preprocess(model, X_test)
    Xte = Xte.sample(min(N_SHAP, len(Xte)), random_state=cfg.SEED)
    display = Xte.rename(columns=names)
    if model_name in TREE_MODELS:
        est = model.named_steps["clf"]
        if "select" in model.named_steps:                 # tree sees only the kept features
            keep = model.named_steps["select"].get_support()
            Xte, display = Xte.loc[:, keep], display.loc[:, keep]
        sv = shap.TreeExplainer(est)(Xte)
        if sv.values.ndim == 3:                            # (n, features, classes) -> positive class
            sv = sv[:, :, 1]
    else:
        background = shap.maskers.Independent(Xtr, max_samples=100)
        explainer = shap.Explainer(_predict_from_imputed(model), background, feature_names=list(Xte.columns))
        sv = explainer(Xte.values, max_evals=2 * Xte.shape[1] + 1)
    sv.feature_names = list(display.columns)
    sv.data = display.values
    pd.DataFrame(sv.values, columns=display.columns, index=Xte.index).to_csv(out / "shap_values.csv")

    a, b = plots.LABEL_NAMES[task]
    for kind in ("beeswarm", "bar"):
        plt.figure()
        getattr(shap.plots, kind)(sv, max_display=TOP, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7, 0.35 * TOP + 1.5)
        plt.title(f"{a} vs {b}: SHAP {kind} ({model_name}, external test)", fontsize=10)
        plots.save(fig, out / f"fig_shap_{kind}")
    return sv, Xte


def pick_cases(y_true: pd.Series, prob: np.ndarray) -> dict[str, int]:
    """Positions of a confident correct positive, a confident correct negative and the most confident error."""
    pred = (prob >= 0.5).astype(int)
    y = y_true.values
    cases = {}
    tp = np.where((y == 1) & (pred == 1))[0]
    tn = np.where((y == 0) & (pred == 0))[0]
    err = np.where(y != pred)[0]
    if len(tp):
        cases["correct_positive"] = tp[np.argmax(prob[tp])]
    if len(tn):
        cases["correct_negative"] = tn[np.argmin(prob[tn])]
    if len(err):
        cases["most_confident_error"] = err[np.argmax(np.abs(prob[err] - 0.5))]
    return cases


def is_icv_ratio(column: str) -> bool:
    """Engineered or ADNIMERGE '_icv' features and FreeSurfer 7 volumes (SV/CV codes) are divided by ICV."""
    return column.endswith("_icv") or (column.startswith("ST") and column[-2:] in ("SV", "CV"))


def lime_explain(model, X_train, y_train, X_test, y_test, names, task, out):
    from lime.lime_tabular import LimeTabularExplainer
    Xtr, Xte = _preprocess(model, X_train), _preprocess(model, X_test)
    f = _predict_from_imputed(model)
    # Volume / ICV ratios are ~0.001-0.05 and LIME prints 2 decimals, so LIME sees them in per mille.
    ratio = np.array([is_icv_ratio(c) for c in Xtr.columns])
    scale = np.where(ratio, 1000.0, 1.0)
    Xtr, Xte = Xtr * scale, Xte * scale
    proba = lambda X: np.column_stack([1 - f(np.asarray(X) / scale), f(np.asarray(X) / scale)])
    feature_names = [names.get(c, c) + (" (‰ ICV)" if r else "") for c, r in zip(Xtr.columns, ratio)]
    categorical = [i for i, c in enumerate(Xtr.columns) if c in DISCRETE]

    def make(discretize):
        return LimeTabularExplainer(Xtr.values, feature_names=feature_names, categorical_features=categorical,
                                    class_names=list(plots.LABEL_NAMES[task]), training_labels=y_train.values,
                                    discretize_continuous=discretize, random_state=cfg.SEED)
    explainer = make(True)
    prob = f(Xte.values / scale)
    a, b = plots.LABEL_NAMES[task]
    for case, i in pick_cases(y_test, prob).items():
        try:
            exp = explainer.explain_instance(Xte.values[i], proba, num_features=10, num_samples=3000)
        except ValueError:
            # Known LIME bug: the quartile discretizer fails when a bin has zero spread.
            explainer = make(False)
            exp = explainer.explain_instance(Xte.values[i], proba, num_features=10, num_samples=3000)
        weights = pd.DataFrame(exp.as_list(), columns=["condition", "weight"])
        weights.to_csv(out / f"lime_{case}.csv", index=False)
        w = weights.iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(w["condition"], w["weight"], height=0.6,
                color=[plots.SERIES[1] if v > 0 else plots.SERIES[0] for v in w["weight"]])
        ax.axvline(0, color=plots.TEXT_2, lw=0.8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"LIME weight (towards {b} if > 0)")
        true = (a, b)[int(y_test.values[i])]
        ax.set_title(f"{case.replace('_', ' ')}: true {true}, P({b}) = {prob[i]:.2f}", fontsize=10)
        plots.save(fig, out / f"fig_lime_{case}")


def permutation_explain(model, X_test, y_test, names, task, out):
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=cfg.SEED,
                               n_jobs=-1)
    imp = pd.DataFrame({"feature": [names.get(c, c) for c in X_test.columns], "mean": r.importances_mean,
                        "sd": r.importances_std}).sort_values("mean", ascending=False)
    imp.to_csv(out / "permutation_importance.csv", index=False)
    top = imp.head(TOP)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * TOP + 1.2))
    ax.barh(top["feature"], top["mean"], xerr=top["sd"], color=plots.SERIES[0], height=0.6,
            error_kw={"ecolor": plots.TEXT_2, "elinewidth": 1})
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled (external test)")
    ax.grid(axis="y", visible=False)
    a, b = plots.LABEL_NAMES[task]
    ax.set_title(f"{a} vs {b}: permutation importance")
    plots.save(fig, out / "fig_permutation_importance")


def main(tasks=cfg.TASKS, mode="fixed_hyperparameters", set_names=EXPLAIN_SETS):
    plots.setup()
    merge, fs = co.load_adnimerge(), co.load_fs7()
    names = {**ft.fs7_names(), **{k: k.replace("_", " ") for k in ft.ENGINEERED}}
    for task in tasks:
        sel = load_selection(task, mode).set_index("features")
        c, matched, _ = co.modeling_cohort(merge, fs, task)
        sets = ft.feature_sets(c, matched)
        y = c["label"]
        train = (c["phase"] == cfg.TRAIN_PHASE).values
        test = c["phase"].isin(cfg.TEST_PHASES).values
        for s in set_names:
            if s not in sel.index:
                continue
            model_name = sel.loc[s, "model"]
            params = sel.loc[s, "params"] if isinstance(sel.loc[s, "params"], str) else "{}"
            X = sets[s]
            model, _ = ev.fit(rebuild(model_name, params, select="anova_selection" in mode), X[train], y[train])
            base = cfg.out_dir("explain", task, mode, s.replace("+", "_"))
            print(f"[{task}] {s}: explaining {model_name}")
            shap_explain(model, model_name, X[train], X[test], names, task, cfg.out_dir("explain", task,
                         mode, s.replace("+", "_"), "shap"))
            lime_explain(model, X[train], y[train], X[test], y[test], names, task,
                         cfg.out_dir("explain", task, mode, s.replace("+", "_"), "lime"))
            permutation_explain(model, X[test], y[test], names, task,
                                cfg.out_dir("explain", task, mode, s.replace("+", "_"), "permutation_importance"))
            (base / "model.txt").write_text(f"{model_name}\n{params}\n", encoding="utf-8")
