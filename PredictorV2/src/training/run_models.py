"""Model comparison: CN vs AD and sMCI vs pMCI from baseline data.

Experiments (one row per participant, so every split is subject-level):
    cv_ADNI1     repeated stratified CV inside ADNI1 (one scanner protocol, 1.5T)
    ext_ADNI2-3  train on all of ADNI1, test once on participants enrolled in ADNIGO/2/3
                 (3T, other sites and years). The ADNIMERGE volumes are skipped here because their
                 FreeSurfer version changes between phases.

Modes, written to separate folders:
    fixed_hyperparameters  hyperparameters fixed in advance (fast; all feature sets)
    tuned_hyperparameters  nested CV: a random search inside every outer training fold; the external
                           model is tuned on all of ADNI1. Best params of every fold are saved.
    ..._anova_selection    (--select) an ANOVA F-test + FDR feature filter is a pipeline step, so it is
                           refit on the training folds only (no selection leakage). The share of folds
                           that kept each feature is saved.

Model selection uses the CV results only: per feature set, the model with the highest mean CV
ROC-AUC is 'selected'. The external test is reported for every model but must not be used to pick one.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from common import cohort as co
from common import config as cfg
from common import features as ft
from common import plots
from common.plots import plt
from training import evaluate as ev
from training.models import SEARCH_SPACES, models

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

ALL_SETS = ["demo", "demo+cog", "demo+mri_merge", "demo+mri_eng", "demo+mri_fs7", "demo+cog+mri_eng",
            "demo+cog+mri_fs7"]
TUNE_SETS = ["demo+mri_eng", "demo+mri_fs7", "demo+cog+mri_eng"]
CURVE_SETS = ["demo", "demo+mri_eng", "demo+mri_fs7"]     # at most three lines per panel
MAIN_SET = "demo+mri_fs7"
EXT = "ext_ADNI2-3"
CV = f"cv_{cfg.TRAIN_PHASE}"


def _jsonable(params: dict) -> str:
    return json.dumps({k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in params.items()},
                      default=str)


def mode_name(tune: bool, select: bool) -> str:
    return ("tuned" if tune else "fixed") + "_hyperparameters" + ("_anova_selection" if select else "")


def run_task(task, merge, fs, repeats, tune, set_names, model_names, select=False):
    mode = mode_name(tune, select)
    tables = cfg.out_dir("models", task, mode, "tables")
    figures = cfg.out_dir("models", task, mode, "figures")
    hyper = cfg.out_dir("models", task, mode, "hyperparameters")

    c, matched, n_all = co.modeling_cohort(merge, fs, task)
    print(f"\n[{task} | {mode}] {n_all} participants, {len(c)} with a FreeSurfer 7 baseline scan")
    sets = {k: v for k, v in ft.feature_sets(c, matched).items() if k in set_names}
    y = c["label"]
    train = (c["phase"] == cfg.TRAIN_PHASE).values
    test = c["phase"].isin(cfg.TEST_PHASES).values

    rows, reps, fold_params, final_params, oof, ext_prob, kept = [], [], [], [], {}, {}, []
    for set_name, X in sets.items():
        for model_name, model in models(select=select).items():
            if model_name not in model_names:
                continue
            key = (set_name, model_name)
            tune_name = model_name if tune else None
            pr, o, fp, freq = ev.repeated_cv(model, X[train], y[train], n_repeats=repeats, tune=tune_name)
            oof[key] = o
            rows.append({"experiment": CV, "features": set_name, "model": model_name, "n": int(train.sum()),
                         "n_pos": int(y[train].sum()), **ev.summarize(pr)})
            reps.append(pr.assign(features=set_name, model=model_name, repeat=range(len(pr))))
            if len(fp):
                fold_params.append(fp.assign(features=set_name, model=model_name))
            line = f"  {set_name:18s} {model_name:10s} CV AUC {pr['roc_auc'].mean():.3f}"
            if "mri_merge" not in set_name:
                m, p, fitted, params = ev.external(model, X[train], y[train], X[test], y[test], tune=tune_name)
                ext_prob[key] = p
                rows.append({"experiment": EXT, "features": set_name, "model": model_name, "n": int(test.sum()),
                             "n_pos": int(y[test].sum()), **{f"{k}_mean": v for k, v in m.items()}})
                final_params.append({"features": set_name, "model": model_name,
                                     "params": _jsonable(params or fitted.named_steps["clf"].get_params())})
                if freq is not None:
                    kept.append(freq.to_frame().assign(
                        kept_in_final_ADNI1_model=fitted.named_steps["select"].get_support(),
                        features=set_name, model=model_name).rename_axis("feature").reset_index())
                line += f"   external AUC {m['roc_auc']:.3f}"
            print(line, flush=True)

    res = pd.DataFrame(rows)
    cv = res[res.experiment == CV]
    best = cv.loc[cv.groupby("features")["roc_auc_mean"].idxmax(), ["features", "model"]]
    res["selected"] = res.set_index(["features", "model"]).index.isin(best.set_index(["features", "model"]).index)
    selected = dict(best.values)

    res.to_csv(tables / "results_all_metrics.csv", index=False)
    pd.concat(reps).to_csv(tables / "results_per_repeat.csv", index=False)
    comparison_table(res).to_csv(tables / "comparison_roc_auc.csv")
    sel = selected_table(res, pd.DataFrame(final_params))
    sel.to_csv(tables / "selected_models.csv", index=False)
    pd.DataFrame(final_params).to_csv(hyper / ("final_params_trained_on_ADNI1.csv"), index=False)
    if fold_params:
        write_param_tables(pd.concat(fold_params), hyper)

    if kept:
        k = pd.concat(kept)
        k = k[["features", "model", "feature", "selected_share", "kept_in_final_ADNI1_model"]]
        k.sort_values(["features", "model", "selected_share"], ascending=[True, True, False]).to_csv(
            tables / "feature_selection_frequency.csv", index=False)
    fig_model_comparison(res, task, figures)
    fig_auc_heatmap(res, task, figures)
    fig_curves(task, y, train, test, oof, ext_prob, selected, figures, kind="roc")
    fig_curves(task, y, train, test, oof, ext_prob, selected, figures, kind="pr")
    fig_calibration(task, y, test, ext_prob, selected, figures)
    if MAIN_SET in selected:
        fig_confusion(task, y, train, test, oof, ext_prob, (MAIN_SET, selected[MAIN_SET]), figures)
    write_summary(task, mode, res, repeats, len(c), n_all)
    return res


def write_param_tables(fold_params: pd.DataFrame, hyper) -> None:
    """best_params_per_outer_fold.csv (long: one row per fold and parameter) and best_params_summary.csv
    (per feature set, model and parameter: median for numbers, most frequent value otherwise)."""
    keys = ["features", "model", "repeat", "fold"]
    fold_params = fold_params.rename(columns=lambda k: k.replace("clf__", ""))
    long = fold_params.melt(id_vars=keys, var_name="parameter", value_name="value")
    # Keep only each model's own parameters; a missing value there means None (e.g. max_depth=None).
    own = long.apply(lambda r: f"clf__{r['parameter']}" in SEARCH_SPACES.get(r["model"], {}), axis=1)
    long = long[own].fillna({"value": "None"})
    long.sort_values(keys + ["parameter"]).to_csv(hyper / "best_params_per_outer_fold.csv", index=False)

    def summary(v: pd.Series) -> pd.Series:
        num = pd.to_numeric(v, errors="coerce")
        if num.notna().all():
            return pd.Series({"chosen_value": num.median(), "statistic": "median", "min": num.min(),
                              "max": num.max(), "n_folds": len(v)})
        mode = v.astype(str).value_counts()
        return pd.Series({"chosen_value": mode.index[0], "statistic": f"most frequent ({mode.iloc[0]}/{len(v)})",
                          "min": "", "max": "", "n_folds": len(v)})

    long.groupby(["features", "model", "parameter"])["value"].apply(summary).unstack().reset_index().to_csv(
        hyper / "best_params_summary.csv", index=False)


def comparison_table(res: pd.DataFrame) -> pd.DataFrame:
    """Model x feature set, 'CV mean ± SD | external' ROC-AUC."""
    cv = res[res.experiment == CV].set_index(["model", "features"])
    ext = res[res.experiment == EXT].set_index(["model", "features"])["roc_auc_mean"]
    cell = cv.apply(lambda r: f"{r.roc_auc_mean:.3f} ± {r.roc_auc_sd:.3f}", axis=1)
    cell = cell + ext.reindex(cell.index).map(lambda v: "" if pd.isna(v) else f" | {v:.3f}")
    return cell.unstack("features")


def selected_table(res: pd.DataFrame, final_params: pd.DataFrame) -> pd.DataFrame:
    """One row per feature set: the CV-selected model with all CV and external metrics and its params."""
    rows = []
    for (s, m), r in res[res.selected & (res.experiment == CV)].set_index(["features", "model"]).iterrows():
        row = {"features": s, "model": m, **{f"cv_{k}": r[f"{k}_mean"] for k in ev.METRICS},
               **{f"cv_{k}_sd": r[f"{k}_sd"] for k in ev.METRICS}}
        e = res[(res.experiment == EXT) & (res.features == s) & (res.model == m)]
        if len(e):
            row.update({f"ext_{k}": e.iloc[0][f"{k}_mean"] for k in ev.METRICS})
        p = final_params[(final_params.features == s) & (final_params.model == m)] if len(final_params) else []
        row["params"] = p.iloc[0]["params"] if len(p) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _title(fig, task, text):
    a, b = plots.LABEL_NAMES[task]
    fig.suptitle(f"{a} vs {b}: {text}", fontsize=11, fontweight="bold", color=plots.TEXT)


def fig_model_comparison(res, task, out):
    cv = res[res.experiment == CV]
    ext = res[res.experiment == EXT].set_index(["features", "model"])["roc_auc_mean"]
    set_names = list(dict.fromkeys(cv["features"]))
    model_names = list(dict.fromkeys(cv["model"]))
    ncols = 4
    nrows = int(np.ceil(len(set_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.9 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax, s in zip(axes.flat, set_names):
        d = cv[cv.features == s].set_index("model").reindex(model_names)
        yy = np.arange(len(model_names))
        ax.errorbar(d["roc_auc_mean"], yy, xerr=d["roc_auc_sd"], fmt="o", ms=5, color=plots.SERIES[0],
                    elinewidth=1.5, capsize=0, label=f"CV in {cfg.TRAIN_PHASE} (mean ± SD)")
        e = [ext.get((s, m), np.nan) for m in model_names]
        if not np.all(np.isnan(e)):
            ax.plot(e, yy, "D", ms=5, color=plots.SERIES[1], label="External ADNIGO/2/3")
        sel = d["selected"].fillna(False).astype(bool).values
        # The y-axis is shared, so the selection is drawn as a ring, not written into the tick labels.
        ax.plot(d["roc_auc_mean"][sel], yy[sel], "o", ms=12, mfc="none", mec=plots.TEXT, mew=1.2,
                label="selected by CV")
        ax.set_yticks(yy, model_names)
        ax.set_title(s, fontsize=9)
        ax.set_xlim(0.45, 1.04)
        ax.axvline(0.5, color=plots.TEXT_2, lw=0.8, ls=":")
        ax.grid(axis="y", visible=False)
    for ax in axes.flat[len(set_names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("ROC-AUC")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.02))
    _title(fig, task, "model comparison (ring = model selected by CV)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plots.save(fig, out / "fig_model_comparison")


def fig_auc_heatmap(res, task, out):
    cv = res[res.experiment == CV].pivot(index="model", columns="features", values="roc_auc_mean")
    cv = cv[[s for s in ALL_SETS if s in cv.columns]]
    fig, ax = plt.subplots(figsize=(1.1 * cv.shape[1] + 2, 0.45 * cv.shape[0] + 1.5))
    im = ax.imshow(cv.values, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(cv.shape[0]):
        for j in range(cv.shape[1]):
            v = cv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.82 else plots.TEXT)
    ax.set_xticks(range(cv.shape[1]), cv.columns, rotation=30, ha="right")
    ax.set_yticks(range(cv.shape[0]), cv.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"mean CV ROC-AUC ({cfg.TRAIN_PHASE})")
    _title(fig, task, "CV ROC-AUC per model and feature set")
    plots.save(fig, out / "fig_auc_heatmap")


def fig_curves(task, y, train, test, oof, ext_prob, selected, out, kind="roc"):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), sharey=True)
    panels = [(axes[0], f"CV in {cfg.TRAIN_PHASE} (out-of-fold, repeat 1)", y[train].values, lambda k: oof[k][0]),
              (axes[1], "External test ADNIGO/2/3", y[test].values, lambda k: ext_prob.get(k))]
    for ax, title, yt, get in panels:
        for i, s in enumerate(CURVE_SETS):
            if s not in selected or get((s, selected[s])) is None:
                continue
            p = get((s, selected[s]))
            m = ev.metrics(yt, p)
            if kind == "roc":
                fx, fy, _ = roc_curve(yt, p)
                label = f"{s} ({selected[s]}), AUC {m['roc_auc']:.3f}"
            else:
                fy, fx, _ = precision_recall_curve(yt, p)
                label = f"{s} ({selected[s]}), AP {m['pr_auc']:.3f}"
            ax.plot(fx, fy, color=plots.SERIES[i], label=label)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color=plots.TEXT_2, lw=0.8, ls=":")
            ax.set_xlabel("False positive rate")
        else:
            ax.axhline(yt.mean(), color=plots.TEXT_2, lw=0.8, ls=":", label=f"chance (prevalence {yt.mean():.2f})")
            ax.set_xlabel("Recall")
        ax.set_title(title, fontsize=9)
        ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("True positive rate" if kind == "roc" else "Precision")
    _title(fig, task, ("ROC" if kind == "roc" else "precision-recall") + " curves of the CV-selected models")
    fig.tight_layout()
    plots.save(fig, out / f"fig_{kind}_curves")


def fig_calibration(task, y, test, ext_prob, selected, out):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    yt = y[test].values
    for i, s in enumerate(CURVE_SETS):
        if s not in selected or (s, selected[s]) not in ext_prob:
            continue
        p = ext_prob[(s, selected[s])]
        frac, mean_p = calibration_curve(yt, p, n_bins=8, strategy="quantile")
        ax.plot(mean_p, frac, "o-", ms=4, color=plots.SERIES[i],
                label=f"{s} ({selected[s]}), Brier {ev.metrics(yt, p)['brier']:.3f}")
    ax.plot([0, 1], [0, 1], color=plots.TEXT_2, lw=0.8, ls=":", label="perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")
    _title(fig, task, "calibration, external test")
    plots.save(fig, out / "fig_calibration")


def fig_confusion(task, y, train, test, oof, ext_prob, key, out):
    names = plots.LABEL_NAMES[task]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for ax, title, yt, p in [(axes[0], f"CV in {cfg.TRAIN_PHASE} (repeat 1)", y[train].values, oof[key][0]),
                             (axes[1], "External test ADNIGO/2/3", y[test].values, ext_prob[key])]:
        cm = confusion_matrix(yt, (p >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({100 * cm[i, j] / cm[i].sum():.0f}%)", ha="center", va="center",
                        fontsize=9, color="white" if cm[i, j] > cm.max() * 0.6 else plots.TEXT)
        ax.set_xticks([0, 1], [f"pred {n}" for n in names])
        ax.set_yticks([0, 1], [f"true {n}" for n in names])
        ax.grid(False)
        ax.set_title(title, fontsize=9)
    _title(fig, task, f"confusion matrices, {key[0]} ({key[1]})")
    fig.tight_layout()
    plots.save(fig, out / "fig_confusion_matrices")


def write_summary(task, mode, res, repeats, n, n_all):
    lines = [f"# {task}: model comparison ({mode.replace('_', ' ')})", "",
             f"{n} of {n_all} participants have a FreeSurfer 7 baseline scan. 5-fold stratified CV x {repeats} "
             f"repeats in {cfg.TRAIN_PHASE}; external test: train {cfg.TRAIN_PHASE}, test "
             f"{'/'.join(cfg.TEST_PHASES)}. CV = mean (SD) over repeats. Selected = best mean CV ROC-AUC per "
             "feature set; the external test is never used for selection. 'cog' is circular for CN_AD.", "",
             "| features | model | CV ROC-AUC | CV PR-AUC | CV bal. acc. | CV F1 | ext ROC-AUC | ext PR-AUC | "
             "ext bal. acc. | ext F1 | ext Brier |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    cv = res[res.experiment == CV].set_index(["features", "model"])
    ext = res[res.experiment == EXT].set_index(["features", "model"])
    f = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    for (s, m), r in cv.iterrows():
        e = ext.loc[(s, m)] if (s, m) in ext.index else None
        g = (lambda k: f(e[f"{k}_mean"])) if e is not None else (lambda k: "")
        lines.append(f"| {s} | {m}{' **(selected)**' if r['selected'] else ''} | {f(r.roc_auc_mean)} "
                     f"({f(r.roc_auc_sd)}) | {f(r.pr_auc_mean)} | {f(r.balanced_acc_mean)} | {f(r.f1_mean)} | "
                     f"{g('roc_auc')} | {g('pr_auc')} | {g('balanced_acc')} | {g('f1')} | {g('brier')} |")
    path = cfg.out_dir("models", task, mode) / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(tasks=cfg.TASKS, repeats=cfg.N_REPEATS, tune=False, set_names=None, model_names=None, select=False):
    plots.setup()
    merge, fs = co.load_adnimerge(), co.load_fs7()
    set_names = set_names or (TUNE_SETS if tune else ALL_SETS)
    model_names = model_names or list(models())
    for task in tasks:
        run_task(task, merge, fs, repeats, tune, set_names, model_names, select)
