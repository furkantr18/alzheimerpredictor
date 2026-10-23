"""ADNI model comparison: CN vs AD and sMCI vs pMCI from baseline data.

    python adni_predictor/src/run_models.py                   # all tasks, 5-fold x 10 repeats
    python adni_predictor/src/run_models.py --repeats 2       # quick run

Experiments (one row per participant, so every split is subject-level):
    cv_ADNI1     repeated stratified CV inside ADNI1 (one scanner protocol, 1.5T)
    ext_ADNI2-3  train on all of ADNI1, test once on participants enrolled in ADNIGO/2/3
                 (3T, other sites and years). The ADNIMERGE volumes are skipped here because their
                 FreeSurfer version changes between phases.
Model selection uses the CV results only: per feature set, the model with the highest mean CV
ROC-AUC is marked 'selected'. The external test is reported for every model but must not be used
to pick one.

Outputs (git-ignored) in adni_predictor/output/models/<task>/:
    results.csv, results_per_repeat.csv, comparison_auc.csv
    fig_model_comparison, fig_roc, fig_importance (PNG + PDF)
and adni_predictor/output/summary.md.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import adni_config as cfg  # noqa: E402
import cohort as co  # noqa: E402
import evaluate as ev  # noqa: E402
import features as ft  # noqa: E402
import plots  # noqa: E402
from plots import plt  # noqa: E402

TASKS = ["CN_AD", "sMCI_pMCI"]
TRAIN_PHASE = "ADNI1"
TEST_PHASES = ["ADNIGO", "ADNI2", "ADNI3"]
ROC_SETS = ["demo", "demo+mri_fs7", "demo+cog+mri_fs7"]      # at most three lines per panel
IMPORTANCE_SET = "demo+mri_fs7"


def run_task(task, merge, fs, repeats, names, out_dir: Path):
    c, matched, n_all = co.modeling_cohort(merge, fs, task)
    print(f"\n[{task}] {n_all} participants, {len(c)} with a FreeSurfer 7 baseline scan")
    sets = ft.feature_sets(c, matched)
    y = c["label"]
    train = (c["phase"] == TRAIN_PHASE).values
    test = c["phase"].isin(TEST_PHASES).values

    rows, reps, oof, ext_prob, fitted = [], [], {}, {}, {}
    for set_name, X in sets.items():
        for model_name, model in ev.models().items():
            key = (set_name, model_name)
            pr, o = ev.repeated_cv(model, X[train], y[train], n_repeats=repeats)
            oof[key] = o
            rows.append({"experiment": f"cv_{TRAIN_PHASE}", "features": set_name, "model": model_name,
                         "n": int(train.sum()), "n_pos": int(y[train].sum()), **ev.summarize(pr)})
            reps.append(pr.assign(features=set_name, model=model_name, repeat=range(len(pr))))
            line = f"  {set_name:20s} {model_name:10s} CV AUC {pr['roc_auc'].mean():.3f}"
            if "mri_merge" not in set_name:
                m, p, f = ev.external(model, X[train], y[train], X[test], y[test])
                ext_prob[key], fitted[key] = p, f
                rows.append({"experiment": "ext_ADNI2-3", "features": set_name, "model": model_name,
                             "n": int(test.sum()), "n_pos": int(y[test].sum()),
                             **{f"{k}_mean": v for k, v in m.items()}})
                line += f"   external AUC {m['roc_auc']:.3f}"
            print(line)

    res = pd.DataFrame(rows)
    cv = res[res.experiment.str.startswith("cv_")]
    best = cv.loc[cv.groupby("features")["roc_auc_mean"].idxmax(), ["features", "model"]]
    res["selected"] = res.set_index(["features", "model"]).index.isin(best.set_index(["features", "model"]).index)

    task_dir = out_dir / "models" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(task_dir / "results.csv", index=False)
    pd.concat(reps).to_csv(task_dir / "results_per_repeat.csv", index=False)
    comparison_table(res).to_csv(task_dir / "comparison_auc.csv")

    selected = dict(best.values)
    fig_model_comparison(res, task, task_dir)
    fig_roc(task, y, train, test, oof, ext_prob, selected, task_dir)
    if IMPORTANCE_SET in selected:
        key = (IMPORTANCE_SET, selected[IMPORTANCE_SET])
        fig_importance(fitted[key], sets[IMPORTANCE_SET][test], y[test], key, names, task, task_dir)
    return c, res


def comparison_table(res: pd.DataFrame) -> pd.DataFrame:
    """Model x feature set, 'CV mean ± SD | external' ROC-AUC."""
    cv = res[res.experiment.str.startswith("cv_")].set_index(["model", "features"])
    ext = res[res.experiment == "ext_ADNI2-3"].set_index(["model", "features"])["roc_auc_mean"]
    cell = cv.apply(lambda r: f"{r.roc_auc_mean:.3f} ± {r.roc_auc_sd:.3f}", axis=1)
    cell = cell + ext.reindex(cell.index).map(lambda v: "" if pd.isna(v) else f" | {v:.3f}")
    return cell.unstack("features")


def fig_model_comparison(res, task, out):
    cv = res[res.experiment.str.startswith("cv_")]
    ext = res[res.experiment == "ext_ADNI2-3"].set_index(["features", "model"])["roc_auc_mean"]
    set_names = list(dict.fromkeys(cv["features"]))
    model_names = list(dict.fromkeys(cv["model"]))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, set_names):
        d = cv[cv.features == s].set_index("model").reindex(model_names)
        yy = np.arange(len(model_names))
        ax.errorbar(d["roc_auc_mean"], yy, xerr=d["roc_auc_sd"], fmt="o", ms=5, color=plots.SERIES[0],
                    ecolor=plots.SERIES[0], elinewidth=1.5, capsize=0, label="CV in ADNI1 (mean ± SD)")
        e = [ext.get((s, m), np.nan) for m in model_names]
        if not np.all(np.isnan(e)):
            ax.plot(e, yy, "D", ms=5, color=plots.SERIES[1], label="External ADNIGO/2/3")
        sel = d.index[d["selected"]].tolist()
        ax.set_yticks(yy, [f"{m} *" if m in sel else m for m in model_names])
        ax.set_title(s, fontsize=9)
        ax.set_xlim(0.45, 1.01)
        ax.axvline(0.5, color=plots.TEXT_2, lw=0.8, ls=":")
        ax.grid(axis="y", visible=False)
    for ax in axes.flat[len(set_names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("ROC-AUC")
    names = plots.LABEL_NAMES[task]
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{names[0]} vs {names[1]}: model comparison (* = selected by CV)", fontsize=11,
                 fontweight="bold", color=plots.TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plots.save(fig, out / "fig_model_comparison")


def fig_roc(task, y, train, test, oof, ext_prob, selected, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    panels = [(axes[0], "CV in ADNI1 (out-of-fold, repeat 1)", y[train].values, lambda k: oof[k][0]),
              (axes[1], "External test ADNIGO/2/3", y[test].values, lambda k: ext_prob.get(k))]
    for ax, title, yt, get in panels:
        for i, s in enumerate(ROC_SETS):
            if s not in selected or get((s, selected[s])) is None:
                continue
            p = get((s, selected[s]))
            fpr, tpr, _ = roc_curve(yt, p)
            auc = ev.metrics(yt, p)["roc_auc"]
            ax.plot(fpr, tpr, color=plots.SERIES[i], label=f"{s} ({selected[s]}), AUC {auc:.3f}")
        ax.plot([0, 1], [0, 1], color=plots.TEXT_2, lw=0.8, ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_aspect("equal")
    axes[0].set_ylabel("True positive rate")
    names = plots.LABEL_NAMES[task]
    fig.suptitle(f"{names[0]} vs {names[1]}: ROC of the CV-selected model per feature set", fontsize=11,
                 fontweight="bold", color=plots.TEXT)
    fig.tight_layout()
    plots.save(fig, out / "fig_roc")


def fig_importance(model, X, y, key, names, task, out, top=15):
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=20, random_state=cfg.SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": [names.get(c, c) for c in X.columns], "mean": r.importances_mean,
                        "sd": r.importances_std}).sort_values("mean", ascending=False)
    imp.to_csv(out / "importance.csv", index=False)
    imp = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp["feature"], imp["mean"], xerr=imp["sd"], color=plots.SERIES[0], height=0.6,
            error_kw={"ecolor": plots.TEXT_2, "elinewidth": 1})
    ax.set_xlabel("Drop in ROC-AUC when shuffled (external test)")
    ax.grid(axis="y", visible=False)
    names_ = plots.LABEL_NAMES[task]
    ax.set_title(f"{names_[0]} vs {names_[1]}: top features, {key[0]} ({key[1]})")
    plots.save(fig, out / "fig_importance")


def write_summary(results: dict, repeats: int, out_dir: Path):
    lines = ["# ADNI model comparison", "",
             f"5-fold stratified CV x {repeats} repeats inside {TRAIN_PHASE}; external test: train {TRAIN_PHASE}, "
             f"test {'/'.join(TEST_PHASES)}. CV values are mean (SD) over repeats. 'selected' = best mean CV "
             "ROC-AUC per feature set (the external test is not used for selection). 'cog' features are circular "
             "for CN_AD.", ""]
    for task, res in results.items():
        lines += [f"## {task}", "", "| features | model | CV ROC-AUC | CV bal. acc. | ext ROC-AUC | ext bal. acc. |",
                  "|---|---|---|---|---|---|"]
        cv = res[res.experiment.str.startswith("cv_")].set_index(["features", "model"])
        ext = res[res.experiment == "ext_ADNI2-3"].set_index(["features", "model"])
        for (s, m), r in cv.iterrows():
            e = ext.loc[(s, m)] if (s, m) in ext.index else None
            mark = " **(selected)**" if r["selected"] else ""
            lines.append(f"| {s} | {m}{mark} | {r.roc_auc_mean:.3f} ({r.roc_auc_sd:.3f}) | "
                         f"{r.balanced_acc_mean:.3f} ({r.balanced_acc_sd:.3f}) | "
                         f"{'' if e is None else f'{e.roc_auc_mean:.3f}'} | "
                         f"{'' if e is None else f'{e.balanced_acc_mean:.3f}'} |")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--repeats", type=int, default=cfg.N_REPEATS)
    args = ap.parse_args(argv)

    plots.setup()
    out_dir = cfg.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    merge, fs, names = co.load_adnimerge(), co.load_fs7(), ft.fs7_names()
    results = {task: run_task(task, merge, fs, args.repeats, names, out_dir)[1] for task in args.tasks}
    write_summary(results, args.repeats, out_dir)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
