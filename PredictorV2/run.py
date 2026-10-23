"""ADNI Predictor: one entry point for every stage.

    python PredictorV2/run.py analysis                 # 1. data analysis: Table 1, statistics, figures
    python PredictorV2/run.py models                   # 2. model comparison, fixed hyperparameters
    python PredictorV2/run.py models --tune            # 2. model comparison, nested-CV tuning
    python PredictorV2/run.py models --select          # 2. with leakage-free ANOVA-FDR feature selection
    python PredictorV2/run.py explain                  # 3. SHAP, LIME, permutation importance
    python PredictorV2/run.py all                      # 1 + 2 + 3 (fixed hyperparameters)

Options: --tasks CN_AD sMCI_pMCI, --repeats N (CV repeats), --feature-sets ..., --models ...
Results go to PredictorV2/output/ (git-ignored), one numbered folder per stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from common import config as cfg  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["analysis", "models", "explain", "all"])
    ap.add_argument("--tasks", nargs="+", default=cfg.TASKS, choices=cfg.TASKS)
    ap.add_argument("--repeats", type=int, default=None,
                    help=f"CV repeats (default {cfg.N_REPEATS}; 3 with --tune)")
    ap.add_argument("--tune", action="store_true", help="nested-CV hyperparameter tuning (slow)")
    ap.add_argument("--select", action="store_true",
                    help="ANOVA F-test + FDR feature selection inside every training fold (no leakage)")
    ap.add_argument("--feature-sets", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args(argv)
    repeats = args.repeats or (3 if args.tune else cfg.N_REPEATS)
    mode = ("tuned" if args.tune else "fixed") + "_hyperparameters" + ("_anova_selection" if args.select else "")

    if args.stage in ("analysis", "all"):
        from data_analysis import eda, statistics
        eda.main(args.tasks)
        statistics.main(args.tasks)
    if args.stage in ("models", "all"):
        from training import run_models
        run_models.main(args.tasks, repeats, args.tune, args.feature_sets, args.models, args.select)
    if args.stage in ("explain", "all"):
        from explainability import explain
        kwargs = {"set_names": args.feature_sets} if args.feature_sets else {}
        explain.main(args.tasks, mode, **kwargs)
    write_output_index()
    print(f"\nResults in {cfg.OUTPUT_DIR}")


def headline_results() -> list[str]:
    """The CV-selected model per task, mode and feature set (from every selected_models.csv found)."""
    import pandas as pd
    rows = []
    for path in sorted((cfg.OUTPUT_DIR / cfg.STAGES["models"]).glob("*/*/tables/selected_models.csv")):
        task, mode = path.parts[-4], path.parts[-3]
        for r in pd.read_csv(path).itertuples():
            ext = "" if pd.isna(getattr(r, "ext_roc_auc", float("nan"))) else f"{r.ext_roc_auc:.3f}"
            ext_ba = "" if pd.isna(getattr(r, "ext_balanced_acc", float("nan"))) else f"{r.ext_balanced_acc:.3f}"
            rows.append(f"| {task} | {mode} | {r.features} | {r.model} | {r.cv_roc_auc:.3f} ± {r.cv_roc_auc_sd:.3f} | "
                        f"{r.cv_balanced_acc:.3f} | {ext} | {ext_ba} |")
    if not rows:
        return []
    return ["## Headline results (model selected by CV; external = ADNIGO/2/3, never used for selection)", "",
            "'cog' feature sets are circular for CN_AD (ceiling, not a finding).", "",
            "| task | mode | features | model | CV ROC-AUC | CV bal. acc. | ext ROC-AUC | ext bal. acc. |",
            "|---|---|---|---|---|---|---|---|", *rows, ""]


def write_output_index():
    """output/README.md: what every folder holds, plus the list of files present."""
    lines = ["# Results (ADNI-derived: do not upload or share outside approved ADNI users)", "",
             "| Folder | Contents |", "|---|---|",
             "| `1_data_analysis/<task>/tables` | Table 1, group comparison (tests, effect sizes, FDR), missingness |",
             "| `1_data_analysis/<task>/figures` | counts, age confound, markers, engineered markers, effect sizes, "
             "correlations |",
             "| `1_data_analysis/all_participants` | ANOVA / Kruskal-Wallis across CN, EMCI, LMCI, AD |",
             "| `2_model_comparison/<task>/<mode>/summary.md` | main results table (start here) |",
             "| `2_model_comparison/<task>/<mode>/tables` | all metrics, selected models, per-repeat results |",
             "| `2_model_comparison/<task>/<mode>/figures` | model comparison, AUC heatmap, ROC, PR, calibration, "
             "confusion matrices |",
             "| `2_model_comparison/<task>/<mode>/hyperparameters` | final params, best params per outer fold (tuned) |",
             "| `3_explainability/<task>/<mode>/<feature set>` | SHAP, LIME, permutation importance |", ""]
    lines += headline_results() + ["## Files", ""]
    for p in sorted(cfg.OUTPUT_DIR.rglob("*")):
        if p.is_file() and p.suffix != ".pdf" and p.name != "README.md":
            lines.append(f"- `{p.relative_to(cfg.OUTPUT_DIR).as_posix()}`")
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
