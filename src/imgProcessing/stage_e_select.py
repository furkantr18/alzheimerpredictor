"""Stage E + headline: inner-CV auto-selection and top-3 ensembles, then the comparison table.

AutoSelect: in every outer fold, the candidate (Stage A/B configuration, non-exploratory) with the
highest INNER score is chosen and its outer predictions are used -> an honest estimate that includes
the cost of model selection. Ensemble3: probability average of the top-3 candidates by inner score
with three different backbones (fold-wise, inner scores only). The fine-tuned baseline is the
comparator and is not a candidate.

Usage: python src/imgProcessing/stage_e_select.py --task T3 [--families A,B] [--report]
Output: output/cv/oof/E_*.npz, output/cv/results/E_*.json, output/cv/results/comparison_<task>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_core as cv  # noqa: E402


def backbone_of(method: str) -> str:
    for b in ("dinov2_vitb14", "dinov2_vits14", "efficientnet_b0", "efficientnet_b3", "convnext_tiny", "resnet50", "biomedclip", "handcrafted"):
        if b in method:
            return b + ("_25d" if "_25d" in method else "")
    return method


def candidates(task: str, families: list[str]) -> pd.DataFrame:
    reg = pd.read_csv(cv.REGISTRY)
    reg = reg[(reg["task"] == task) & (reg["stage"].isin(families)) & (reg["exploratory"].astype(str) != "True")]
    reg = reg[~reg["method"].str.contains("_pos")]
    reg = reg.drop_duplicates(["method", "repeat", "outer_fold"], keep="last")
    have = {p.name.split("__")[0] for p in (cv.CV_DIR / "oof").glob(f"*__{task}.npz")}
    return reg[reg["method"].isin(have)]


def fold_predictions(method: str, task: str, fold_df: pd.DataFrame) -> dict:
    """(repeat, fold) -> (ids, y, P) for one method, using the fold file for membership."""
    oof = cv.load_oof(method, task)
    out = {}
    for r, (ids, y, P) in oof.items():
        fr = fold_df[fold_df["repeat"] == r].set_index("group")["outer_fold"]
        k_of = fr.loc[ids].to_numpy()
        for k in np.unique(k_of):
            m = k_of == k
            out[(r, int(k))] = (ids[m], y[m], P[m])
    return out


def run(task: str, families: list[str]) -> dict:
    fold_df = cv.folds()
    cand = candidates(task, families)
    methods = sorted(cand["method"].unique())
    preds = {m: fold_predictions(m, task, fold_df) for m in methods}
    nc = len(cv.TASKS[task]["names"])
    auto, ens = {}, {}
    picks = []
    for (r, k), part in cand.groupby(["repeat", "outer_fold"]):
        part = part.sort_values("inner_score", ascending=False)
        best = part.iloc[0]["method"]
        ids, y, P = preds[best][(r, k)]
        auto.setdefault(r, [[], [], []]); [auto[r][i].append(v) for i, v in enumerate((ids, y, P))]
        chosen, seen = [], set()
        for m in part["method"]:
            b = backbone_of(m)
            if b not in seen:
                chosen.append(m); seen.add(b)
            if len(chosen) == 3:
                break
        base_ids = preds[chosen[0]][(r, k)][0]
        Ps = []
        for m in chosen:
            i2, _, P2 = preds[m][(r, k)]
            Ps.append(pd.DataFrame(P2, index=i2).loc[base_ids].to_numpy())
        ens.setdefault(r, [[], [], []]); [ens[r][i].append(v) for i, v in enumerate((base_ids, preds[chosen[0]][(r, k)][1], np.mean(Ps, 0)))]
        picks.append({"repeat": int(r), "fold": int(k), "auto": best, "ensemble": chosen})
        cv.log_run(stage="E", method="E_autoselect_" + "".join(families), config={"picked": best}, task=task, repeat=r, outer_fold=k,
                   inner_score=float(part.iloc[0]["inner_score"]), exploratory=False, note="selection by inner score")
    res = {}
    for name, d in (("E_autoselect_" + "".join(families), auto), ("E_ensemble3_" + "".join(families), ens)):
        oof = {r: (np.concatenate(v[0]), np.concatenate(v[1]), np.vstack(v[2])) for r, v in d.items()}
        cv.save_oof(name, task, oof)
        s = cv.summarize(oof, nc)
        s["picks"] = picks
        (cv.CV_DIR / "results" / f"{name}__{task}.json").write_text(json.dumps(s, indent=2))
        res[name] = s
        print(f"[E] {name} {task}: macro-F1 {s['macro_f1']['mean']:.3f} {np.round(s['macro_f1']['ci95'], 3)} bal-acc {s['balanced_acc']['mean']:.3f} "
              f"QWK {s['qwk']['mean']:.3f} AUC {s['roc_auc']['mean']:.3f}", flush=True)
    counts = pd.Series([p["auto"] for p in picks]).value_counts().to_dict()
    print("[E] auto-selected methods (count over 15 folds):", counts)
    return res


def comparison(task: str, baseline: str = "B_ft_resnet18") -> None:
    """Every method on this task vs the baseline: summary + paired difference (macro-F1, balanced acc)."""
    nc = len(cv.TASKS[task]["names"])
    base = cv.load_oof(baseline, task)
    rows = []
    for p in sorted((cv.CV_DIR / "oof").glob(f"*__{task}.npz")):
        m = p.name.split("__")[0]
        oof = cv.load_oof(m, task)
        rf = cv.CV_DIR / "results" / f"{m}__{task}.json"
        s = json.loads(rf.read_text()) if rf.exists() else cv.summarize(oof, nc)
        row = {"method": m, **{f"{k}": s[k]["mean"] for k in cv.METRICS}, **{f"{k}_ci": s[k]["ci95"] for k in ("macro_f1", "balanced_acc", "qwk")},
               "slice_macro_f1": s.get("slice_level", {}).get("macro_f1")}
        if m != baseline:
            for met in ("macro_f1", "balanced_acc"):
                d = cv.paired(oof, base, nc, metric=met, n_boot=2000)
                row[f"diff_{met}"] = d["diff"]; row[f"diff_{met}_ci"] = d["ci95"]; row[f"diff_{met}_distinguishable"] = d["distinguishable"]
        rows.append(row)
    out = cv.CV_DIR / "results" / f"comparison_{task}.json"
    out.write_text(json.dumps(rows, indent=2))
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    cols = ["method", "macro_f1", "balanced_acc", "qwk", "roc_auc", "slice_macro_f1", "diff_macro_f1", "diff_macro_f1_distinguishable"]
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="T3")
    ap.add_argument("--families", default="A,B")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        run(a.task, a.families.split(","))
    comparison(a.task)


if __name__ == "__main__":
    main()
