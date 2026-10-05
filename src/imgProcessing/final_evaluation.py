"""The ONE evaluation on the held-out TEST splits. Run once, after all training is finished.

Evaluates every pre-registered model:
* leakage_<setting>.joblib  (fixed simple model under naive / image-group / subject-group splits,
  plus the shuffled-label sanity checks) and a majority-class baseline per split;
* classical_split_subject_<model>.joblib (6 tuned classical models);
* deep runs named final_<arch>_s<seed> (several seeds -> mean +/- std) and naive_<arch>_s<seed>.

Writes output/reports/final_results.json and final_results.csv, and a lock file
output/reports/test_used.json. A second run is refused unless --force (and is then logged).

Usage: python src/imgProcessing/final_evaluation.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402
from mri_data import compute_metrics, load_split, pixel_features, save_json  # noqa: E402


def subject_level(y, prob, groups) -> dict:
    """Patient-level view: average the slice probabilities of each pseudo-subject."""
    d = pd.DataFrame(prob).assign(g=groups, y=y)
    agg = d.groupby("g").mean(numeric_only=True)
    yy = agg.pop("y").round().astype(int).to_numpy()
    m = compute_metrics(yy, agg.to_numpy(), n_boot=0)
    return {k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "n", "confusion_matrix")}


def eval_classical(path: Path, cache: dict) -> tuple[dict, np.ndarray, pd.DataFrame]:
    from mri_model_trainer import load_artifact
    art = load_artifact(path)
    split = art.get("split") or Path(art["split_file"]).stem
    if split not in cache:
        cache[split] = load_split(split)
    df = cache[split]
    te = df[df["split"] == "test"].copy()
    if art.get("labels_override") is not None:
        te["y"] = te["pixel_md5"].map(dict(zip(art["labels_override"]["pixel_md5"], art["labels_override"]["y"]))).to_numpy()
    from mri_model_trainer import predict_proba
    prob = predict_proba(art["pipeline"], pixel_features(te, art.get("feature_size", 64)))
    return art, prob, te


def eval_deep(run_dir: Path, cache: dict) -> tuple[dict, np.ndarray, pd.DataFrame]:
    import torch

    from mri_dl_trainer import GpuBatcher, build_model, evaluate
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    split = meta["config"]["split"]
    if split not in cache:
        cache[split] = load_split(split)
    te = cache[split][cache[split]["split"] == "test"].copy()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(meta["arch"], len(C.CLASS_NAMES), meta["config"]["dropout"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=dev))
    model.to(dev)
    b = GpuBatcher(te["cache_row"].to_numpy().astype(np.int64), te["y"].to_numpy().astype(np.int64), dev, meta["image_size"])
    _, prob = evaluate(model, b, np.arange(len(te)), 128, dev)
    return meta, prob, te


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    lock = C.REPORTS_DIR / "test_used.json"
    if lock.exists() and not args.force:
        sys.exit(f"Test set already used ({lock}). Refusing to evaluate again; pass --force to override (it is logged).")

    cache: dict = {}
    results, rows = {}, []

    def record(name: str, family: str, split: str, y, prob, groups, extra=None):
        m = compute_metrics(y, prob, groups, n_boot=1000)
        if split != "split_naive":
            m["subject_level"] = subject_level(y, prob, groups)
        m.update(extra or {})
        results[name] = {"family": family, "split": split, **m}
        rows.append({"name": name, "family": family, "split": split, "n": m["n"], "accuracy": m["accuracy"],
                     "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"],
                     "roc_auc_ovr_macro": m.get("roc_auc_ovr_macro"),
                     "macro_f1_ci95": m.get("bootstrap_ci95_by_subject", {}).get("macro_f1")})
        print(f"[test] {name:40s} acc {m['accuracy']:.3f} bal-acc {m['balanced_accuracy']:.3f} macro-F1 {m['macro_f1']:.3f}", flush=True)

    for p in sorted(C.MODELS_DIR.glob("leakage_*.joblib")):
        art, prob, te = eval_classical(p, cache)
        record(f"leakage/{art['setting']}", "leakage_simple", art["split"], te["y"].to_numpy(), prob, te["subject_group"].to_numpy())
        if not art["shuffled"]:
            maj = np.zeros_like(prob); maj[:, art["majority_class"]] = 1.0
            record(f"baseline/majority_{art['split']}", "baseline", art["split"], te["y"].to_numpy(), maj, te["subject_group"].to_numpy())

    for p in sorted(C.MODELS_DIR.glob("classical_split_subject_*.joblib")):
        art, prob, te = eval_classical(p, cache)
        record(f"classical/{art['model_name']}", "classical", "split_subject", te["y"].to_numpy(), prob, te["subject_group"].to_numpy(),
               {"params": art["params"]})

    runs = sorted(d for d in (C.MODELS_DIR / "dl_runs").glob("*") if (d / "meta.json").exists() and d.name.startswith(("final_", "naive_")))
    for d in runs:
        meta, prob, te = eval_deep(d, cache)
        record(f"deep/{d.name}", "deep", meta["config"]["split"], te["y"].to_numpy(), prob, te["subject_group"].to_numpy(),
               {"arch": meta["arch"], "seed": meta["config"]["seed"], "best_epoch": meta["best"]["epoch"]})
        np.save(C.REPORTS_DIR / f"test_prob_{d.name}.npy", prob)

    # mean +/- std over seeds per deep architecture (honest split only)
    seeds = {}
    for name, r in results.items():
        if r["family"] == "deep" and name.startswith("deep/final_"):
            seeds.setdefault(r["arch"], []).append(r)
    summary = {arch: {k: {"mean": float(np.mean([x[k] for x in rs])), "std": float(np.std([x[k] for x in rs], ddof=1)) if len(rs) > 1 else 0.0,
                          "n_seeds": len(rs)} for k in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr_macro")}
               for arch, rs in seeds.items()}
    out = {"evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results, "deep_seed_summary": summary}
    save_json(out, C.REPORTS_DIR / "final_results.json")
    pd.DataFrame(rows).to_csv(C.REPORTS_DIR / "final_results.csv", index=False)
    prev = json.loads(lock.read_text(encoding="utf-8")) if lock.exists() else {"runs": []}
    prev["runs"].append({"at": out["evaluated_at"], "forced": args.force, "models": list(results)})
    save_json(prev, lock)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
