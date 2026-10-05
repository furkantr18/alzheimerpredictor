"""Copy a trained MRI model into output/models/served/ for the API (/predict-mri).

Writes served/served.json with class names, preprocessing spec, library versions, the split
file, and the honest metrics (test metrics from final_results.json when available).

Usage:
    python src/imgProcessing/serve_model.py --run final_resnet50_s42
    python src/imgProcessing/serve_model.py --classical output/models/classical_split_subject_SVM_RBF.joblib
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402

DATASET = {"name": "uraninjo/augmented-alzheimer-mri-dataset-v2 (Kaggle), folder datasets/archive (1)",
           "note": "6,400 original 2D axial slices of 200 pseudo-subjects (patient IDs reconstructed from file names) "
                   "+ 33,984 augmented copies; train only uses copies of train subjects."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--classical", type=Path)
    args = ap.parse_args()

    dst = C.SERVED_MODEL_DIR
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    final = C.REPORTS_DIR / "final_results.json"
    test = json.loads(final.read_text(encoding="utf-8"))["results"] if final.exists() else {}

    if args.run:
        src = C.MODELS_DIR / "dl_runs" / args.run
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        shutil.copy2(src / "best.pt", dst / "best.pt")
        t = test.get(f"deep/{args.run}")
        served = {"kind": "deep", "model_name": f"{meta['arch']} ({args.run})", "arch": meta["arch"], "config": meta["config"],
                  "image_size": meta["image_size"], "weights_file": "best.pt"}
        metrics_val = meta["val_metrics"]
    else:
        from mri_model_trainer import load_artifact
        art = load_artifact(args.classical)
        shutil.copy2(args.classical, dst / "model.joblib")
        name = art["model_name"]
        t = test.get(f"classical/{name}")
        meta = art
        served = {"kind": "classical", "model_name": f"classical {name}", "weights_file": "model.joblib"}
        metrics_val = None
    served.update({
        "version": time.strftime("%Y%m%d-%H%M"), "class_names": meta["class_names"], "preprocess": meta["preprocess"],
        "split_file": meta["split_file"], "versions": meta["versions"], "dataset": DATASET,
        "evaluation": "pseudo-subject-grouped split; test = original images of unseen subjects, evaluated once",
        "metrics": {"test": {k: t[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr_macro", "bootstrap_ci95_by_subject") if k in t} if t else None,
                    "validation": {k: metrics_val[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")} if metrics_val else None}})
    (dst / "served.json").write_text(json.dumps(served, indent=2, default=str), encoding="utf-8")
    shutil.copy2(C.SPLITS_DIR / meta["split_file"], dst / meta["split_file"])
    if args.run:
        shutil.copy2(src / "meta.json", dst / "meta.json")
    print(f"served -> {dst}\n{json.dumps(served['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
