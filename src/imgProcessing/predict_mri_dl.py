"""Predict with a trained DEEP MRI model (run folder from mri_dl_trainer.py).

Fixed vs. ae3bce0: shared preprocessing (identical to training), class names and input size
read from the run's meta.json, no lambda transforms, absolute default paths, unreadable
files skipped with a warning.

Usage:
    python src/imgProcessing/predict_mri_dl.py --image x.jpg [--run final_resnet50_s42]
    python src/imgProcessing/predict_mri_dl.py --image-dir some/folder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402
from mri_dl_trainer import build_model  # noqa: E402
from mri_preprocess import PreprocessSpec, load_preprocessed, to_tensor  # noqa: E402


def load_run(run_dir: Path, device):
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    model = build_model(meta["arch"], len(meta["class_names"]), meta["config"]["dropout"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / meta.get("weights_file", "best.pt"), map_location=device))
    return model.to(device).eval(), meta


@torch.no_grad()
def predict_paths(paths: list[Path], model, meta: dict, device) -> pd.DataFrame:
    spec = PreprocessSpec.from_dict(meta["preprocess"])
    names, size = meta["class_names"], meta["image_size"]
    rows = []
    for p in paths:
        try:
            img = load_preprocessed(p, spec)
        except Exception as err:
            print(f"[WARN] skipped unreadable image {p}: {err}")
            continue
        x = to_tensor(img).unsqueeze(0).to(device)
        if size != img.shape[0]:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        prob = torch.softmax(model(x).float(), 1)[0].cpu().numpy()
        k = int(np.argmax(prob))
        rows.append({"image_path": str(p), "predicted_class": names[k], "confidence": float(prob[k]),
                     "probabilities_json": json.dumps({n: float(v) for n, v in zip(names, prob)})})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", type=Path)
    ap.add_argument("--image-dir", type=Path)
    ap.add_argument("--run", default=None, help="run folder name under output/models/dl_runs (default: the served model)")
    ap.add_argument("--output-csv", type=Path, default=C.OUTPUT_DIR / "predictions" / "mri_dl_predictions.csv")
    args = ap.parse_args()
    if not args.image and not args.image_dir:
        sys.exit("Provide --image or --image-dir")
    run_dir = C.MODELS_DIR / "dl_runs" / args.run if args.run else C.SERVED_MODEL_DIR
    if not (run_dir / "meta.json").exists():
        sys.exit(f"No trained deep model in {run_dir}. Train first: python src/imgProcessing/mri_dl_trainer.py")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_run(run_dir, dev)
    paths = ([args.image] if args.image else []) + (list(C.iter_images(args.image_dir)) if args.image_dir else [])
    out = predict_paths(paths, model, meta, dev)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.head(10).to_string(index=False))
    print(f"{len(out)} predictions -> {args.output_csv}")


if __name__ == "__main__":
    main()
