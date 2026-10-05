"""Predict with a trained CLASSICAL MRI model (joblib artifact from mri_model_trainer.py).

Fixed vs. ae3bce0: uses the shared preprocessing (identical to training), the class names
saved WITH the model (no label-order mismatch), absolute default paths, and skips
unreadable files with a warning instead of crashing.

Usage:
    python src/imgProcessing/predict_mri.py --image x.jpg [--model-path ...joblib]
    python src/imgProcessing/predict_mri.py --image-dir some/folder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402
from mri_model_trainer import predict_proba  # noqa: E402
from mri_preprocess import PreprocessSpec, load_preprocessed  # noqa: E402


def predict_paths(paths: list[Path], artifact: dict) -> pd.DataFrame:
    spec = PreprocessSpec.from_dict(artifact["preprocess"])
    fs, names = artifact["feature_size"], artifact["class_names"]
    rows = []
    for p in paths:
        try:
            img = load_preprocessed(p, spec)
        except Exception as err:
            print(f"[WARN] skipped unreadable image {p}: {err}")
            continue
        feat = (cv2.resize(img, (fs, fs), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0).reshape(1, -1)
        prob = predict_proba(artifact["pipeline"], feat)[0]
        k = int(np.argmax(prob))
        rows.append({"image_path": str(p), "predicted_class": names[k], "confidence": float(prob[k]),
                     "probabilities_json": json.dumps({n: float(v) for n, v in zip(names, prob)})})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", type=Path)
    ap.add_argument("--image-dir", type=Path)
    ap.add_argument("--model-path", type=Path, default=C.MODELS_DIR / "classical_split_subject_LogisticRegression.joblib")
    ap.add_argument("--output-csv", type=Path, default=C.OUTPUT_DIR / "predictions" / "mri_predictions.csv")
    args = ap.parse_args()
    if not args.image and not args.image_dir:
        sys.exit("Provide --image or --image-dir")
    if not args.model_path.exists():
        sys.exit(f"Model not found: {args.model_path}. Train first: python src/imgProcessing/mri_model_trainer.py")
    paths = ([args.image] if args.image else []) + (list(C.iter_images(args.image_dir)) if args.image_dir else [])
    from mri_model_trainer import load_artifact
    out = predict_paths(paths, load_artifact(args.model_path))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.head(10).to_string(index=False))
    print(f"{len(out)} predictions -> {args.output_csv}")


if __name__ == "__main__":
    main()
