"""Run inference with the trained MRI dementia classifier.

Usage:
    python src/imgProcessing/predict_mri.py --image src/data/img_processed/NonDemented/example.jpg
    python src/imgProcessing/predict_mri.py --image-dir src/data/img_processed/NonDemented
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in VALID_EXTENSIONS else []
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]


def _load_vector(image_path: Path, image_size: int) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0).reshape(1, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict dementia class from MRI image(s)")
    parser.add_argument("--image", type=Path, default=None, help="Single image path")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory with MRI images")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("src/imgProcessing/output/models/best_mri_model.pkl"),
        help="Path to saved model artifact",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("src/imgProcessing/output/mri_predictions.csv"),
        help="Where batch predictions will be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image is None and args.image_dir is None:
        raise ValueError("Provide --image or --image-dir")

    artifact = joblib.load(args.model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    pca = artifact["pca"]
    label_encoder = artifact["label_encoder"]
    image_size = int(artifact["image_size"])

    paths: list[Path] = []
    if args.image is not None:
        paths.extend(_iter_images(args.image))
    if args.image_dir is not None:
        paths.extend(_iter_images(args.image_dir))

    if not paths:
        raise FileNotFoundError("No valid image files found for prediction")

    rows = []
    for p in sorted(set(paths)):
        vec = _load_vector(p, image_size)
        vec_scaled = scaler.transform(vec)
        vec_pca = pca.transform(vec_scaled)

        pred_idx = int(model.predict(vec_pca)[0])
        pred_label = str(label_encoder.inverse_transform([pred_idx])[0])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec_pca)[0]
            confidence = float(np.max(probs))
            prob_map = {
                str(label_encoder.inverse_transform([i])[0]): float(prob)
                for i, prob in enumerate(probs)
            }
        else:
            confidence = None
            prob_map = {}

        rows.append(
            {
                "image_path": str(p),
                "predicted_class": pred_label,
                "confidence": confidence,
                "probabilities_json": json.dumps(prob_map),
            }
        )

    out_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"Predictions complete: {len(out_df)} images")
    print(f"Output CSV: {args.output_csv}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
