"""Run inference with a trained deep learning MRI classifier.

Usage:
    python src/imgProcessing/predict_mri_dl.py --image path/to/image.jpg
    python src/imgProcessing/predict_mri_dl.py --image-dir src/data/img_processed/NonDemented
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in VALID_EXTENSIONS else []
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]


def _build_model(architecture: str, num_classes: int) -> nn.Module:
    arch = architecture.lower()

    if arch == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError("Unsupported architecture in metadata")


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _load_tensor(path: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(path) as img:
        gray = img.convert("L")
    return transform(gray).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict dementia class from MRI images using DL model")
    parser.add_argument("--image", type=Path, default=None, help="Single image path")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory with MRI images")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("src/imgProcessing/output/models/best_mri_dl_model.pt"),
        help="Path to model state dict checkpoint",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("src/imgProcessing/output/models/best_mri_dl_model_metadata.json"),
        help="Path to checkpoint metadata JSON",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("src/imgProcessing/output/mri_dl_predictions.csv"),
        help="Where predictions are saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image is None and args.image_dir is None:
        raise ValueError("Provide --image or --image-dir")

    if not args.metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    class_names: list[str] = metadata["class_names"]
    architecture = str(metadata["architecture"])
    image_size = int(metadata["image_size"])

    model = _build_model(architecture=architecture, num_classes=len(class_names))
    state = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    transform = _build_transform(image_size=image_size)

    paths: list[Path] = []
    if args.image is not None:
        paths.extend(_iter_images(args.image))
    if args.image_dir is not None:
        paths.extend(_iter_images(args.image_dir))

    if not paths:
        raise FileNotFoundError("No valid image files found for prediction")

    rows = []
    with torch.no_grad():
        for p in sorted(set(paths)):
            tensor = _load_tensor(p, transform)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx])
            prob_map = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

            rows.append(
                {
                    "image_path": str(p),
                    "predicted_class": pred_class,
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
