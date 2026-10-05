"""MRI image preprocessing and augmentation pipeline.

IMPORTANT:
- Use this script only for NEW, UNPROCESSED MRI images.
- The current repository dataset under ``src/data/img_processed`` is already preprocessed
    (skull-stripped and upsampled/augmented from the Kaggle source).
- Prefer placing truly raw input in ``src/data/img_unprocessed`` and run this
    script against that folder.

Expected input structure for unprocessed images:

        src/data/img_unprocessed/
            MildDemented/
            ModerateDemented/
            NonDemented/
            VeryMildDemented/

Pipeline (default):
1. Grayscale load
2. Background/artifact reduction via largest connected brain-mask extraction
3. ROI crop using mask bounding box
4. Geometric standardization (resize with padding)
5. Noise reduction (Gaussian + Median + Bilateral + Wiener)
6. Contrast enhancement (CLAHE)
7. Intensity normalization (z-score then min-max)
8. Radiomic-style intensity discretization (binning)
9. Feature enhancement (unsharp masking + Sobel edge blend)
10. Optional data augmentation for class balancing and robustness

Outputs:
- Clean images: <output_dir>/clean/<class_name>/*.jpg
- Augmented images: <output_dir>/augmented/<class_name>/*.jpg
- Manifest CSV: <output_dir>/manifest.csv
- Metadata JSON: <output_dir>/preprocessing_metadata.json

Usage:
    python src/imgProcessing/preprocess_mri.py
    python src/imgProcessing/preprocess_mri.py --input-dir src/data/my_raw_mri
    python src/imgProcessing/preprocess_mri.py --augment-per-image 2 --target-size 224
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import wiener

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for MRI preprocessing. Install with: pip install opencv-python"
    ) from exc


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_ALREADY_PREPROCESSED_INPUT = Path("src/data/img_processed")


@dataclass
class PreprocessConfig:
    input_dir: Path
    output_dir: Path
    target_size: int = 224
    apply_rotation_correction: bool = False
    apply_hist_eq: bool = False
    apply_wiener: bool = True
    discretization_bins: int = 64
    edge_enhance_weight: float = 0.15
    unsharp_amount: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    augment_per_image: int = 1
    random_seed: int = 42
    max_images: int | None = None


def iter_image_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
            yield p


def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def minmax_normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = img.astype(np.float32)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx - mn < eps:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def zscore_then_minmax(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = img.astype(np.float32)
    mean = float(np.mean(img))
    std = float(np.std(img))
    if std < eps:
        return minmax_normalize(img)
    z = (img - mean) / std
    return minmax_normalize(z)


def extract_largest_component_mask(img_u8: np.ndarray) -> np.ndarray:
    # Otsu threshold gives a stable foreground mask for brain region extraction.
    _, mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.count_nonzero(mask) == 0:
        return np.ones_like(img_u8, dtype=np.uint8) * 255

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas) + 1)

    largest = np.zeros_like(mask, dtype=np.uint8)
    largest[labels == largest_idx] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    largest = cv2.morphologyEx(largest, cv2.MORPH_CLOSE, kernel, iterations=1)
    largest = cv2.morphologyEx(largest, cv2.MORPH_OPEN, kernel, iterations=1)
    return largest


def crop_to_mask(img_u8: np.ndarray, mask_u8: np.ndarray, margin: int = 8) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_u8, np.ones_like(img_u8, dtype=np.uint8) * 255

    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(img_u8.shape[1], int(xs.max()) + margin)
    y1 = min(img_u8.shape[0], int(ys.max()) + margin)

    return img_u8[y0:y1, x0:x1], mask_u8[y0:y1, x0:x1]


def rotation_correction_from_mask(img_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 20:
        return img_u8

    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)

    eigvals, eigvecs = np.linalg.eig(cov)
    major_axis = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))

    if abs(angle) < 2.0:
        return img_u8

    center = (img_u8.shape[1] / 2.0, img_u8.shape[0] / 2.0)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img_u8,
        rot,
        (img_u8.shape[1], img_u8.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated


def resize_with_letterbox(img_u8: np.ndarray, target_size: int) -> np.ndarray:
    h, w = img_u8.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    scale = min(target_size / h, target_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - nh) // 2
    x_off = (target_size - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def denoise_image(img_u8: np.ndarray, apply_wiener: bool) -> np.ndarray:
    x = cv2.GaussianBlur(img_u8, (3, 3), sigmaX=0.8)
    x = cv2.medianBlur(x, 3)
    x = cv2.bilateralFilter(x, d=5, sigmaColor=30, sigmaSpace=30)

    if apply_wiener:
        xw = wiener(x.astype(np.float32), (5, 5))
        x = _to_uint8(xw)

    return x


def enhance_contrast(img_u8: np.ndarray, clip_limit: float, tile_grid: int, apply_hist_eq: bool) -> np.ndarray:
    x = img_u8
    if apply_hist_eq:
        x = cv2.equalizeHist(x)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(x)


def intensity_discretization(img_01: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 1:
        return img_01

    img_01 = np.clip(img_01, 0.0, 1.0)
    # Radiomic-style fixed-width binning in normalized intensity domain.
    bin_ids = np.floor(img_01 * bins).astype(np.int32)
    bin_ids = np.clip(bin_ids, 0, bins - 1)
    return bin_ids.astype(np.float32) / float(bins - 1)


def unsharp_and_edge_enhance(img_01: np.ndarray, amount: float, edge_weight: float) -> np.ndarray:
    base_u8 = _to_uint8(img_01 * 255.0)
    blurred = cv2.GaussianBlur(base_u8, (0, 0), sigmaX=1.2)
    unsharp = cv2.addWeighted(base_u8, 1.0 + amount, blurred, -amount, 0)

    sobelx = cv2.Sobel(unsharp, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(unsharp, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    mag = minmax_normalize(mag)

    blended = minmax_normalize(unsharp.astype(np.float32) / 255.0 + edge_weight * mag)
    return blended


def preprocess_single_image(img_path: Path, cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {img_path}")

    mask = extract_largest_component_mask(img)
    img = cv2.bitwise_and(img, img, mask=mask)
    img, mask = crop_to_mask(img, mask, margin=8)

    if cfg.apply_rotation_correction:
        img = rotation_correction_from_mask(img, mask)

    img = resize_with_letterbox(img, cfg.target_size)
    img = denoise_image(img, apply_wiener=cfg.apply_wiener)
    img = enhance_contrast(
        img,
        clip_limit=cfg.clahe_clip_limit,
        tile_grid=cfg.clahe_tile_grid,
        apply_hist_eq=cfg.apply_hist_eq,
    )

    img_01 = zscore_then_minmax(img)
    img_01 = intensity_discretization(img_01, bins=cfg.discretization_bins)
    img_01 = unsharp_and_edge_enhance(
        img_01,
        amount=cfg.unsharp_amount,
        edge_weight=cfg.edge_enhance_weight,
    )

    final_u8 = _to_uint8(img_01 * 255.0)
    return final_u8, mask


def random_augment(img_u8: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img_u8.shape[:2]

    angle = rng.uniform(-15.0, 15.0)
    scale = rng.uniform(0.92, 1.08)
    shear_deg = rng.uniform(-8.0, 8.0)
    tx = rng.uniform(-0.03 * w, 0.03 * w)
    ty = rng.uniform(-0.03 * h, 0.03 * h)

    center = (w / 2.0, h / 2.0)
    rot = cv2.getRotationMatrix2D(center, angle, scale)
    rot[0, 2] += tx
    rot[1, 2] += ty

    shear = np.array(
        [[1.0, np.tan(np.radians(shear_deg)), 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    out = cv2.warpAffine(
        img_u8,
        rot,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    out = cv2.warpAffine(
        out,
        shear,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    alpha = rng.uniform(0.9, 1.2)
    beta = rng.uniform(-15, 15)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < 0.5:
        sigma = rng.uniform(2.0, 8.0)
        noise = np.random.normal(0.0, sigma, size=out.shape).astype(np.float32)
        out = _to_uint8(out.astype(np.float32) + noise)

    if rng.random() < 0.3:
        k = 3 if rng.random() < 0.5 else 5
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0.7)

    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_pipeline(cfg: PreprocessConfig) -> tuple[int, int]:
    rng = random.Random(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    image_paths = sorted(iter_image_files(cfg.input_dir))
    if cfg.max_images is not None and cfg.max_images > 0:
        image_paths = image_paths[: cfg.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {cfg.input_dir}")

    clean_root = cfg.output_dir / "clean"
    aug_root = cfg.output_dir / "augmented"
    ensure_dir(clean_root)
    ensure_dir(aug_root)

    rows = []
    clean_count = 0
    aug_count = 0

    for img_path in image_paths:
        class_name = img_path.parent.name
        clean_class_dir = clean_root / class_name
        aug_class_dir = aug_root / class_name
        ensure_dir(clean_class_dir)
        ensure_dir(aug_class_dir)

        try:
            clean_img, _ = preprocess_single_image(img_path, cfg)
        except Exception as err:
            rows.append(
                {
                    "source_path": str(img_path),
                    "class_name": class_name,
                    "output_path": "",
                    "is_augmented": False,
                    "status": f"failed: {err}",
                }
            )
            continue

        out_clean_name = f"{img_path.stem}.jpg"
        out_clean_path = clean_class_dir / out_clean_name
        cv2.imwrite(str(out_clean_path), clean_img)
        clean_count += 1

        rows.append(
            {
                "source_path": str(img_path),
                "class_name": class_name,
                "output_path": str(out_clean_path),
                "is_augmented": False,
                "status": "ok",
            }
        )

        for i in range(cfg.augment_per_image):
            aug_img = random_augment(clean_img, rng)
            aug_name = f"{img_path.stem}__aug_{i + 1:02d}.jpg"
            out_aug_path = aug_class_dir / aug_name
            cv2.imwrite(str(out_aug_path), aug_img)
            aug_count += 1

            rows.append(
                {
                    "source_path": str(img_path),
                    "class_name": class_name,
                    "output_path": str(out_aug_path),
                    "is_augmented": True,
                    "status": "ok",
                }
            )

    manifest_path = cfg.output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    metadata_path = cfg.output_dir / "preprocessing_metadata.json"
    metadata = {
        "config": {
            **asdict(cfg),
            "input_dir": str(cfg.input_dir),
            "output_dir": str(cfg.output_dir),
        },
        "counts": {
            "total_input_images": len(image_paths),
            "clean_images_written": clean_count,
            "augmented_images_written": aug_count,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return clean_count, aug_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MRI image preprocessing and augmentation")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("src/data/img_unprocessed"),
        help="Input image root for unprocessed MRI images organized by class folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/data/processed/mri"),
        help="Output directory for clean/augmented images and metadata",
    )
    parser.add_argument("--target-size", type=int, default=224, help="Output square image size")
    parser.add_argument(
        "--augment-per-image",
        type=int,
        default=1,
        help="How many augmented samples to generate per clean image",
    )
    parser.add_argument(
        "--discretization-bins",
        type=int,
        default=64,
        help="Intensity discretization bins for radiomics-style quantization",
    )
    parser.add_argument(
        "--rotation-correction",
        action="store_true",
        help="Apply mask-based PCA rotation correction",
    )
    parser.add_argument(
        "--hist-eq",
        action="store_true",
        help="Apply global histogram equalization before CLAHE",
    )
    parser.add_argument(
        "--disable-wiener",
        action="store_true",
        help="Disable Wiener filtering in denoising stage",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process only first N images (useful for quick testing)",
    )
    parser.add_argument(
        "--allow-preprocessed-input",
        action="store_true",
        help=(
            "Allow running on src/data/img_processed even though it is already preprocessed "
            "in this repository"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir_resolved = args.input_dir.resolve()
    already_preprocessed_input_resolved = DEFAULT_ALREADY_PREPROCESSED_INPUT.resolve()
    if (
        input_dir_resolved == already_preprocessed_input_resolved
        and not args.allow_preprocessed_input
    ):
        raise ValueError(
            "Refusing to run on src/data/img_processed because this repository already stores "
            "preprocessed MRI images there. Use unprocessed input (for example "
            "src/data/img_unprocessed) or pass --allow-preprocessed-input if you "
            "really need to override this guard."
        )

    cfg = PreprocessConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        apply_rotation_correction=args.rotation_correction,
        apply_hist_eq=args.hist_eq,
        apply_wiener=not args.disable_wiener,
        discretization_bins=args.discretization_bins,
        augment_per_image=max(0, args.augment_per_image),
        random_seed=args.seed,
        max_images=args.max_images,
    )

    ensure_dir(cfg.output_dir)

    clean_count, aug_count = run_pipeline(cfg)
    print("MRI preprocessing complete")
    print(f"Input dir       : {cfg.input_dir}")
    print(f"Output dir      : {cfg.output_dir}")
    print(f"Clean images    : {clean_count}")
    print(f"Augmented images: {aug_count}")
    print(f"Manifest        : {cfg.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
