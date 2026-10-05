"""ONE shared preprocessing function for MRI training and inference.

Every trainer, predictor and the API call `preprocess_gray()` (or `load_gray()` +
`preprocess_gray()`), so train and inference images always go through identical steps.
The config is saved with each model (`PreprocessSpec.to_dict()`) and re-applied at
inference from that saved dict.

Steps (decided in Phase 2, see IMAGE_INTEGRATION.md "Decisions"):
1. Decode to 8-bit grayscale (any common format; RGB is converted).
1b. Background level = median of the 3-pixel image border; it is subtracted, so augmented
   images whose background was brightened to grey get a black background again
   (originals have a 0 border, so nothing changes for them).
2. Brain bounding box: pixels brighter than `fg_threshold` (the Kaggle images are already
   skull-stripped on a black background), plus `margin` pixels. Cropping removes the black
   border, whose size differs between source images and could become a shortcut.
3. Letterbox to a square (keeps the aspect ratio, pads with black) and resize to `size`.
Intensity is NOT equalised here: the images are already normalised by the dataset author,
and the deep model adds ImageNet normalisation in `to_tensor()`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python-headless is required: pip install -r requirements-image.txt") from exc

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessSpec:
    size: int = 224
    crop_to_brain: bool = True
    fg_threshold: int = 10
    margin: int = 4
    version: str = "mri-pre-v2"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessSpec":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def decode_gray(data: bytes) -> np.ndarray:
    """Decode image bytes to uint8 grayscale. Raises ValueError if not decodable."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None
    if img is None or img.size == 0:
        raise ValueError("not a decodable image")
    return img


def load_gray(path: str | Path) -> np.ndarray:
    """Read an image file as uint8 grayscale (works with non-ASCII / spaced Windows paths)."""
    return decode_gray(Path(path).read_bytes())


def brain_bbox(img: np.ndarray, threshold: int, margin: int) -> tuple[int, int, int, int] | None:
    # Background level from the image border: some augmented images have a brightness
    # shift that turns the black background grey, so a fixed threshold would keep it all.
    border = np.concatenate([img[:3].ravel(), img[-3:].ravel(), img[:, :3].ravel(), img[:, -3:].ravel()])
    thr = max(threshold, int(np.median(border)) + threshold)
    ys, xs = np.nonzero(img > thr)
    if xs.size < 50:  # almost empty image: keep it whole
        return None
    h, w = img.shape[:2]
    return (max(0, int(ys.min()) - margin), min(h, int(ys.max()) + 1 + margin),
            max(0, int(xs.min()) - margin), min(w, int(xs.max()) + 1 + margin))


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(canvas, (size, size), interpolation=interp)


def preprocess_gray(img: np.ndarray, spec: PreprocessSpec) -> np.ndarray:
    """uint8 gray (H, W) -> uint8 gray (size, size). Deterministic."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Shift the background back to black (no-op for originals, whose border is 0), so the
    # padded letterbox border and the brain surround look the same for every image.
    bg = int(np.median(np.concatenate([img[:3].ravel(), img[-3:].ravel(), img[:, :3].ravel(), img[:, -3:].ravel()])))
    if bg > 0:
        img = cv2.subtract(img, np.full_like(img, bg))
    if spec.crop_to_brain:
        box = brain_bbox(img, spec.fg_threshold, spec.margin)
        if box is not None:
            y0, y1, x0, x1 = box
            img = img[y0:y1, x0:x1]
    return letterbox(img, spec.size)


def load_preprocessed(path: str | Path, spec: PreprocessSpec) -> np.ndarray:
    return preprocess_gray(load_gray(path), spec)


def to_tensor(img_u8: np.ndarray):
    """Preprocessed uint8 gray -> float tensor (3, H, W) with ImageNet normalisation.

    Gray is repeated to 3 channels because the pretrained backbones expect RGB input.
    """
    import torch

    t = torch.from_numpy(img_u8).float().div_(255.0).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std
