"""Central paths and constants for the MRI image pipeline.

Every path is absolute and built from this file's location, so scripts work from any
working directory. Datasets are never copied into the repo: they are referenced by path.

Override with environment variables or the CLI flags of each script:
    MRI_DATASETS_DIR   folder that contains the two Kaggle downloads (default: <alzheimerdx_fixed>/datasets)
    MRI_OUTPUT_DIR     where splits, reports, plots and models are written (default: src/imgProcessing/output)
"""

from __future__ import annotations

import os
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent            # src/imgProcessing
SRC_DIR = IMG_DIR.parent                              # src
REPO_ROOT = SRC_DIR.parent                            # alzheimerpredictor

DATASETS_DIR = Path(os.environ.get("MRI_DATASETS_DIR", REPO_ROOT.parent.parent / "datasets")).resolve()
OUTPUT_DIR = Path(os.environ.get("MRI_OUTPUT_DIR", IMG_DIR / "output")).resolve()

# Identified in Phase 1 (see _backend/IMAGE_INTEGRATION.md, "Decisions"):
#   datasets/archive      -> aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented (combined_images/<class>)
#   datasets/archive (1)  -> uraninjo/augmented-alzheimer-mri-dataset-v2 (data/train = augmented, data/val = originals)
COMBINED_ROOT = DATASETS_DIR / "archive" / "combined_images"
URANINJO_ROOT = DATASETS_DIR / "archive (1)" / "data"

# Fixed class order. Saved with every model and used by every predictor.
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

MODELS_DIR = OUTPUT_DIR / "models"
SPLITS_DIR = OUTPUT_DIR / "splits"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"
CACHE_DIR = OUTPUT_DIR / "cache"

# Model served by the API (written by the training scripts, git-ignored)
SERVED_MODEL_DIR = MODELS_DIR / "served"


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, MODELS_DIR, SPLITS_DIR, REPORTS_DIR, PLOTS_DIR, LOGS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
            yield p
