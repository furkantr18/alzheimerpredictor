"""Phase 1: inventory of both Kaggle MRI datasets.

For every image: dataset, split, class, file-name pattern, size, mode, exact hashes
(file bytes and decoded pixels) and perceptual hashes (dHash, pHash, pHash of the
mirrored image). Writes <output>/inventory/images.csv, a summary JSON and per-class
sample grids. Read-only on the datasets.

Usage:
    python src/imgProcessing/dataset_inventory.py [--datasets-dir PATH] [--workers 12]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402

UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
PATTERNS = [
    ("uuid", re.compile(rf"^{UUID_RE}$")),
    ("aug_n_uuid", re.compile(rf"^aug_\d+_{UUID_RE}$")),
    ("aug_n_name", re.compile(r"^aug_\d+_.+$")),
    ("classDemN", re.compile(r"^(mild|moderate|non|verymild)Dem\d+$")),
    ("N (N)", re.compile(r"^\d+ \(\d+\)$")),
    ("N", re.compile(r"^\d+$")),
]


def name_pattern(stem: str) -> str:
    for label, rx in PATTERNS:
        if rx.match(stem):
            return label
    return "other"


def dhash(gray: Image.Image) -> int:
    a = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def phash_arr(a32: np.ndarray) -> int:
    d = dctn(a32, norm="ortho")[:8, :8].flatten()
    med = np.median(d[1:])
    return int("".join("1" if v > med else "0" for v in d), 2)


def describe(path_str: str) -> dict:
    p = Path(path_str)
    raw = p.read_bytes()
    row = {"path": path_str, "file_md5": hashlib.md5(raw).hexdigest(), "bytes": len(raw)}
    try:
        with Image.open(p) as im:
            row.update(width=im.width, height=im.height, mode=im.mode)
            gray = im.convert("L")
        arr = np.asarray(gray)
        row["pixel_md5"] = hashlib.md5(arr.tobytes()).hexdigest()
        row["mean_intensity"] = float(arr.mean())
        row["frac_black"] = float((arr < 10).mean())
        a32 = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        row["dhash"] = f"{dhash(gray):016x}"
        row["phash"] = f"{phash_arr(a32):016x}"
        row["phash_flip"] = f"{phash_arr(a32[:, ::-1].copy()):016x}"
        row["readable"] = True
    except Exception as err:  # unreadable file: recorded, never fatal
        row["readable"] = False
        row["error"] = str(err)[:200]
    return row


def collect_paths(datasets_dir: Path) -> pd.DataFrame:
    rows = []
    combined = datasets_dir / "archive" / "combined_images"
    for p in C.iter_images(combined):
        rows.append({"path": str(p), "dataset": "archive", "split": "all", "class_name": p.parent.name})
    uran = datasets_dir / "archive (1)" / "data"
    for p in C.iter_images(uran):
        rows.append({"path": str(p), "dataset": "archive (1)", "split": p.parent.parent.name, "class_name": p.parent.name})
    df = pd.DataFrame(rows)
    df["stem"] = df["path"].map(lambda s: Path(s).stem)
    df["name_pattern"] = df["stem"].map(name_pattern)
    return df


def sample_grids(df: pd.DataFrame, out_dir: Path, per_class: int = 12, seed: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for (ds, split), sub in df.groupby(["dataset", "split"]):
        classes = sorted(sub["class_name"].unique())
        fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.4, len(classes) * 1.6))
        for r, cls in enumerate(classes):
            pick = sub[sub["class_name"] == cls].sample(n=min(per_class, (sub["class_name"] == cls).sum()), random_state=seed)
            for c in range(per_class):
                ax = axes[r, c]
                ax.axis("off")
                if c < len(pick):
                    row = pick.iloc[c]
                    with Image.open(row["path"]) as im:
                        ax.imshow(im.convert("L"), cmap="gray")
                    ax.set_title(row["name_pattern"], fontsize=6)
            axes[r, 0].text(-0.15, 0.5, cls, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
        fig.suptitle(f"{ds} / {split}: random samples per class (title = file-name pattern)", fontsize=10)
        fig.tight_layout()
        safe = f"{ds}_{split}".replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(out_dir / f"samples_{safe}.png", dpi=110)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets-dir", type=Path, default=C.DATASETS_DIR)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    out_dir = C.OUTPUT_DIR / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_paths(args.datasets_dir)
    print(f"[inventory] {len(df)} image files found under {args.datasets_dir}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        info = list(ex.map(describe, df["path"].tolist(), chunksize=256))
    df = df.merge(pd.DataFrame(info), on="path", how="left")
    df.to_csv(out_dir / "images.csv", index=False)
    print(f"[inventory] wrote {out_dir / 'images.csv'}; unreadable: {(~df['readable']).sum()}")

    sample_grids(df[df["readable"]], out_dir)
    print(f"[inventory] sample grids in {out_dir}")


if __name__ == "__main__":
    main()
