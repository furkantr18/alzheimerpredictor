"""Stage B strong-baseline check: handcrafted per-slice features ("how much is ventricle size?").

Per slice (shared preprocessing, 224 px): brain area share, intensity mean/std/10/50/90th percentile
inside the (hole-filled) brain mask, dark-pixel (CSF proxy, < 60) share inside the brain, and the
dark share inside a central box (middle 40% of the brain bounding box: lateral-ventricle proxy).
Saved as an "embedding" file so the same nested CV (stage_a_heads.py) evaluates it.
Usage: python src/imgProcessing/handcrafted.py ; python src/imgProcessing/stage_a_heads.py --emb handcrafted --head logreg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_core as cv  # noqa: E402
from build_cache import open_cache  # noqa: E402

NAMES = ["brain_area", "int_mean", "int_std", "int_p10", "int_p50", "int_p90", "dark_share", "central_dark_share"]


def features(g: np.ndarray) -> list[float]:
    m = (g > 10).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    mask = (m | (1 - flood)).astype(bool)
    vals = g[mask].astype(np.float32) if mask.any() else np.zeros(1, np.float32)
    ys, xs = np.nonzero(mask)
    if len(xs):
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cy0, cy1 = int(y0 + 0.3 * (y1 - y0)), int(y0 + 0.7 * (y1 - y0))
        cx0, cx1 = int(x0 + 0.3 * (x1 - x0)), int(x0 + 0.7 * (x1 - x0))
        c = g[cy0:cy1, cx0:cx1][mask[cy0:cy1, cx0:cx1]]
        cdark = float((c < 60).mean()) if c.size else 0.0
    else:
        cdark = 0.0
    return [float(mask.mean()), float(vals.mean()), float(vals.std()), *np.percentile(vals, [10, 50, 90]).tolist(),
            float((vals < 60).mean()), cdark]


def main() -> None:
    df = cv.slices()
    arr, _ = open_cache(224)
    F = np.array([features(np.asarray(arr[r])) for r in df["cache_row"].to_numpy()], dtype=np.float32)
    d = cv.CV_DIR / "emb"
    np.save(d / "handcrafted.npy", F)
    st_p = d / "status.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    st["handcrafted"] = {"ok": True, "dim": F.shape[1], "features": NAMES, "note": "not a pretrained backbone; excluded from --all"}
    st_p.write_text(json.dumps(st, indent=2))
    print(F.shape, dict(zip(NAMES, np.round(F.mean(0), 3).tolist())))


if __name__ == "__main__":
    main()
