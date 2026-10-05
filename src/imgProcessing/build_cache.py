"""Preprocess every unique image ONCE with the shared preprocessing and store it in a
uint8 memmap (N, size, size). Training/evaluation then read from the cache, which is
much faster than decoding JPEGs each epoch and guarantees identical preprocessing.

Rows are keyed by `pixel_md5` (decoded-pixel hash from the inventory), so the 40,384
images that exist in both dataset folders are stored once.

Usage:
    python src/imgProcessing/build_cache.py [--size 224] [--workers 12]
Output: output/cache/pre_<size>.u8 (+ pre_<size>_index.csv, pre_<size>_meta.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402
from mri_preprocess import PreprocessSpec, load_preprocessed  # noqa: E402

_SPEC: PreprocessSpec | None = None


def _init(spec_dict: dict) -> None:
    global _SPEC
    _SPEC = PreprocessSpec.from_dict(spec_dict)


def _work(path: str) -> np.ndarray | None:
    try:
        return load_preprocessed(path, _SPEC)
    except Exception as err:  # unreadable: skipped and counted by the caller
        print(f"[WARN] unreadable image skipped: {path} ({err})", flush=True)
        return None


def cache_paths(size: int) -> tuple[Path, Path, Path]:
    base = C.CACHE_DIR / f"pre_{size}"
    return base.with_suffix(".u8"), Path(str(base) + "_index.csv"), Path(str(base) + "_meta.json")


def open_cache(size: int = 224) -> tuple[np.memmap, dict[str, int]]:
    data_p, idx_p, meta_p = cache_paths(size)
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    arr = np.memmap(data_p, dtype=np.uint8, mode="r", shape=(meta["rows"], size, size))
    idx = pd.read_csv(idx_p)
    return arr, dict(zip(idx["pixel_md5"], idx["row"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    C.ensure_dirs()

    inv = pd.read_csv(C.OUTPUT_DIR / "inventory" / "images.csv")
    inv = inv[inv["readable"]]
    uniq = inv.drop_duplicates("pixel_md5")[["pixel_md5", "path"]].reset_index(drop=True)
    spec = PreprocessSpec(size=args.size)
    data_p, idx_p, meta_p = cache_paths(args.size)

    arr = np.memmap(data_p, dtype=np.uint8, mode="w+", shape=(len(uniq), args.size, args.size))
    ok = np.zeros(len(uniq), dtype=bool)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(spec.to_dict(),)) as ex:
        for i, img in enumerate(ex.map(_work, uniq["path"].tolist(), chunksize=128)):
            if img is not None:
                arr[i] = img
                ok[i] = True
            if i % 5000 == 0:
                print(f"[cache] {i}/{len(uniq)}", flush=True)
    arr.flush()
    uniq["row"] = np.arange(len(uniq))
    uniq = uniq[ok]
    uniq[["pixel_md5", "row"]].to_csv(idx_p, index=False)
    meta_p.write_text(json.dumps({"rows": int(len(ok)), "valid": int(ok.sum()), "skipped_unreadable": int((~ok).sum()),
                                  "spec": spec.to_dict()}, indent=2), encoding="utf-8")
    print(f"[cache] done: {ok.sum()} images, {(~ok).sum()} skipped -> {data_p}")


if __name__ == "__main__":
    main()
