"""Extra data checks for the paper (originals only, subject split).

1. Cross-subject near-duplicates: for every val/test original, the highest correlation to ANY
   train original (64x64, zero-mean, unit-norm) compared with the typical correlation of
   neighbouring slices of the same patient (~0.99). Many cross-split pairs at that level would
   mean the reconstructed subject IDs miss repeat scans of the same person.
2. Global-feature baseline: a classifier that only sees mean intensity, intensity std, brain
   area and dark-pixel share inside the brain. If it does well, the classes differ in global
   image statistics (could be real atrophy/CSF signal or a dataset artefact = shortcut risk).

Usage: python src/imgProcessing/data_checks.py      (validation only; test is not touched)
Output: output/reports/data_checks.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402
from build_cache import open_cache  # noqa: E402
from mri_data import compute_metrics, load_split, pixel_features, save_json  # noqa: E402


def global_features(df) -> np.ndarray:
    arr, _ = open_cache(224)
    feats = []
    for r in df["cache_row"].to_numpy():
        g = np.asarray(arr[r]).astype(np.float32)
        m = g > 10
        inside = g[m] if m.any() else np.zeros(1)
        feats.append([inside.mean(), inside.std(), m.mean(), (inside < 60).mean()])
    return np.array(feats)


def main() -> None:
    df = load_split("split_subject")
    tr, va = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "val"].reset_index(drop=True)
    out = {}

    X = pixel_features(df, 64)
    X = X - X.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
    is_tr = (df["split"] == "train").to_numpy()
    held = ~is_tr
    sims = X[held] @ X[is_tr].T
    best = sims.max(1)
    out["cross_split_max_corr_quantiles"] = {q: float(np.quantile(best, q)) for q in (0.5, 0.9, 0.99, 1.0)}
    out["heldout_images_with_train_corr_>=0.98"] = int((best >= 0.98).sum())
    out["heldout_images_with_train_corr_>=0.95"] = int((best >= 0.95).sum())
    out["heldout_images"] = int(held.sum())
    # reference: same-subject neighbours inside train
    from subject_ids import parse
    t = df[is_tr].reset_index(drop=True)
    key = {(c, *parse(c, s)): i for i, (c, s) in enumerate(zip(t["class_name"], t["stem"]))}
    Xt = X[is_tr]
    nb = [float(Xt[i] @ Xt[key[(c, s, k + 1)]]) for (c, s, k), i in key.items() if (c, s, k + 1) in key]
    out["same_subject_neighbour_corr_median"] = float(np.median(nb))

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    gtr, gva = global_features(tr), global_features(va)
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(gtr, tr["y"])
    p = np.zeros((len(va), 4)); p[:, clf[-1].classes_] = clf.predict_proba(gva)
    m = compute_metrics(va["y"].to_numpy(), p, n_boot=0)
    out["global_feature_baseline_val"] = {k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}
    out["global_features_by_class_train"] = {c: dict(zip(["mean_int", "std_int", "brain_area", "dark_share"], np.round(gtr[tr["class_name"] == c].mean(0), 3).tolist()))
                                            for c in C.CLASS_NAMES}
    save_json(out, C.REPORTS_DIR / "data_checks.json")
    import json
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
