"""Phase 3.3: quantify leakage with ONE fixed simple model (no tuning).

Model: shared preprocessing -> 64x64 pixels -> StandardScaler -> PCA(128) -> LogisticRegression(C=0.01, balanced).
Trained on the TRAIN part of each split; held-out predictions are produced by
final_evaluation.py (test, once). Here only validation numbers are printed.

Settings
  naive            split_naive   (archive/combined_images, random per image, like Yasemin's code)
  image_group      split_image   (copies grouped with their source image; slices of one patient may cross)
  subject_group    split_subject (pseudo-patient grouping; the honest setting)
Sanity checks
  naive_shuffled_by_source    labels replaced by a random label PER SOURCE SUBJECT (all copies/slices of a
                              subject share it). Labels carry no disease information, so any held-out accuracy
                              above chance can only come from leakage (memorised copies) -> real-data version of
                              the synthetic test in IMAGE_DIAGNOSIS.md B2.
  subject_shuffled_by_source  same random labels under the subject split: should be at chance.
  majority                    always the most frequent training class.

Usage: python src/imgProcessing/leakage_experiment.py
Output: output/models/leakage_<setting>.joblib
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402
from mri_data import compute_metrics, load_split, pixel_features  # noqa: E402

SEED = 42


def simple_model() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=128, svd_solver="randomized", random_state=SEED)),
                     ("clf", LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced", random_state=SEED))])


def source_subject_map() -> dict[str, str]:
    """pixel_md5 -> subject group (originals + confidently matched copies) from groups.csv."""
    g = pd.read_csv(C.SPLITS_DIR / "groups.csv")
    g = g[g["subject_group"] != "UNCERTAIN"]
    return dict(zip(g["pixel_md5"], g["subject_group"]))


def random_labels_by_subject(df: pd.DataFrame, subj: dict[str, str]) -> np.ndarray:
    """One random class per subject; images without a known subject get a random class of their own."""
    rng = np.random.default_rng(SEED)
    keys = df["pixel_md5"].map(subj).fillna(df["pixel_md5"])
    uniq = keys.unique()
    lab = dict(zip(uniq, rng.integers(0, len(C.CLASS_NAMES), len(uniq))))
    return keys.map(lab).to_numpy()


def settings():
    subj = source_subject_map()
    for name, split, shuffled in [("naive", "split_naive", False), ("image_group", "split_image", False),
                                  ("subject_group", "split_subject", False),
                                  ("naive_shuffled_by_source", "split_naive", True),
                                  ("subject_shuffled_by_source", "split_subject", True)]:
        df = load_split(split)
        if shuffled:
            df["y"] = random_labels_by_subject(df, subj)
        yield name, split, df


def main() -> None:
    C.ensure_dirs()
    for name, split, df in settings():
        tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
        model = simple_model().fit(pixel_features(tr), tr["y"].to_numpy())
        full = np.zeros((len(va), len(C.CLASS_NAMES)))
        full[:, model[-1].classes_] = model.predict_proba(pixel_features(va))
        m = compute_metrics(va["y"].to_numpy(), full, n_boot=0)
        majority = int(np.bincount(tr["y"]).argmax())
        joblib.dump({"setting": name, "split": split, "pipeline": model, "shuffled": "shuffled" in name,
                     "labels_override": df[["pixel_md5", "y"]] if "shuffled" in name else None,
                     "majority_class": majority, "feature_size": 64}, C.MODELS_DIR / f"leakage_{name}.joblib")
        print(f"[leakage] {name:28s} train={len(tr):6d} val={len(va):5d} val acc {m['accuracy']:.3f} "
              f"macro-F1 {m['macro_f1']:.3f} bal-acc {m['balanced_accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
