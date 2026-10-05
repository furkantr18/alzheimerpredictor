"""Train multiple MRI image classification models and save the best artifact.

Default input in this repository:
    src/data/img_processed/
            MildDemented/*.jpg
            ModerateDemented/*.jpg
            NonDemented/*.jpg
            VeryMildDemented/*.jpg

Also supported (optional preprocessed layout):
        src/data/processed/mri/
            clean/<class_name>/*.jpg
            augmented/<class_name>/*.jpg

This trainer:
1. Loads clean images (and optionally augmented images).
2. Splits data by source-group to avoid leakage from augmentation pairs.
3. Builds pixel features with StandardScaler + PCA.
4. Trains multiple classifiers and compares validation/test performance.
5. Saves the best model bundle and evaluation artifacts.

Usage:
    python src/imgProcessing/mri_model_trainer.py
    python src/imgProcessing/mri_model_trainer.py --include-augmented --image-size 128
    python src/imgProcessing/mri_model_trainer.py --input-root src/data/img_processed
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class TrainerConfig:
    input_root: Path
    output_root: Path
    include_augmented: bool
    image_size: int
    val_size: float
    test_size: float
    pca_variance: float
    random_seed: int


class MRIModelTrainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.models_dir = cfg.output_root / "models"
        self.plots_dir = cfg.output_root / "plots"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=cfg.pca_variance, svd_solver="full", random_state=cfg.random_seed)

        self.records: list[dict[str, str]] = []
        self.model_results: list[dict[str, Any]] = []
        self.data_sources: list[str] = []

    @staticmethod
    def _canonical_stem(path: Path) -> str:
        stem = path.stem
        return stem.split("__aug_")[0]

    @staticmethod
    def _has_image_files(root: Path) -> bool:
        if not root.exists() or not root.is_dir():
            return False
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                return True
        return False

    def _resolve_input_roots(self) -> list[tuple[Path, str]]:
        """Resolve available roots in priority order.

        Supported layouts:
        1) Preprocessed root: <input_root>/clean and optional <input_root>/augmented
        2) Direct class root: <input_root>/<class_name>/*.jpg
        3) Fallback direct class root: src/data/img_processed/<class_name>/*.jpg
        """
        resolved: list[tuple[Path, str]] = []

        clean_root = self.cfg.input_root / "clean"
        aug_root = self.cfg.input_root / "augmented"

        if self._has_image_files(clean_root):
            resolved.append((clean_root, "clean"))
        if self.cfg.include_augmented and self._has_image_files(aug_root):
            resolved.append((aug_root, "augmented"))

        if resolved:
            return resolved

        if self._has_image_files(self.cfg.input_root):
            resolved.append((self.cfg.input_root, "clean"))
            return resolved

        fallback_raw = Path("src/data/img_processed")
        if fallback_raw != self.cfg.input_root and self._has_image_files(fallback_raw):
            print(
                f"[INFO] No images found under '{self.cfg.input_root}'. "
                f"Falling back to '{fallback_raw}'."
            )
            resolved.append((fallback_raw, "clean"))

        return resolved

    def collect_image_records(self) -> pd.DataFrame:
        roots = self._resolve_input_roots()

        rows: list[dict[str, str]] = []
        for root, subset in roots:
            if not root.exists() or not root.is_dir():
                continue

            for class_dir in sorted(root.iterdir()):
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                for image_path in class_dir.rglob("*"):
                    if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                        base_id = self._canonical_stem(image_path)
                        group_id = f"{class_name}::{base_id}"
                        rows.append(
                            {
                                "image_path": str(image_path),
                                "class_name": class_name,
                                "subset": subset,
                                "group_id": group_id,
                            }
                        )

        if not rows:
            raise FileNotFoundError(
                "No MRI images found. Checked input_root and fallback paths. "
                "Provide a valid '--input-root' such as 'src/data/processed/mri' "
                "or 'src/data/img_processed'."
            )

        df = pd.DataFrame(rows)
        self.records = rows
        self.data_sources = [str(root) for root, _ in roots]

        has_augmented_subset = any(x[1] == "augmented" for x in roots)
        if self.cfg.include_augmented and not has_augmented_subset:
            print("[WARN] '--include-augmented' requested, but no augmented folder was found. Training with available images only.")

        return df

    def split_grouped(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        groups = df[["group_id", "class_name"]].drop_duplicates().reset_index(drop=True)

        if self.cfg.val_size <= 0 or self.cfg.test_size <= 0:
            raise ValueError("val_size and test_size must be > 0")

        total_holdout = self.cfg.val_size + self.cfg.test_size
        if total_holdout >= 1.0:
            raise ValueError("val_size + test_size must be < 1.0")

        train_groups, holdout_groups = train_test_split(
            groups,
            test_size=total_holdout,
            random_state=self.cfg.random_seed,
            stratify=groups["class_name"],
        )

        val_ratio_in_holdout = self.cfg.val_size / total_holdout
        val_groups, test_groups = train_test_split(
            holdout_groups,
            test_size=1.0 - val_ratio_in_holdout,
            random_state=self.cfg.random_seed,
            stratify=holdout_groups["class_name"],
        )

        train_df = df[df["group_id"].isin(set(train_groups["group_id"]))].reset_index(drop=True)
        val_df = df[df["group_id"].isin(set(val_groups["group_id"]))].reset_index(drop=True)
        test_df = df[df["group_id"].isin(set(test_groups["group_id"]))].reset_index(drop=True)

        return train_df, val_df, test_df

    def _load_images(self, df: pd.DataFrame) -> np.ndarray:
        images: list[np.ndarray] = []
        for p in df["image_path"].tolist():
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape[0] != self.cfg.image_size or img.shape[1] != self.cfg.image_size:
                img = cv2.resize(img, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            images.append(img.reshape(-1))

        if not images:
            raise ValueError("Unable to load any images for feature extraction.")

        return np.vstack(images)

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        }

    def _get_models(self) -> dict[str, Any]:
        models: dict[str, Any] = {
            "LogisticRegression": LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                random_state=self.cfg.random_seed,
                n_jobs=-1,
            ),
            "SVM_RBF": SVC(
                C=3.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=self.cfg.random_seed,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced_subsample",
                random_state=self.cfg.random_seed,
                n_jobs=-1,
            ),
            "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(256, 128),
                early_stopping=True,
                max_iter=200,
                random_state=self.cfg.random_seed,
            ),
        }

        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=self.cfg.random_seed,
                n_jobs=-1,
            )

        return models

    def train(self) -> None:
        df = self.collect_image_records()
        train_df, val_df, test_df = self.split_grouped(df)

        y_train = self.label_encoder.fit_transform(train_df["class_name"].values)
        y_val = self.label_encoder.transform(val_df["class_name"].values)
        y_test = self.label_encoder.transform(test_df["class_name"].values)

        X_train = self._load_images(train_df)
        X_val = self._load_images(val_df)
        X_test = self._load_images(test_df)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)

        print(f"Data sources used: {', '.join(self.data_sources)}")
        print(f"Train/Val/Test sizes: {len(y_train)}/{len(y_val)}/{len(y_test)}")
        print(f"PCA components kept: {self.pca.n_components_}")

        models = self._get_models()
        best_name = None
        best_model = None
        best_val_f1 = -1.0

        for name, model in models.items():
            print(f"Training: {name}")
            model.fit(X_train_pca, y_train)

            y_val_pred = model.predict(X_val_pca)
            y_test_pred = model.predict(X_test_pca)

            val_metrics = self._metrics(y_val, y_val_pred)
            test_metrics = self._metrics(y_test, y_test_pred)

            row = {
                "model": name,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_f1_weighted": test_metrics["f1_weighted"],
            }
            self.model_results.append(row)

            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_name = name
                best_model = model

        if best_model is None or best_name is None:
            raise RuntimeError("No model was successfully trained.")

        results_df = pd.DataFrame(self.model_results).sort_values("val_f1_macro", ascending=False)
        results_df.to_csv(self.cfg.output_root / "model_comparison.csv", index=False)

        artifact = {
            "model_name": best_name,
            "model": best_model,
            "scaler": self.scaler,
            "pca": self.pca,
            "label_encoder": self.label_encoder,
            "image_size": self.cfg.image_size,
            "class_names": self.label_encoder.classes_.tolist(),
        }
        joblib.dump(artifact, self.models_dir / "best_mri_model.pkl")

        y_test_best = best_model.predict(X_test_pca)
        report = classification_report(
            y_test,
            y_test_best,
            target_names=self.label_encoder.classes_,
            digits=4,
        )
        (self.cfg.output_root / "classification_report.txt").write_text(report, encoding="utf-8")

        cm = confusion_matrix(y_test, y_test_best)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.title(f"Confusion Matrix ({best_name})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "best_model_confusion_matrix.png", dpi=200)
        plt.close()

        summary = {
            "best_model": best_name,
            "best_validation_f1_macro": best_val_f1,
            "train_size": int(len(y_train)),
            "val_size": int(len(y_val)),
            "test_size": int(len(y_test)),
            "pca_components": int(self.pca.n_components_),
            "input_root": str(self.cfg.input_root),
            "resolved_data_sources": self.data_sources,
            "output_root": str(self.cfg.output_root),
            "include_augmented": self.cfg.include_augmented,
            "image_size": self.cfg.image_size,
        }
        with open(self.cfg.output_root / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("MRI model training complete")
        print(f"Best model: {best_name}")
        print(f"Artifacts: {self.cfg.output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MRI dementia classification models")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("src/data/img_processed"),
        help="Input root. Supports preprocessed layout (clean/augmented) or direct class folders (e.g., src/data/img_processed)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("src/imgProcessing/output"),
        help="Directory where trained model and reports will be written",
    )
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Include augmented images in training pipeline",
    )
    parser.add_argument("--image-size", type=int, default=128, help="Image side length used during training")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split ratio")
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.98,
        help="PCA explained variance ratio to retain (0 < value <= 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainerConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        include_augmented=args.include_augmented,
        image_size=args.image_size,
        val_size=args.val_size,
        test_size=args.test_size,
        pca_variance=args.pca_variance,
        random_seed=args.seed,
    )

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    trainer = MRIModelTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
