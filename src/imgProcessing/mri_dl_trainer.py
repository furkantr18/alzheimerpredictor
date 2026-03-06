"""Train deep learning MRI classifiers with transfer learning.

Supported architectures:
- resnet50
- efficientnet_b0

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

Usage:
    python src/imgProcessing/mri_dl_trainer.py --architecture resnet50
    python src/imgProcessing/mri_dl_trainer.py --architecture efficientnet_b0 --epochs 25
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DLTrainerConfig:
    input_root: Path
    output_root: Path
    include_augmented: bool
    architecture: str
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    val_size: float
    test_size: float
    random_seed: int
    patience: int
    num_workers: int
    pretrained: bool


class MRISliceDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.records.iloc[idx]
        image_path = Path(row["image_path"])

        with Image.open(image_path) as img:
            image = img.convert("L")

        tensor = self.transform(image)
        label = int(self.class_to_idx[str(row["class_name"])])
        return tensor, label


class MRITransferTrainer:
    def __init__(self, cfg: DLTrainerConfig) -> None:
        self.cfg = cfg
        self.output_root = cfg.output_root
        self.models_dir = self.output_root / "models"
        self.plots_dir = self.output_root / "plots"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_sources: list[str] = []

    @staticmethod
    def _canonical_stem(path: Path) -> str:
        return path.stem.split("__aug_")[0]

    @staticmethod
    def _has_image_files(root: Path) -> bool:
        if not root.exists() or not root.is_dir():
            return False
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                return True
        return False

    def _resolve_input_roots(self) -> list[tuple[Path, str]]:
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
            return [(self.cfg.input_root, "clean")]

        fallback_root = Path("src/data/img_processed")
        if fallback_root != self.cfg.input_root and self._has_image_files(fallback_root):
            print(
                f"[INFO] No images found under '{self.cfg.input_root}'. "
                f"Falling back to '{fallback_root}'."
            )
            return [(fallback_root, "clean")]

        return []

    def collect_records(self) -> pd.DataFrame:
        roots = self._resolve_input_roots()
        rows: list[dict[str, str]] = []

        for root, subset in roots:
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
                "No MRI images found. Provide '--input-root' with class folders "
                "or clean/augmented layout."
            )

        self.data_sources = [str(root) for root, _ in roots]
        return pd.DataFrame(rows)

    def split_grouped(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.cfg.val_size <= 0 or self.cfg.test_size <= 0:
            raise ValueError("val_size and test_size must be > 0")

        total_holdout = self.cfg.val_size + self.cfg.test_size
        if total_holdout >= 1.0:
            raise ValueError("val_size + test_size must be < 1.0")

        groups = df[["group_id", "class_name"]].drop_duplicates().reset_index(drop=True)

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

    def _build_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        train_tf = transforms.Compose(
            [
                transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

        eval_tf = transforms.Compose(
            [
                transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

        return train_tf, eval_tf

    def _build_model(self, num_classes: int) -> nn.Module:
        arch = self.cfg.architecture.lower()

        if arch == "resnet50":
            weights = None
            if self.cfg.pretrained and hasattr(models, "ResNet50_Weights"):
                weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model

        if arch == "efficientnet_b0":
            weights = None
            if self.cfg.pretrained and hasattr(models, "EfficientNet_B0_Weights"):
                weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model

        raise ValueError(
            "Unsupported architecture. Use one of: resnet50, efficientnet_b0"
        )

    def _evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        model.eval()
        total_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = model(inputs)
                loss = criterion(logits, labels)
                total_loss += float(loss.item()) * int(inputs.size(0))

                preds = torch.argmax(logits, dim=1)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        y_true_arr = np.array(y_true, dtype=np.int64)
        y_pred_arr = np.array(y_pred, dtype=np.int64)
        avg_loss = total_loss / max(len(y_true_arr), 1)
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
        return avg_loss, acc, f1_macro, y_true_arr, y_pred_arr

    def train(self) -> None:
        df = self.collect_records()
        train_df, val_df, test_df = self.split_grouped(df)

        class_names = sorted(df["class_name"].unique().tolist())
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        train_tf, eval_tf = self._build_transforms()
        train_ds = MRISliceDataset(train_df, class_to_idx, train_tf)
        val_ds = MRISliceDataset(val_df, class_to_idx, eval_tf)
        test_ds = MRISliceDataset(test_df, class_to_idx, eval_tf)

        pin_memory = bool(self.device.type == "cuda")
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=pin_memory,
        )

        train_labels = [class_to_idx[x] for x in train_df["class_name"].tolist()]
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(class_names)),
            y=np.array(train_labels, dtype=np.int64),
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        model = self._build_model(num_classes=len(class_names)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, self.cfg.patience // 3),
        )

        print(f"Device: {self.device}")
        print(f"Architecture: {self.cfg.architecture}")
        print(f"Data sources: {', '.join(self.data_sources)}")
        print(f"Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

        best_state: dict[str, Any] | None = None
        best_epoch = -1
        best_val_f1 = -1.0
        best_val_loss = float("inf")
        epochs_without_improve = 0

        history_rows: list[dict[str, float | int]] = []

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            y_true_train: list[int] = []
            y_pred_train: list[int] = []

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * int(inputs.size(0))
                preds = torch.argmax(logits, dim=1)
                y_true_train.extend(labels.cpu().numpy().tolist())
                y_pred_train.extend(preds.cpu().numpy().tolist())

            train_loss = running_loss / max(len(y_true_train), 1)
            train_acc = float(accuracy_score(y_true_train, y_pred_train))
            train_f1 = float(f1_score(y_true_train, y_pred_train, average="macro"))

            val_loss, val_acc, val_f1, _, _ = self._evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_f1_macro": train_f1,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1_macro": val_f1,
                }
            )

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_f1={train_f1:.4f} val_f1={val_f1:.4f}"
            )

            improved = (val_f1 > best_val_f1) or (
                np.isclose(val_f1, best_val_f1) and val_loss < best_val_loss
            )

            if improved:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= self.cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience={self.cfg.patience})")
                break

        if best_state is None:
            raise RuntimeError("Training ended without a valid best checkpoint.")

        model.load_state_dict(best_state)

        test_loss, test_acc, test_f1, y_test_true, y_test_pred = self._evaluate(
            model, test_loader, criterion
        )

        checkpoint_path = self.models_dir / "best_mri_dl_model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        metadata = {
            "architecture": self.cfg.architecture,
            "class_names": class_names,
            "image_size": self.cfg.image_size,
            "pretrained": self.cfg.pretrained,
            "best_epoch": best_epoch,
            "best_val_f1_macro": best_val_f1,
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
            "input_root": str(self.cfg.input_root),
            "resolved_data_sources": self.data_sources,
        }
        metadata_path = self.models_dir / "best_mri_dl_model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(self.output_root / "dl_training_history.csv", index=False)

        report = classification_report(
            y_test_true,
            y_test_pred,
            target_names=class_names,
            digits=4,
        )
        (self.output_root / "dl_classification_report.txt").write_text(report, encoding="utf-8")

        cm = confusion_matrix(y_test_true, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"DL Confusion Matrix ({self.cfg.architecture})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "best_mri_dl_confusion_matrix.png", dpi=200)
        plt.close()

        summary = {
            "best_model": self.cfg.architecture,
            "best_checkpoint": str(checkpoint_path),
            "metadata_path": str(metadata_path),
            "best_epoch": best_epoch,
            "best_validation_f1_macro": float(best_val_f1),
            "test_accuracy": float(test_acc),
            "test_f1_macro": float(test_f1),
            "test_loss": float(test_loss),
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "batch_size": int(self.cfg.batch_size),
            "epochs_requested": int(self.cfg.epochs),
            "patience": int(self.cfg.patience),
            "architecture": self.cfg.architecture,
            "device": str(self.device),
        }
        with open(self.output_root / "dl_training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("DL MRI training complete")
        print(f"Best epoch: {best_epoch}")
        print(f"Best val F1 (macro): {best_val_f1:.4f}")
        print(f"Test F1 (macro): {test_f1:.4f}")
        print(f"Artifacts written to: {self.output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train transfer learning MRI dementia classifiers"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("src/data/img_processed"),
        help="Input root with class folders or clean/augmented structure",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("src/imgProcessing/output"),
        help="Directory where checkpoints and reports are written",
    )
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Include augmented subset when input root contains it",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0"],
        help="Transfer learning backbone",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input image side length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = DLTrainerConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        include_augmented=args.include_augmented,
        architecture=args.architecture,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_size=args.val_size,
        test_size=args.test_size,
        random_seed=args.seed,
        patience=args.patience,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
    )

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    trainer = MRITransferTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
