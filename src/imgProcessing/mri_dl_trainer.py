"""Deep MRI classifiers with transfer learning (ResNet-50, EfficientNet-B0, ResNet-18).

Rewritten from Yasemin's version (ae3bce0). Fixes:
* data from a saved leakage-aware split CSV; val = original images of held-out subjects;
  the test split is never touched here (final_evaluation.py uses it once);
* images come from the shared preprocessing cache (identical to inference);
* ImageNet weights + ImageNet normalisation, gray repeated to 3 channels (no lambda, so it
  also works with Windows multiprocessing);
* augmentation (random affine, flip, brightness/contrast) ONLY on train batches, on the GPU;
* optional frozen backbone for the first epochs, then fine-tuning with a lower backbone LR;
* class-weighted cross-entropy, AdamW, cosine LR schedule with warm-up, early stopping on
  validation loss, mixed precision on GPU, seeds for python/numpy/torch;
* model.train()/model.eval() used correctly; checkpoint every epoch, `--resume` continues;
* saves class_names, preprocessing spec, library versions and the split file name.

Usage:
    python src/imgProcessing/mri_dl_trainer.py --arch resnet18 --run-name r18_s42 --seed 42
    python src/imgProcessing/mri_dl_trainer.py --arch resnet18 --overfit-batch     # debug check
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402
from build_cache import open_cache  # noqa: E402
from mri_data import compute_metrics, library_versions, load_split, save_json  # noqa: E402
from mri_preprocess import IMAGENET_MEAN, IMAGENET_STD, PreprocessSpec  # noqa: E402

ARCHS = ("resnet50", "efficientnet_b0", "resnet18")


@dataclass
class TrainConfig:
    arch: str = "resnet18"
    split: str = "split_subject"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    samples_per_epoch: int = 12000      # originals + random augmented images per epoch
    lr: float = 3e-4                    # head LR; backbone uses lr * backbone_lr_mult
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    dropout: float = 0.2
    freeze_epochs: int = 1
    warmup_epochs: int = 1
    aug_strength: float = 1.0           # 0 = no augmentation, 1 = default, 2 = strong
    label_smoothing: float = 0.0
    patience: int = 5
    seed: int = 42
    pretrained: bool = True
    run_name: str = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch: str, num_classes: int, dropout: float, pretrained: bool) -> nn.Module:
    from torchvision import models
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    elif arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(max(dropout, 0.2)), nn.Linear(m.classifier[1].in_features, num_classes))
    else:
        raise ValueError(f"arch must be one of {ARCHS}")
    return m


def head_parameters(model: nn.Module, arch: str):
    return (model.classifier if arch.startswith("efficientnet") else model.fc).parameters()


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    head = set(id(p) for p in head_parameters(model, arch))
    for p in model.parameters():
        if id(p) not in head:
            p.requires_grad = trainable


class GpuBatcher:
    """Reads uint8 images from the memmap cache and returns normalised GPU tensors."""

    def __init__(self, rows: np.ndarray, labels: np.ndarray, device, size: int):
        self.cache, _ = open_cache(224)
        self.rows, self.labels, self.device, self.size = rows, labels, device, size
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def get(self, idx: np.ndarray, augment: float = 0.0):
        r = self.rows[idx]
        order = np.argsort(r)
        x = np.asarray(self.cache[r[order]])[np.argsort(order)]
        x = torch.from_numpy(x).to(self.device, non_blocking=True).float().div_(255.0).unsqueeze(1)
        if self.size != 224:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)
        if augment > 0:
            x = augment_batch(x, augment)
        x = x.repeat(1, 3, 1, 1)
        y = torch.from_numpy(self.labels[idx]).to(self.device)
        return (x - self.mean) / self.std, y


def augment_batch(x: torch.Tensor, s: float) -> torch.Tensor:
    """Per-sample random affine (rotate +-10*s deg, scale 1+-0.1*s, shift +-5*s %), h-flip,
    brightness/contrast +-0.1*s. Train only."""
    n = x.shape[0]
    dev = x.device
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(10 * s)
    scale = 1 + (torch.rand(n, device=dev) * 2 - 1) * 0.1 * s
    tx, ty = [(torch.rand(n, device=dev) * 2 - 1) * 0.05 * s * 2 for _ in range(2)]
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.1 * s
    c = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.1 * s
    m = x.mean(dim=(2, 3), keepdim=True)
    return ((x - m) * b + m + c).clamp_(0, 1)


@torch.no_grad()
def evaluate(model, batcher: GpuBatcher, idx_all: np.ndarray, batch: int, device) -> tuple[float, np.ndarray]:
    model.eval()
    probs, loss_sum = [], 0.0
    for i in range(0, len(idx_all), batch):
        x, y = batcher.get(idx_all[i:i + batch])
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        loss_sum += float(F.cross_entropy(logits.float(), y, reduction="sum"))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return loss_sum / max(len(idx_all), 1), np.concatenate(probs)


def plot_history(hist: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train loss (augmented batches)")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="val loss (originals)")
    ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8); ax[0].set_title("loss")
    ax[1].plot(hist["epoch"], hist["train_acc"], label="train acc")
    ax[1].plot(hist["epoch"], hist["val_acc"], label="val acc")
    ax[1].plot(hist["epoch"], hist["val_macro_f1"], label="val macro-F1")
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8); ax[1].set_title("accuracy / F1")
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def train(cfg: TrainConfig, overfit_batch: bool = False, resume: bool = False) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    run = cfg.run_name or f"{cfg.arch}_{cfg.split}_s{cfg.seed}"
    run_dir = C.MODELS_DIR / "dl_runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "train.log", "a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    df = load_split(cfg.split)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "val"].reset_index(drop=True)
    rows = np.concatenate([tr["cache_row"].to_numpy(), va["cache_row"].to_numpy()]).astype(np.int64)
    labels = np.concatenate([tr["y"].to_numpy(), va["y"].to_numpy()]).astype(np.int64)
    batcher = GpuBatcher(rows, labels, device, cfg.image_size)
    tr_idx = np.arange(len(tr))
    va_idx = np.arange(len(tr), len(tr) + len(va))
    is_orig = (~tr["kind"].isin(["augmented", "augmented_extra"])).to_numpy() if "kind" in tr else np.ones(len(tr), bool)
    orig_idx, aug_idx = tr_idx[is_orig], tr_idx[~is_orig]

    model = build_model(cfg.arch, len(C.CLASS_NAMES), cfg.dropout, cfg.pretrained).to(device)  # NCHW: channels_last was 5-8x slower on this GPU/cuDNN
    counts = np.bincount(tr["y"], minlength=len(C.CLASS_NAMES)).astype(float)
    w = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device), label_smoothing=cfg.label_smoothing)
    head_ids = set(id(p) for p in head_parameters(model, cfg.arch))
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": cfg.lr * cfg.backbone_lr_mult},
        {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": cfg.lr}], weight_decay=cfg.weight_decay)
    # one epoch = all training originals + random dataset copies up to samples_per_epoch (honest split: originals only)
    epoch_size = len(orig_idx) + min(len(aug_idx), max(0, cfg.samples_per_epoch - len(orig_idx)))
    steps_per_epoch = max(1, math.ceil(epoch_size / cfg.batch_size))
    total, warm = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    if overfit_batch:  # debug: 32 fixed images, no augmentation, must reach ~100% train accuracy
        rng = np.random.default_rng(cfg.seed)
        fixed = np.concatenate([rng.choice(orig_idx[tr["y"].to_numpy()[orig_idx] == c], 8, replace=False) for c in range(4)
                                if (tr["y"].to_numpy()[orig_idx] == c).sum() >= 8])
        set_backbone_trainable(model, cfg.arch, True)
        # constant LR here: the warm-up schedule of the real run would keep the LR near 0 for these few steps
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        t_start = time.time()
        for step in range(150):
            model.train()
            x, y = batcher.get(fixed)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = F.cross_entropy(out.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if step % 25 == 0 or step == 149:
                acc = float((out.argmax(1) == y).float().mean())
                say(f"[overfit-batch] step {step} loss {float(loss.detach()):.4f} acc {acc:.3f} ({time.time() - t_start:.1f}s)")
        return {"overfit_batch_final_acc": acc, "overfit_batch_final_loss": float(loss)}

    ckpt_path = run_dir / "last.pt"
    hist, best = [], {"val_loss": float("inf"), "epoch": -1}
    start_epoch, bad_epochs = 1, 0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        hist, best, start_epoch, bad_epochs = ck["hist"], ck["best"], ck["epoch"] + 1, ck["bad_epochs"]
        torch.set_rng_state(ck["rng_torch"].cpu()); np.random.set_state(ck["rng_np"])  # rng state must stay on CPU
        say(f"resumed from epoch {ck['epoch']}")
    say(f"device={device} cfg={json.dumps(asdict(cfg))}")
    say(f"train images={len(tr)} (originals {len(orig_idx)}, augmented {len(aug_idx)}), val originals={len(va)}; class weights={np.round(w, 3).tolist()}")

    rng = np.random.default_rng(cfg.seed + 1000)
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        set_backbone_trainable(model, cfg.arch, epoch > cfg.freeze_epochs)
        n_aug = max(0, cfg.samples_per_epoch - len(orig_idx))
        ep_idx = np.concatenate([orig_idx, rng.choice(aug_idx, size=min(n_aug, len(aug_idx)), replace=False)]) if len(aug_idx) else orig_idx
        rng.shuffle(ep_idx)
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for b in range(steps_per_epoch):
            bi = ep_idx[(b * cfg.batch_size) % len(ep_idx):][:cfg.batch_size]
            if len(bi) < 2:
                continue
            x, y = batcher.get(bi, augment=cfg.aug_strength)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            loss_sum += float(loss) * len(bi); correct += int((out.argmax(1) == y).sum()); seen += len(bi)
        val_loss, val_prob = evaluate(model, batcher, va_idx, 128, device)
        vm = compute_metrics(va["y"].to_numpy(), val_prob, n_boot=0)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen, "val_loss": val_loss,
               "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"], "val_bal_acc": vm["balanced_accuracy"],
               "lr_head": opt.param_groups[1]["lr"], "backbone_trainable": epoch > cfg.freeze_epochs, "seconds": round(time.time() - t0, 1)}
        hist.append(row)
        improved = val_loss < best["val_loss"] - 1e-4
        if improved:
            best = {"val_loss": val_loss, "epoch": epoch, "val_macro_f1": vm["macro_f1"], "val_acc": vm["accuracy"]}
            torch.save(model.state_dict(), run_dir / "best.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1
        say(f"epoch {epoch:02d} train_loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | val_loss {val_loss:.4f} "
            f"acc {vm['accuracy']:.3f} macroF1 {vm['macro_f1']:.3f} {'*' if improved else ''} ({row['seconds']}s)")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "hist": hist, "best": best, "epoch": epoch, "bad_epochs": bad_epochs,
                    "rng_torch": torch.get_rng_state(), "rng_np": np.random.get_state()}, ckpt_path)
        if bad_epochs >= cfg.patience:
            say(f"early stopping at epoch {epoch} (no val-loss improvement for {cfg.patience} epochs)")
            break

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(run_dir / "history.csv", index=False)
    plot_history(hist_df, run_dir / "curves.png", f"{run}: best epoch {best['epoch']} (val loss {best['val_loss']:.3f})")
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    _, tr_prob = evaluate(model, batcher, orig_idx, 128, device)
    train_m = compute_metrics(tr["y"].to_numpy()[orig_idx], tr_prob, n_boot=0)
    _, val_prob = evaluate(model, batcher, va_idx, 128, device)
    val_m = compute_metrics(va["y"].to_numpy(), val_prob, va["subject_group"].to_numpy(), n_boot=200)
    meta = {"kind": "deep", "run": run, "arch": cfg.arch, "class_names": C.CLASS_NAMES, "config": asdict(cfg),
            "preprocess": PreprocessSpec(size=224).to_dict(), "image_size": cfg.image_size,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD, "gray_to_3ch": True},
            "split_file": f"{cfg.split}.csv", "best": best, "train_originals_metrics": train_m, "val_metrics": val_m,
            "versions": library_versions(), "weights_file": "best.pt"}
    save_json(meta, run_dir / "meta.json")
    say(f"done: best epoch {best['epoch']} | train(originals) acc {train_m['accuracy']:.3f} | val acc {val_m['accuracy']:.3f} "
        f"macroF1 {val_m['macro_f1']:.3f} bal-acc {val_m['balanced_accuracy']:.3f}")
    log.close()
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k, v in asdict(TrainConfig()).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--overfit-batch", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = vars(ap.parse_args())
    flags = {k: a.pop(k) for k in ("overfit_batch", "resume")}
    C.ensure_dirs()
    out = train(TrainConfig(**a), **flags)
    if flags["overfit_batch"]:
        print(json.dumps(out))


if __name__ == "__main__":
    main()
