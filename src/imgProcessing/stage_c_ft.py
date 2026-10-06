"""Stage C: partial fine-tuning with stronger regularisation, on the same nested folds (T3).

EfficientNet-B0 (ImageNet), only the last 2 feature stages + head trainable, EMA of weights
(decay 0.999) used for validation/test, mixup (alpha 0.2), MRI-appropriate augmentation
(GPU affine: rotation <= 10 deg, shift <= 5%, scale +-10%, h-flip, intensity jitter; no vertical
flip), class-weighted CE, AdamW (lr 3e-4 head / 1e-4 unfrozen blocks, wd 1e-2), warm-up + cosine,
<= 25 epochs, early stopping on inner-val loss of the EMA model (patience 6), AMP, batch 32.
Patient level = mean of slice probabilities. Resumable per fold.

Usage: python src/imgProcessing/stage_c_ft.py --task T3
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_core as cv  # noqa: E402
from mri_dl_trainer import GpuBatcher, set_seed  # noqa: E402
from stage_baseline_ft import predict  # noqa: E402

CFG = dict(arch="efficientnet_b0", image_size=224, batch_size=32, epochs=25, lr_head=3e-4, lr_blocks=1e-4, weight_decay=1e-2,
           unfreeze_stages=2, ema=0.999, mixup=0.2, aug_strength=1.0, warmup_epochs=1, patience=6, dropout=0.3)
NAME = "C_ft_effb0_partial_ema_mixup"


def build(n_classes):
    from torchvision import models
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(CFG["dropout"]), nn.Linear(m.classifier[1].in_features, n_classes))
    for p in m.parameters():
        p.requires_grad = False
    blocks = list(m.features)[-CFG["unfreeze_stages"]:]
    for b in blocks:
        for p in b.parameters():
            p.requires_grad = True
    for p in m.classifier.parameters():
        p.requires_grad = True
    return m, [p for b in blocks for p in b.parameters()], list(m.classifier.parameters())


def train_fold(df, y, nc, itr, iva, te, seed, dev):
    set_seed(seed)
    batcher = GpuBatcher(df["cache_row"].to_numpy().astype(np.int64), y.astype(np.int64), dev, CFG["image_size"])
    model, pb, ph = build(nc)
    model.to(dev)
    ema = copy.deepcopy(model).eval()
    counts = np.bincount(y[itr], minlength=nc).astype(float)
    w = torch.tensor(np.where(counts > 0, counts.sum() / (nc * np.maximum(counts, 1)), 0.0), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW([{"params": pb, "lr": CFG["lr_blocks"]}, {"params": ph, "lr": CFG["lr_head"]}], weight_decay=CFG["weight_decay"])
    spe = math.ceil(len(itr) / CFG["batch_size"])
    total, warm = CFG["epochs"] * spe, CFG["warmup_epochs"] * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    rng = np.random.default_rng(seed)
    best, best_state, best_ep, bad, hist = float("inf"), None, 0, 0, []
    for ep in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        model.train()
        for mod in model.modules():  # frozen BatchNorm layers keep their ImageNet statistics
            if isinstance(mod, nn.BatchNorm2d) and not any(p.requires_grad for p in mod.parameters()):
                mod.eval()
        order = rng.permutation(itr)
        ls, n = 0.0, 0
        for b in range(spe):
            bi = order[b * CFG["batch_size"]:(b + 1) * CFG["batch_size"]]
            if len(bi) < 2:
                continue
            x, yy = batcher.get(bi, augment=CFG["aug_strength"])
            lam = float(rng.beta(CFG["mixup"], CFG["mixup"])) if CFG["mixup"] > 0 else 1.0
            perm = torch.randperm(len(bi), device=dev)
            x = lam * x + (1 - lam) * x[perm]
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                out = model(x).float()
                loss = lam * F.cross_entropy(out, yy, weight=w) + (1 - lam) * F.cross_entropy(out, yy[perm], weight=w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(CFG["ema"]).add_(pm.detach(), alpha=1 - CFG["ema"])
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            ls += float(loss.detach()) * len(bi); n += len(bi)
        vl, vp = predict(ema, batcher, iva, dev)
        hist.append({"epoch": ep, "train_loss": ls / n, "val_loss": vl, "val_acc": float((vp.argmax(1) == y[iva]).mean()), "seconds": round(time.time() - t0, 1)})
        if vl < best - 1e-4:
            best, best_ep, bad, best_state = vl, ep, 0, copy.deepcopy(ema.state_dict())
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break
    ema.load_state_dict(best_state)
    _, pte = predict(ema, batcher, te, dev)
    _, ptr = predict(ema, batcher, itr, dev)
    _, pva = predict(ema, batcher, iva, dev)
    del model, ema, opt
    torch.cuda.empty_cache()
    return pte, ptr, pva, hist, best_ep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="T3")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = cv.slices()
    y = cv.task_labels(df["stage4"].to_numpy(), a.task)
    nc = len(cv.TASKS[a.task]["names"])
    groups = df["group"].to_numpy()
    part_dir = cv.CV_DIR / "parts" / f"{NAME}__{a.task}"
    part_dir.mkdir(parents=True, exist_ok=True)
    (cv.CV_DIR / "curves").mkdir(parents=True, exist_ok=True)

    def py(ix):
        return np.array([y[np.where(groups == g)[0][0]] for g in ix])

    for r, k, tr, te in cv.outer_splits(df, cv.folds()):
        pf = part_dir / f"r{r}_k{k}.npz"
        if pf.exists():
            continue
        seed = 1000 * r + k
        itr, iva = cv.inner_holdout(df, tr, seed=seed)
        t0 = time.time()
        pte, ptr, pva, hist, bep = train_fold(df, y, nc, itr, iva, te, seed, dev)
        ids, P = cv.aggregate(pte, groups[te], "mean")
        np.savez_compressed(pf, te=te, slice_prob=pte, ids=ids.astype(str), P=P, y=py(ids))
        (cv.CV_DIR / "curves" / f"{NAME}__{a.task}__r{r}_k{k}.json").write_text(json.dumps({"best_epoch": bep, "history": hist}))
        m = cv.patient_metrics(py(ids), P, nc)
        tri, trP = cv.aggregate(ptr, groups[itr], "mean"); vai, vaP = cv.aggregate(pva, groups[iva], "mean")
        trm, vam = cv.patient_metrics(py(tri), trP, nc)["macro_f1"], cv.patient_metrics(py(vai), vaP, nc)["macro_f1"]
        cv.log_run(stage="C", method=NAME, config={**CFG, "train_patient_mF1": round(trm, 4), "best_epoch": bep}, task=a.task, repeat=r,
                   outer_fold=k, seed=seed, inner_score=round(vam, 4), outer_macro_f1=round(m["macro_f1"], 4),
                   outer_balanced_acc=round(m["balanced_acc"], 4), seconds=round(time.time() - t0, 1), exploratory=False, note="")
        print(f"[C] {NAME} {a.task} r{r} k{k}: best ep {bep} train {trm:.3f} inner-val {vam:.3f} outer {m['macro_f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    oof, slice_oof = {}, {}
    for r in range(len(cv.REPEAT_SEEDS)):
        parts = [np.load(p) for p in sorted(part_dir.glob(f"r{r}_k*.npz"))]
        oof[r] = (np.concatenate([p["ids"] for p in parts]), np.concatenate([p["y"] for p in parts]), np.vstack([p["P"] for p in parts]))
        sp = np.zeros((len(df), nc))
        for p in parts:
            sp[p["te"]] = p["slice_prob"]
        slice_oof[f"slice_P_{r}"] = sp
    cv.save_oof(NAME, a.task, oof, slice_oof)
    s = cv.summarize(oof, nc)
    sl = [cv.patient_metrics(y, slice_oof[f"slice_P_{r}"], nc) for r in sorted(oof)]
    s["slice_level"] = {k2: float(np.mean([q[k2] for q in sl])) for k2 in ("macro_f1", "balanced_acc", "roc_auc")}
    (cv.CV_DIR / "results" / f"{NAME}__{a.task}.json").write_text(json.dumps(s, indent=2))
    print(f"[C] {NAME} {a.task}: patient macro-F1 {s['macro_f1']['mean']:.3f} {s['macro_f1']['ci95']} bal-acc {s['balanced_acc']['mean']:.3f} QWK {s['qwk']['mean']:.3f} AUC {s['roc_auc']['mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
