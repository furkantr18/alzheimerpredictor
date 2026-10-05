"""Baseline re-run on the SAME nested folds: fine-tuned ResNet-18 with the earlier recipe.

Per outer fold: inner grouped 80/20 split of the outer-training patients -> early stopping on
inner-val loss (patience 6, <= 25 epochs) -> predict held-out slices with the best epoch ->
patient probability = mean of slice probabilities. Resumable per fold (results saved per fold).

Usage: python src/imgProcessing/stage_baseline_ft.py --task T3 [--arch resnet18] [--name B_ft_resnet18]
Output: output/cv/oof/<name>__<task>.npz, output/cv/results/<name>__<task>.json, curves in output/cv/curves/
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
from mri_dl_trainer import GpuBatcher, build_model, head_parameters, set_backbone_trainable, set_seed  # noqa: E402

RECIPE = dict(image_size=224, batch_size=64, epochs=25, lr=3e-4, backbone_lr_mult=0.1, weight_decay=1e-4, dropout=0.2,
              freeze_epochs=3, warmup_epochs=1, aug_strength=1.0, patience=6)


@torch.no_grad()
def predict(model, batcher, idx, dev, batch=128):
    model.eval()
    out, loss = [], 0.0
    for i in range(0, len(idx), batch):
        x, y = batcher.get(idx[i:i + batch])
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            lg = model(x)
        loss += float(F.cross_entropy(lg.float(), y, reduction="sum"))
        out.append(torch.softmax(lg.float(), 1).cpu().numpy())
    return loss / max(1, len(idx)), np.concatenate(out)


def train_fold(df, y, n_classes, itr, iva, te, arch, seed, recipe, dev, augment_fn=None, extra=None):
    """Returns (test slice probs, history list, best epoch, best state not kept)."""
    set_seed(seed)
    rows = df["cache_row"].to_numpy().astype(np.int64)
    batcher = GpuBatcher(rows, y.astype(np.int64), dev, recipe["image_size"])
    model = build_model(arch, n_classes, recipe["dropout"], True).to(dev)
    counts = np.bincount(y[itr], minlength=n_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    hid = set(id(p) for p in head_parameters(model, arch))
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in hid], "lr": recipe["lr"] * recipe["backbone_lr_mult"]},
                             {"params": [p for p in model.parameters() if id(p) in hid], "lr": recipe["lr"]}], weight_decay=recipe["weight_decay"])
    spe = math.ceil(len(itr) / recipe["batch_size"])
    total, warm = recipe["epochs"] * spe, recipe["warmup_epochs"] * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(1, recipe["epochs"] + 1):
        t0 = time.time()
        set_backbone_trainable(model, arch, ep > recipe["freeze_epochs"])
        model.train()
        order = rng.permutation(itr)
        ls, cor, n = 0.0, 0, 0
        for b in range(spe):
            bi = order[b * recipe["batch_size"]:(b + 1) * recipe["batch_size"]]
            if len(bi) < 2:
                continue
            x, yy = batcher.get(bi, augment=recipe["aug_strength"])
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                out = model(x)
                loss = crit(out.float(), yy)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            ls += float(loss.detach()) * len(bi); cor += int((out.argmax(1) == yy).sum()); n += len(bi)
        vl, vp = predict(model, batcher, iva, dev)
        hist.append({"epoch": ep, "train_loss": ls / n, "train_acc": cor / n, "val_loss": vl,
                     "val_acc": float((vp.argmax(1) == y[iva]).mean()), "seconds": round(time.time() - t0, 1)})
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
            best_ep = ep
        else:
            bad += 1
        if bad >= recipe["patience"]:
            break
    model.load_state_dict(best_state)
    _, pte = predict(model, batcher, te, dev)
    _, ptr = predict(model, batcher, itr, dev)
    _, pva = predict(model, batcher, iva, dev)
    del model, opt
    torch.cuda.empty_cache()
    return pte, ptr, pva, hist, best_ep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="T3")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--name", default="B_ft_resnet18")
    ap.add_argument("--folds", default="folds")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = cv.slices()
    if a.folds != "folds":
        cv.slices.use_override = True
        df = cv.slices()
    y = cv.task_labels(df["stage4"].to_numpy(), a.task)
    nc = len(cv.TASKS[a.task]["names"])
    groups = df["group"].to_numpy()
    part_dir = cv.CV_DIR / "parts" / f"{a.name}__{a.task}"
    part_dir.mkdir(parents=True, exist_ok=True)
    curves = cv.CV_DIR / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    for r, k, tr, te in cv.outer_splits(df, cv.folds(a.folds)):
        pf = part_dir / f"r{r}_k{k}.npz"
        if pf.exists():
            continue
        seed = 1000 * r + k
        itr, iva = cv.inner_holdout(df, tr, seed=seed)
        t0 = time.time()
        pte, ptr, pva, hist, bep = train_fold(df, y, nc, itr, iva, te, a.arch, seed, RECIPE, dev)
        ids, P = cv.aggregate(pte, groups[te], "mean")
        yp = np.array([y[np.where(groups == g)[0][0]] for g in ids])
        np.savez_compressed(pf, te=te, slice_prob=pte, ids=ids.astype(str), P=P, y=yp)
        (curves / f"{a.name}__{a.task}__r{r}_k{k}.json").write_text(json.dumps({"best_epoch": bep, "history": hist}))
        m = cv.patient_metrics(yp, P, nc)
        tri, trP = cv.aggregate(ptr, groups[itr], "mean")
        vai, vaP = cv.aggregate(pva, groups[iva], "mean")
        trm = cv.patient_metrics(np.array([y[np.where(groups == g)[0][0]] for g in tri]), trP, nc)["macro_f1"]
        vam = cv.patient_metrics(np.array([y[np.where(groups == g)[0][0]] for g in vai]), vaP, nc)["macro_f1"]
        cv.log_run(stage="baseline", method=a.name, config={**RECIPE, "arch": a.arch, "train_patient_mF1": round(trm, 4), "innerval_patient_mF1": round(vam, 4), "best_epoch": bep},
                   task=a.task, repeat=r, outer_fold=k, seed=seed, inner_score=round(vam, 4), outer_macro_f1=round(m["macro_f1"], 4),
                   outer_balanced_acc=round(m["balanced_acc"], 4), seconds=round(time.time() - t0, 1), exploratory=False, note="")
        print(f"[ft] {a.name} {a.task} r{r} k{k}: best ep {bep}, train pt-mF1 {trm:.3f} inner-val {vam:.3f} outer {m['macro_f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    # assemble
    oof, slice_oof = {}, {}
    for r in range(len(cv.REPEAT_SEEDS)):
        parts = [np.load(p) for p in sorted(part_dir.glob(f"r{r}_k*.npz"))]
        oof[r] = (np.concatenate([p["ids"] for p in parts]), np.concatenate([p["y"] for p in parts]), np.vstack([p["P"] for p in parts]))
        sp = np.zeros((len(df), nc))
        for p in parts:
            sp[p["te"]] = p["slice_prob"]
        slice_oof[f"slice_P_{r}"] = sp
    cv.save_oof(a.name, a.task, oof, slice_oof)
    summ = cv.summarize(oof, nc)
    sl = [cv.patient_metrics(y, slice_oof[f"slice_P_{r}"], nc) for r in sorted(oof)]
    summ["slice_level"] = {k2: float(np.mean([s[k2] for s in sl])) for k2 in ("macro_f1", "balanced_acc", "roc_auc")}
    (cv.CV_DIR / "results").mkdir(parents=True, exist_ok=True)
    (cv.CV_DIR / "results" / f"{a.name}__{a.task}.json").write_text(json.dumps(summ, indent=2))
    print(f"[ft] {a.name} {a.task}: patient macro-F1 {summ['macro_f1']['mean']:.3f} {summ['macro_f1']['ci95']} bal-acc {summ['balanced_acc']['mean']:.3f} "
          f"QWK {summ['qwk']['mean']:.3f} AUC {summ['roc_auc']['mean']:.3f} | slice mF1 {summ['slice_level']['macro_f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
