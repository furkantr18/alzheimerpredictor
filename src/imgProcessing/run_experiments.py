"""Phase 4 driver: hyper-parameter search, final multi-seed runs, naive (leaky) runs.

Resumable: a run whose meta.json exists is skipped; an interrupted run continues from its
last checkpoint (--resume). Progress goes to output/logs/experiments.log.

Stages
  search  ResNet-18, seed 42: one-factor-at-a-time around a baseline (documented below).
          Selection metric: validation macro-F1 at the best-val-loss epoch (val = originals of
          29 unseen subjects; ModerateDemented has no val subject).
  final   best config for resnet18, efficientnet_b0, resnet50 x seeds 42, 43, 44 (split_subject)
  naive   the same best config on split_naive (Yasemin's leaky setting), resnet18 + resnet50, seed 42

Usage: python src/imgProcessing/run_experiments.py --stage search|final|naive|all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402
from mri_dl_trainer import TrainConfig  # noqa: E402

BASE = dict(arch="resnet18", split="split_subject", image_size=224, batch_size=32, epochs=25, lr=3e-4,
            backbone_lr_mult=0.1, weight_decay=1e-4, dropout=0.2, freeze_epochs=1, warmup_epochs=1,
            aug_strength=1.0, label_smoothing=0.0, patience=6, seed=42)
VARIANTS = {
    "base": {},
    "lr1e-4": {"lr": 1e-4}, "lr1e-3": {"lr": 1e-3},
    "bbmult1": {"backbone_lr_mult": 1.0},
    "wd1e-2": {"weight_decay": 1e-2},
    "drop0": {"dropout": 0.0}, "drop0.5": {"dropout": 0.5},
    "freeze0": {"freeze_epochs": 0}, "freeze3": {"freeze_epochs": 3},
    "aug0": {"aug_strength": 0.0}, "aug2": {"aug_strength": 2.0},
    "size160": {"image_size": 160},
    "bs64": {"batch_size": 64},
    "ls0.1": {"label_smoothing": 0.1},
}
LOG = C.LOGS_DIR / "experiments.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name: str, cfg: dict) -> dict | None:
    run_dir = C.MODELS_DIR / "dl_runs" / name
    if (run_dir / "meta.json").exists():
        log(f"skip {name} (done)")
        return json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    args = [sys.executable, str(Path(__file__).with_name("mri_dl_trainer.py")), "--run-name", name, "--resume"]
    for k, v in cfg.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    log(f"start {name}: {json.dumps(cfg)}")
    t0 = time.time()
    with open(C.LOGS_DIR / f"run_{name}.out", "a", encoding="utf-8") as out:
        rc = subprocess.run(args, stdout=out, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not (run_dir / "meta.json").exists():
        log(f"FAILED {name} (exit {rc}); see logs/run_{name}.out")
        return None
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    v = meta["val_metrics"]
    log(f"done {name} in {time.time() - t0:.0f}s: best epoch {meta['best']['epoch']} val macroF1 {v['macro_f1']:.3f} "
        f"bal-acc {v['balanced_accuracy']:.3f} acc {v['accuracy']:.3f}")
    return meta


def best_config() -> dict:
    p = C.REPORTS_DIR / "hp_search.csv"
    df = pd.read_csv(p)
    top = df.sort_values(["val_macro_f1", "val_loss"], ascending=[False, True]).iloc[0]
    return {**BASE, **json.loads(top["overrides"])}


def stage_search() -> None:
    rows = []
    for name, ov in VARIANTS.items():
        meta = run(f"search_{name}", {**BASE, **ov})
        if meta:
            rows.append({"variant": name, "overrides": json.dumps(ov), "best_epoch": meta["best"]["epoch"],
                         "val_loss": meta["best"]["val_loss"], "val_macro_f1": meta["val_metrics"]["macro_f1"],
                         "val_bal_acc": meta["val_metrics"]["balanced_accuracy"], "val_acc": meta["val_metrics"]["accuracy"],
                         "train_orig_acc": meta["train_originals_metrics"]["accuracy"]})
    pd.DataFrame(rows).to_csv(C.REPORTS_DIR / "hp_search.csv", index=False)
    log(f"search finished; best = {json.dumps(best_config())}")


def stage_final() -> None:
    cfg = best_config()
    for arch in ("resnet18", "efficientnet_b0", "resnet50"):
        for seed in (42, 43, 44):
            run(f"final_{arch}_s{seed}", {**cfg, "arch": arch, "seed": seed})


def stage_naive() -> None:
    cfg = best_config()
    for arch in ("resnet18", "resnet50"):
        run(f"naive_{arch}_s42", {**cfg, "arch": arch, "split": "split_naive", "seed": 42})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all", choices=["search", "final", "naive", "all"])
    a = ap.parse_args()
    C.ensure_dirs()
    assert set(BASE) <= set(asdict(TrainConfig())), "BASE keys must match TrainConfig"
    if a.stage in ("search", "all"):
        stage_search()
    if a.stage in ("final", "all"):
        stage_final()
    if a.stage in ("naive", "all"):
        stage_naive()


if __name__ == "__main__":
    main()
