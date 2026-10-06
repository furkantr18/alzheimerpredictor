"""Figures and markdown tables for MRI_IMPROVEMENT_REPORT.md (reads output/cv/*).

Usage: python src/imgProcessing/report_figures.py
Output: output/cv/report/{tables.md, fig_*.png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_core as cv  # noqa: E402

OUT = cv.CV_DIR / "report"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def fmt_ci(m, ci):
    return f"{m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def table(task: str) -> str:
    p = cv.CV_DIR / "results" / f"comparison_{task}.json"
    if not p.exists():
        return f"(no comparison for {task})\n"
    rows = json.loads(p.read_text())
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    lines = [f"| Method | Patient macro-F1 [95% CI] | Balanced acc. [95% CI] | QWK | ROC-AUC | Slice macro-F1 | Δ macro-F1 vs baseline [95% CI] | distinguishable |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        d = "–" if pd.isna(r.get("diff_macro_f1")) else f"{r['diff_macro_f1']:+.3f} [{r['diff_macro_f1_ci'][0]:+.3f}, {r['diff_macro_f1_ci'][1]:+.3f}]"
        dist = "–" if pd.isna(r.get("diff_macro_f1")) else ("**yes**" if r["diff_macro_f1_distinguishable"] else "no")
        sl = "–" if pd.isna(r.get("slice_macro_f1")) else f"{r['slice_macro_f1']:.3f}"
        lines.append(f"| {r['method']} | {fmt_ci(r['macro_f1'], r['macro_f1_ci'])} | {fmt_ci(r['balanced_acc'], r['balanced_acc_ci'])} | "
                     f"{r['qwk']:.3f} | {r['roc_auc']:.3f} | {sl} | {d} | {dist} |")
    return "\n".join(lines) + "\n"


def fig_curves(name: str, task: str = "T3") -> None:
    files = sorted((cv.CV_DIR / "curves").glob(f"{name}__{task}__r*_k*.json"))
    if not files:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for f in files:
        h = pd.DataFrame(json.loads(f.read_text())["history"])
        ax[0].plot(h["epoch"], h["train_loss"], color=BLUE, alpha=0.35, lw=1)
        ax[0].plot(h["epoch"], h["val_loss"], color=ORANGE, alpha=0.35, lw=1)
        if "train_acc" in h:
            ax[1].plot(h["epoch"], h["train_acc"], color=BLUE, alpha=0.35, lw=1)
        ax[1].plot(h["epoch"], h["val_acc"], color=ORANGE, alpha=0.35, lw=1)
    for a, t in zip(ax, ("loss (blue train, orange inner-val)", "slice accuracy (blue train, orange inner-val)")):
        a.set_title(t, fontsize=9, color=INK, loc="left"); a.set_xlabel("epoch", color=MUTED, fontsize=8)
        a.grid(True, color=GRID, lw=0.8); [a.spines[s].set_visible(False) for s in ("top", "right")]
    fig.suptitle(f"{name} ({task}): all {len(files)} outer folds", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(OUT / f"fig_curves_{name}_{task}.png", dpi=130); plt.close(fig)


def fig_attention(name: str, task: str = "T3") -> None:
    p = cv.CV_DIR / "results" / f"{name}__{task}__attention.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d["rel"] = d["attention"] * d.groupby(["repeat", "group"])["attention"].transform("size")  # 1.0 = uniform
    s = d.groupby("slice_pos")["rel"].agg(["mean", "std", "size"])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(s.index, s["mean"], color=BLUE, width=0.8)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.text(s.index.max(), 1.02, "uniform", color=MUTED, fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("slice position (0..32, order inside the scan)", color=MUTED, fontsize=8)
    ax.set_ylabel("attention / uniform", color=MUTED, fontsize=8)
    ax.set_title(f"MIL attention by slice position ({name}, {task}, held-out patients)", fontsize=9.5, color=INK, loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); [ax.spines[x].set_visible(False) for x in ("top", "right")]
    fig.tight_layout(); fig.savefig(OUT / f"fig_attention_{name}_{task}.png", dpi=130); plt.close(fig)
    s.round(3).to_csv(OUT / f"attention_by_position_{name}_{task}.csv")


def fig_reliability(names: list[str], task: str = "T3") -> dict:
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    out = {}
    for name, col in zip(names, (BLUE, ORANGE, "#1baf7a", "#e87ba4")):
        try:
            oof = cv.load_oof(name, task)
        except FileNotFoundError:
            continue
        conf, corr = [], []
        e = []
        for r, (ids, y, P) in oof.items():
            conf += P.max(1).tolist(); corr += (P.argmax(1) == y).tolist(); e.append(cv.ece(y, P))
        conf, corr = np.array(conf), np.array(corr)
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean()); ys.append(corr[m].mean())
        ax.plot(xs, ys, marker="o", ms=4, lw=2, color=col, label=f"{name} (ECE {np.mean(e):.3f})")
        out[name] = float(np.mean(e))
    ax.set_xlabel("confidence", color=MUTED, fontsize=8); ax.set_ylabel("accuracy", color=MUTED, fontsize=8)
    ax.set_title(f"Reliability, patient level ({task})", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left"); ax.grid(True, color=GRID, lw=0.8)
    [ax.spines[x].set_visible(False) for x in ("top", "right")]
    fig.tight_layout(); fig.savefig(OUT / f"fig_reliability_{task}.png", dpi=130); plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = []
    for t in ("T3", "T2", "T4"):
        md.append(f"### {t}\n\n" + table(t))
    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    for n in ("B_ft_resnet18", "C_ft_effb0_partial_ema_mixup"):
        fig_curves(n)
    for p in (cv.CV_DIR / "results").glob("B_mil_attn_*__T3__attention.csv"):
        fig_attention(p.name.split("__")[0])
    calib = sorted(p.name.split("__")[0] for p in (cv.CV_DIR / "oof").glob("B_mil_*__T3.npz"))[:2]
    ece = fig_reliability(["B_ft_resnet18", "E_ensemble3_AB"] + calib)
    (OUT / "ece.json").write_text(json.dumps(ece, indent=2))
    print("report assets in", OUT)


if __name__ == "__main__":
    main()
