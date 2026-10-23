"""Exploratory analysis of the ADNI baseline cohorts (figures + tables for the paper).
Uses the same participants as run_models.py (those with a matched FreeSurfer 7 baseline scan).

    python adni_predictor/src/eda.py

Writes to adni_predictor/output/eda/ (git-ignored):
    table1_<task>.csv            participant characteristics by phase and label (paper Table 1)
    missingness_<task>.csv       share of missing values per feature
    fig_counts_<task>            participants per phase and label
    fig_age_<task>               age by label per phase (shows the age confound in ADNI3)
    fig_markers_<task>           key MRI and cognitive markers by label
    fig_correlation_<task>       correlation of the ADNIMERGE features
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adni_config as cfg  # noqa: E402
import cohort as co  # noqa: E402
import features as ft  # noqa: E402
import plots  # noqa: E402
from plots import plt  # noqa: E402

TASKS = ["CN_AD", "sMCI_pMCI"]

MARKERS = [  # (column, axis label)
    ("Hippocampus_icv", "Hippocampus / ICV"),
    ("Entorhinal_icv", "Entorhinal / ICV"),
    ("Ventricles_icv", "Ventricles / ICV"),
    ("MMSE", "MMSE"),
    ("ADAS13", "ADAS-Cog 13"),
    ("RAVLT_immediate", "RAVLT immediate"),
]


def cohort_table(c: pd.DataFrame) -> pd.DataFrame:
    g = c.groupby(["phase", "label"])
    return pd.DataFrame({
        "n": g.size(),
        "age_mean": g["AGE"].mean().round(1),
        "age_sd": g["AGE"].std().round(1),
        "female_pct": g["PTGENDER"].apply(lambda s: round(100 * (s == "Female").mean(), 1)),
        "educ_mean": g["PTEDUCAT"].mean().round(1),
        "apoe4_carrier_pct": g["APOE4"].apply(lambda s: round(100 * (s > 0).mean(), 1)),
        "mmse_mean": g["MMSE"].mean().round(1),
    }).reset_index()


def _phases(c):
    return [p for p in co.PHASE_ORDER if p in set(c["phase"])]


def fig_counts(c, task, out):
    names = plots.LABEL_NAMES[task]
    phases = _phases(c)
    counts = c.groupby(["phase", "label"]).size().unstack(fill_value=0).reindex(phases, fill_value=0)
    x = np.arange(len(phases))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for k in (0, 1):
        bars = ax.bar(x + (k - 0.5) * w, counts.get(k, 0), w - 0.02, color=plots.SERIES[k], label=names[k])
        ax.bar_label(bars, padding=2, fontsize=7, color=plots.TEXT_2)
    ax.set_xticks(x, phases)
    ax.set_ylabel("Participants")
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=2, loc="upper right")
    ax.set_title(f"{names[0]} vs {names[1]}: participants per ADNI phase")
    plots.save(fig, out / f"fig_counts_{task}")


def _box_pair(ax, data0, data1, pos, names):
    for k, d in enumerate((data0, data1)):
        d = d.dropna()
        if d.empty:
            continue
        b = ax.boxplot(d, positions=[pos + (k - 0.5) * 0.36], widths=0.3, patch_artist=True,
                       showfliers=False, medianprops={"color": plots.SURFACE, "linewidth": 1.5},
                       whiskerprops={"color": plots.TEXT_2}, capprops={"color": plots.TEXT_2})
        b["boxes"][0].set(facecolor=plots.SERIES[k], edgecolor=plots.SERIES[k])


def fig_age(c, task, out):
    names = plots.LABEL_NAMES[task]
    phases = _phases(c)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for i, p in enumerate(phases):
        s = c[c["phase"] == p]
        _box_pair(ax, s.loc[s.label == 0, "AGE"], s.loc[s.label == 1, "AGE"], i, names)
    ax.set_xticks(range(len(phases)), phases)
    ax.set_ylabel("Age at baseline (years)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=plots.SERIES[k]) for k in (0, 1)], labels=names,
              loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("Age by diagnosis and phase")
    plots.save(fig, out / f"fig_age_{task}")


def fig_markers(c, task, out):
    names = plots.LABEL_NAMES[task]
    x = ft.mri_merge(c).join(ft.cog(c))
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.6))
    for ax, (col, label) in zip(axes.flat, MARKERS):
        _box_pair(ax, x.loc[c.label == 0, col], x.loc[c.label == 1, col], 0, names)
        ax.set_xticks([-0.18, 0.18], names)
        ax.set_title(label, fontsize=9)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Baseline markers, {names[0]} vs {names[1]} (all phases)", fontsize=11, fontweight="bold",
                 color=plots.TEXT)
    fig.tight_layout()
    plots.save(fig, out / f"fig_markers_{task}")


def fig_correlation(c, task, out):
    x = ft.demo(c).join(ft.cog(c)).join(ft.mri_merge(c)).assign(label=c["label"])
    corr = x.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation")
    ax.set_title("Feature correlations (incl. label)")
    plots.save(fig, out / f"fig_correlation_{task}")


def main():
    plots.setup()
    out = cfg.OUTPUT_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)
    merge, fs = co.load_adnimerge(), co.load_fs7()
    for task in TASKS:
        c, _, _ = co.modeling_cohort(merge, fs, task)
        cohort_table(c).to_csv(out / f"table1_{task}.csv", index=False)
        x = ft.demo(c).join(ft.cog(c)).join(ft.mri_merge(c))
        x.isna().mean().round(3).rename("missing_share").to_csv(out / f"missingness_{task}.csv")
        fig_counts(c, task, out)
        fig_age(c, task, out)
        fig_markers(c, task, out)
        fig_correlation(c, task, out)
        print(f"[{task}] {len(c)} participants -> {out}")


if __name__ == "__main__":
    main()
