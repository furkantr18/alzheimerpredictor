"""Exploratory analysis of the modeling cohorts (same participants as the model comparison: those with
a matched FreeSurfer 7 baseline scan).

Writes to output/1_data_analysis/<task>/:
    tables/table1_characteristics.csv   participants by phase and label (paper Table 1)
    tables/missingness.csv              share of missing values per feature
    figures/fig_counts                  participants per phase and label
    figures/fig_age                     age by label per phase (shows the age confound in ADNI3)
    figures/fig_markers                 ADNIMERGE MRI and cognitive markers by label
    figures/fig_engineered_markers      the 10 engineered FreeSurfer 7 markers by label
    figures/fig_correlation             Spearman correlations of the interpretable features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import cohort as co
from common import config as cfg
from common import features as ft
from common import plots
from common.plots import plt

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


def _legend_handles():
    return [plt.Rectangle((0, 0), 1, 1, color=plots.SERIES[k]) for k in (0, 1)]


def box_pair(ax, data0, data1, pos):
    for k, d in enumerate((data0, data1)):
        d = d.dropna()
        if d.empty:
            continue
        b = ax.boxplot(d, positions=[pos + (k - 0.5) * 0.36], widths=0.3, patch_artist=True,
                       showfliers=False, medianprops={"color": plots.SURFACE, "linewidth": 1.5},
                       whiskerprops={"color": plots.TEXT_2}, capprops={"color": plots.TEXT_2})
        b["boxes"][0].set(facecolor=plots.SERIES[k], edgecolor=plots.SERIES[k])


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
    plots.save(fig, out / "fig_counts")


def fig_age(c, task, out):
    names = plots.LABEL_NAMES[task]
    phases = _phases(c)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for i, p in enumerate(phases):
        s = c[c["phase"] == p]
        box_pair(ax, s.loc[s.label == 0, "AGE"], s.loc[s.label == 1, "AGE"], i)
    ax.set_xticks(range(len(phases)), phases)
    ax.set_ylabel("Age at baseline (years)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=_legend_handles(), labels=names, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("Age by diagnosis and phase")
    plots.save(fig, out / "fig_age")


def fig_box_grid(x, labels, items, task, title, out, name, ncols=3):
    names = plots.LABEL_NAMES[task]
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.3 * nrows), squeeze=False)
    for ax, (col, label) in zip(axes.flat, items):
        box_pair(ax, x.loc[labels == 0, col], x.loc[labels == 1, col], 0)
        ax.set_xticks([-0.18, 0.18], names)
        ax.set_title(label, fontsize=8)
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(items):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=11, fontweight="bold", color=plots.TEXT)
    fig.tight_layout()
    plots.save(fig, out / name)


def fig_correlation(c, matched, out):
    x = ft.demo(c).join(ft.cog(c)).join(ft.mri_engineered(c, matched)).assign(label=c["label"])
    corr = x.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation")
    ax.set_title("Feature correlations (incl. label)")
    plots.save(fig, out / "fig_correlation")


def main(tasks=cfg.TASKS, merge=None, fs=None):
    plots.setup()
    merge = co.load_adnimerge() if merge is None else merge
    fs = co.load_fs7() if fs is None else fs
    for task in tasks:
        tables = cfg.out_dir("analysis", task, "tables")
        figures = cfg.out_dir("analysis", task, "figures")
        c, matched, _ = co.modeling_cohort(merge, fs, task)
        cohort_table(c).to_csv(tables / "table1_characteristics.csv", index=False)
        x = ft.demo(c).join(ft.cog(c)).join(ft.mri_merge(c)).join(ft.mri_engineered(c, matched))
        x.isna().mean().round(3).rename("missing_share").to_csv(tables / "missingness.csv")
        names = plots.LABEL_NAMES[task]
        fig_counts(c, task, figures)
        fig_age(c, task, figures)
        fig_box_grid(x, c["label"], MARKERS, task, f"Baseline markers, {names[0]} vs {names[1]} (all phases)",
                     figures, "fig_markers")
        eng = [(k, k.replace("_", " ")) for k in ft.ENGINEERED]
        fig_box_grid(x, c["label"], eng, task, f"Engineered MRI markers, {names[0]} vs {names[1]}", figures,
                     "fig_engineered_markers", ncols=5)
        fig_correlation(c, matched, figures)
        print(f"[{task}] {len(c)} participants -> {figures.parent}")
