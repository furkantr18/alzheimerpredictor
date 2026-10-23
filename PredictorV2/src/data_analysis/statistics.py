"""Statistical group comparisons (descriptive only: never used to select features for the models,
because choosing features on all participants before CV would leak test information).

Writes:
    output/1_data_analysis/<task>/tables/group_comparison.csv
        per feature: mean (SD) per group, Welch t-test, Mann-Whitney U, Hedges' g, chi-square for binary
        features, Benjamini-Hochberg FDR q-values over all tests in the table
    output/1_data_analysis/<task>/figures/fig_effect_sizes
    output/1_data_analysis/all_participants/tables/anova_baseline_diagnosis.csv
        one-way ANOVA and Kruskal-Wallis across baseline diagnosis CN / EMCI / LMCI / AD, eta squared, FDR
    output/1_data_analysis/all_participants/figures/fig_markers_by_diagnosis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import cohort as co
from common import config as cfg
from common import features as ft
from common import plots
from common.plots import plt

DX_ORDER = ["CN", "EMCI", "LMCI", "AD"]
BINARY = {"female": lambda c: (c["PTGENDER"] == "Female").astype(float),
          "apoe4_carrier": lambda c: (c["APOE4"] > 0).astype(float)}
DOMAINS = [("demographics", ["AGE", "PTEDUCAT"]), ("cognition", ft.COG),
           ("mri_merge", [f"{v}_icv" for v in ft.MERGE_VOLUMES]), ("mri_eng", list(ft.ENGINEERED))]
BY_DX_MARKERS = ["hippocampus_icv", "hippocampal_occupancy", "entorhinal_thickness", "ad_signature_thickness",
                 "ventricles_icv", "MMSE"]


def fdr(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs are kept)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        q[ok] = stats.false_discovery_control(p[ok], method="bh")
    return q


def hedges_g(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2))
    return (b.mean() - a.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9))


def _features(c, matched):
    return ft.demo(c).join(ft.cog(c)).join(ft.mri_merge(c)).join(ft.mri_engineered(c, matched))


def group_comparison(c: pd.DataFrame, matched: pd.DataFrame, task: str) -> pd.DataFrame:
    n0, n1 = plots.LABEL_NAMES[task]
    x = _features(c, matched)
    y = c["label"]
    rows = []
    for domain, cols in DOMAINS:
        for col in cols:
            a, b = x.loc[y == 0, col].dropna(), x.loc[y == 1, col].dropna()
            rows.append({"domain": domain, "feature": col, "test": "Welch t / Mann-Whitney",
                         f"{n0}_n": len(a), f"{n0}_mean": a.mean(), f"{n0}_sd": a.std(),
                         f"{n1}_n": len(b), f"{n1}_mean": b.mean(), f"{n1}_sd": b.std(),
                         "statistic": stats.ttest_ind(a, b, equal_var=False).statistic,
                         "p_welch_t": stats.ttest_ind(a, b, equal_var=False).pvalue,
                         "p_mann_whitney": stats.mannwhitneyu(a, b).pvalue, "effect_size": hedges_g(a, b),
                         "effect_measure": "Hedges g"})
    for name, f in BINARY.items():
        v = f(c)
        table = pd.crosstab(v, y)
        chi2, p, _, _ = stats.chi2_contingency(table)
        rows.append({"domain": "demographics", "feature": name, "test": "chi-square",
                     f"{n0}_n": int((y == 0).sum()), f"{n0}_mean": v[y == 0].mean(),
                     f"{n1}_n": int((y == 1).sum()), f"{n1}_mean": v[y == 1].mean(),
                     "statistic": chi2, "p_welch_t": p,
                     "effect_size": np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))),
                     "effect_measure": "Cramer V"})
    out = pd.DataFrame(rows)
    out["q_fdr"] = fdr(out["p_welch_t"])
    out["significant_q05"] = out["q_fdr"] < 0.05
    return out


def fig_effect_sizes(table: pd.DataFrame, task: str, out):
    n0, n1 = plots.LABEL_NAMES[task]
    t = table[table.effect_measure == "Hedges g"].sort_values("effect_size")
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(t) + 1.2))
    colors = [plots.SERIES[1] if g > 0 else plots.SERIES[0] for g in t["effect_size"]]
    ax.barh(range(len(t)), t["effect_size"], color=colors, height=0.65)
    ax.set_yticks(range(len(t)), [f"{f}{'' if s else '  (n.s.)'}" for f, s in zip(t.feature, t.significant_q05)],
                  fontsize=7)
    ax.axvline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xlabel(f"Hedges' g ({n1} minus {n0})")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=plots.SERIES[k]) for k in (1, 0)],
              labels=[f"higher in {n1}", f"lower in {n1}"], loc="lower right")
    ax.set_title(f"{n0} vs {n1}: effect sizes (n.s. = FDR q ≥ 0.05)")
    plots.save(fig, out / "fig_effect_sizes")


def anova_by_diagnosis(merge: pd.DataFrame, fs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bl = co.baseline(merge)
    bl = bl[bl["DX_bl"].isin(DX_ORDER)].copy()
    bl["phase"] = bl["ORIGPROT"]
    matched = co.match_baseline_scan(bl, fs)
    bl = bl.loc[bl.index.intersection(matched.index)]
    x = _features(bl, matched)
    rows = []
    for domain, cols in DOMAINS:
        for col in cols:
            groups = [x.loc[bl["DX_bl"] == g, col].dropna() for g in DX_ORDER]
            groups = [g for g in groups if len(g) > 1]
            f, p = stats.f_oneway(*groups)
            h, pk = stats.kruskal(*groups)
            allv = pd.concat(groups)
            ss_between = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
            ss_total = ((allv - allv.mean()) ** 2).sum()
            rows.append({"domain": domain, "feature": col, "F": f, "p_anova": p, "H": h, "p_kruskal": pk,
                         "eta_squared": ss_between / ss_total,
                         **{f"{g}_mean": x.loc[bl["DX_bl"] == g, col].mean() for g in DX_ORDER},
                         **{f"{g}_n": int(x.loc[bl["DX_bl"] == g, col].notna().sum()) for g in DX_ORDER}})
    out = pd.DataFrame(rows)
    out["q_fdr_anova"] = fdr(out["p_anova"])
    out["q_fdr_kruskal"] = fdr(out["p_kruskal"])
    return out, x.assign(DX_bl=bl["DX_bl"])


def fig_markers_by_diagnosis(x: pd.DataFrame, out):
    # Ordinal disease stage -> one hue, light to dark.
    shades = ["#9ec5f4", "#5a9be6", "#2a78d6", "#14498a"]
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.8))
    for ax, col in zip(axes.flat, BY_DX_MARKERS):
        for i, g in enumerate(DX_ORDER):
            d = x.loc[x["DX_bl"] == g, col].dropna()
            b = ax.boxplot(d, positions=[i], widths=0.6, patch_artist=True, showfliers=False,
                           medianprops={"color": plots.SURFACE, "linewidth": 1.5},
                           whiskerprops={"color": plots.TEXT_2}, capprops={"color": plots.TEXT_2})
            b["boxes"][0].set(facecolor=shades[i], edgecolor=shades[i])
        ax.set_xticks(range(len(DX_ORDER)), DX_ORDER)
        ax.set_title(col.replace("_", " "), fontsize=9)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Markers across baseline diagnosis (all phases)", fontsize=11, fontweight="bold", color=plots.TEXT)
    fig.tight_layout()
    plots.save(fig, out / "fig_markers_by_diagnosis")


def main(tasks=cfg.TASKS, merge=None, fs=None):
    plots.setup()
    merge = co.load_adnimerge() if merge is None else merge
    fs = co.load_fs7() if fs is None else fs
    for task in tasks:
        c, matched, _ = co.modeling_cohort(merge, fs, task)
        t = group_comparison(c, matched, task)
        t.to_csv(cfg.out_dir("analysis", task, "tables") / "group_comparison.csv", index=False)
        fig_effect_sizes(t, task, cfg.out_dir("analysis", task, "figures"))
        print(f"[{task}] {int(t.significant_q05.sum())} of {len(t)} features differ (FDR q < 0.05)")
    anova, x = anova_by_diagnosis(merge, fs)
    anova.to_csv(cfg.out_dir("analysis", "all_participants", "tables") / "anova_baseline_diagnosis.csv", index=False)
    fig_markers_by_diagnosis(x, cfg.out_dir("analysis", "all_participants", "figures"))
    print(f"[all participants] ANOVA on {len(x)} participants, {len(anova)} features")
