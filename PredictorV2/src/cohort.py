"""Baseline cohorts and labels from ADNIMERGE, plus the matching FreeSurfer 7 baseline scan.

One row per participant (RID), so any split on rows is a subject-level split.

Tasks
    CN_AD       DX_bl CN (0) vs AD (1). SMC is excluded.
    sMCI_pMCI   baseline EMCI/LMCI. pMCI (1): first Dementia diagnosis within CONVERSION_MONTHS.
                sMCI (0): MCI at every diagnosed visit and followed for >= CONVERSION_MONTHS.
                Late converters and MCI->CN reverters are excluded (definition of Wen et al. 2020).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import adni_config as cfg

PHASE_ORDER = ["ADNI1", "ADNIGO", "ADNI2", "ADNI3", "ADNI4"]


def load_adnimerge(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or cfg.find_file(cfg.ADNIMERGE_GLOB), low_memory=False)
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df["M"] = pd.to_numeric(df["M"], errors="coerce")
    return df


def baseline(df: pd.DataFrame) -> pd.DataFrame:
    bl = df[df["VISCODE"] == "bl"].copy()
    if bl["RID"].duplicated().any():
        raise ValueError("ADNIMERGE has more than one baseline row for some RIDs")
    return bl.set_index("RID", drop=False)


def mci_conversion_labels(df: pd.DataFrame, months: int = cfg.CONVERSION_MONTHS) -> pd.Series:
    """Return 1 (pMCI) / 0 (sMCI) per RID for baseline-MCI participants; others are left out."""
    bl = baseline(df)
    mci_rids = bl.index[bl["DX_bl"].isin(["EMCI", "LMCI"])]
    visits = df[df["RID"].isin(mci_rids) & df["DX"].notna()]

    labels = {}
    for rid, v in visits.groupby("RID"):
        dem = v.loc[v["DX"] == "Dementia", "M"]
        if len(dem):
            if dem.min() <= months:
                labels[rid] = 1
            continue                                   # converts later: excluded
        if (v["DX"] == "MCI").all() and v["M"].max() >= months:
            labels[rid] = 0
    return pd.Series(labels, name="label", dtype=int)


def build_cohort(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Baseline rows of the task's participants with a 'label' column and a 'phase' column (ORIGPROT)."""
    bl = baseline(df)
    if task == "CN_AD":
        bl = bl[bl["DX_bl"].isin(["CN", "AD"])].copy()
        bl["label"] = (bl["DX_bl"] == "AD").astype(int)
    elif task == "sMCI_pMCI":
        y = mci_conversion_labels(df)
        bl = bl.loc[y.index].copy()
        bl["label"] = y
    else:
        raise ValueError(f"unknown task {task!r}")
    bl["phase"] = bl["ORIGPROT"]
    return bl.sort_index()


def load_fs7(path=None) -> pd.DataFrame:
    fs = pd.read_csv(path or cfg.find_file(cfg.FS7_GLOB), low_memory=False)
    fs["EXAMDATE"] = pd.to_datetime(fs["EXAMDATE"], errors="coerce")
    return fs


def match_baseline_scan(cohort: pd.DataFrame, fs: pd.DataFrame,
                        max_gap_days: int = cfg.MAX_SCAN_GAP_DAYS) -> pd.DataFrame:
    """Pick one FreeSurfer 7 row per participant: the scan closest to the baseline exam.

    Scans rated OVERALLQC == 'Fail' are dropped. ADNI1 participants prefer 1.5T (the ADNI1 standard
    protocol; only a subset also had 3T), later phases prefer 3T. Returns FS rows indexed by RID.
    """
    fs = fs[fs["RID"].isin(cohort.index) & (fs["OVERALLQC"] != "Fail")].copy()
    exam = cohort["EXAMDATE"]
    fs["gap_days"] = (fs["EXAMDATE"] - fs["RID"].map(exam)).dt.days.abs()
    fs = fs[fs["gap_days"] <= max_gap_days]
    preferred = np.where(fs["RID"].map(cohort["phase"]) == "ADNI1", "1.5T", "3T")
    fs["field_mismatch"] = (fs["FIELD_STRENGTH"] != preferred).astype(int)
    fs = fs.sort_values(["RID", "field_mismatch", "gap_days", "IMAGEUID"])
    return fs.drop_duplicates("RID").set_index("RID")


def modeling_cohort(merge: pd.DataFrame, fs: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Task cohort restricted to participants with a matched FreeSurfer 7 baseline scan, so every
    feature set is evaluated on the same people. Returns (cohort, matched scans, size before matching)."""
    c = build_cohort(merge, task)
    matched = match_baseline_scan(c, fs)
    return c.loc[c.index.intersection(matched.index)], matched, len(c)
