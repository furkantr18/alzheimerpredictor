"""Baseline feature sets.

    demo      age, sex, education, APOE4 count
    cog       baseline cognitive / functional tests. For CN_AD these are CIRCULAR (ADNI diagnosis is
              defined with MMSE, CDR and memory tests), so CN_AD results with 'cog' are a reference
              ceiling, not a finding.
    mri_merge the 6 ADNIMERGE FreeSurfer summary volumes / ICV. FreeSurfer version differs by phase
              (4.3 for ADNI1 1.5T, 5.1/6 later), so do not use them across phases.
    mri_fs7   FreeSurfer 7 regional volumes (/ICV) and cortical thickness, one version for all phases.

Only raw values are built here; imputation and scaling are fit inside each CV fold.
"""

from __future__ import annotations

import pandas as pd

DEMO = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4"]
COG = ["MMSE", "CDRSB", "ADAS13", "FAQ", "RAVLT_immediate", "RAVLT_learning",
       "RAVLT_perc_forgetting", "LDELTOTAL", "TRABSCOR"]
MERGE_VOLUMES = ["Hippocampus", "Entorhinal", "Ventricles", "WholeBrain", "Fusiform", "MidTemp"]

FS7_ICV = "ST10CV"
# Regions with mostly missing values or that are not brain tissue.
FS7_EXCLUDE = {"ST8SV", "ST68SV", FS7_ICV}
FS7_MAX_MISSING = 0.2


def demo(cohort: pd.DataFrame) -> pd.DataFrame:
    x = cohort[DEMO].copy()
    x["PTGENDER"] = (x["PTGENDER"] == "Female").astype(float)
    return x.astype(float)


def cog(cohort: pd.DataFrame) -> pd.DataFrame:
    return cohort[COG].apply(pd.to_numeric, errors="coerce")


def mri_merge(cohort: pd.DataFrame) -> pd.DataFrame:
    vol = cohort[MERGE_VOLUMES].apply(pd.to_numeric, errors="coerce")
    icv = pd.to_numeric(cohort["ICV"], errors="coerce")
    return vol.div(icv, axis=0).add_suffix("_icv")


def fs7_columns(fs: pd.DataFrame) -> list[str]:
    """Subcortical volumes (SV), cortical volumes (CV) and thickness averages (TA)."""
    cols = [c for c in fs.columns
            if c.startswith("ST") and c[-2:] in ("SV", "CV", "TA") and c not in FS7_EXCLUDE]
    return [c for c in cols if fs[c].isna().mean() <= FS7_MAX_MISSING]


def mri_fs7(cohort: pd.DataFrame, fs_matched: pd.DataFrame) -> pd.DataFrame:
    """FS7 features for the cohort; participants without a matched scan get NaN rows."""
    fs = fs_matched.reindex(cohort.index)
    cols = fs7_columns(fs_matched)
    x = fs[cols].apply(pd.to_numeric, errors="coerce")
    icv = pd.to_numeric(fs[FS7_ICV], errors="coerce")
    vol = [c for c in cols if c.endswith(("SV", "CV"))]
    x[vol] = x[vol].div(icv, axis=0)
    return x


def feature_sets(cohort: pd.DataFrame, fs_matched: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    d, c, m = demo(cohort), cog(cohort), mri_merge(cohort)
    sets = {
        "demo": d,
        "demo+mri_merge": d.join(m),
        "demo+cog": d.join(c),
        "demo+cog+mri_merge": d.join(c).join(m),
    }
    if fs_matched is not None:
        f = mri_fs7(cohort, fs_matched)
        sets["demo+mri_fs7"] = d.join(f)
        sets["demo+cog+mri_fs7"] = d.join(c).join(f)
    return sets


def fs7_names(data_dir=None) -> dict[str, str]:
    """Readable names for FreeSurfer 7 codes from the ADNI data dictionary (DATADIC), if present."""
    import adni_config as cfg
    try:
        path = cfg.find_file("*DATADIC*.csv", data_dir, exclude=None)
    except (FileNotFoundError, FileExistsError):
        return {}
    d = pd.read_csv(path, usecols=["TBLNAME", "FLDNAME", "TEXT"], low_memory=False)
    d = d[d["TBLNAME"] == "UCSFFSX7"]
    kinds = {"Subcortical Volume": "volume", "Cortical Volume": "cortical volume", "Thickness Average": "thickness"}
    names = {}
    for code, text in zip(d["FLDNAME"], d["TEXT"].astype(str)):
        for k, v in kinds.items():
            if text.startswith(k) and " of " in text:
                names[code] = f"{text.rsplit(' of ', 1)[1]} {v}"
    return names
