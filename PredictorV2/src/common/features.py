"""Baseline feature sets.

    demo      age, sex, education, APOE4 count
    cog       baseline cognitive / functional tests. For CN_AD these are CIRCULAR (ADNI diagnosis is
              defined with MMSE, CDR and memory tests), so CN_AD results with 'cog' are a reference
              ceiling, not a finding.
    mri_merge the 6 ADNIMERGE FreeSurfer summary volumes / ICV. FreeSurfer version differs by phase
              (4.3 for ADNI1 1.5T, 5.1/6 later), so they are only used within ADNI1.
    mri_fs7   FreeSurfer 7 regional volumes (/ICV) and cortical thickness (~200 features).
    mri_eng   10 engineered FreeSurfer 7 markers from the AD imaging literature (see ENGINEERED).

Only raw values are built here; imputation and scaling are fit inside each CV fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import config as cfg

DEMO = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4"]
COG = ["MMSE", "CDRSB", "ADAS13", "FAQ", "RAVLT_immediate", "RAVLT_learning",
       "RAVLT_perc_forgetting", "LDELTOTAL", "TRABSCOR"]
MERGE_VOLUMES = ["Hippocampus", "Entorhinal", "Ventricles", "WholeBrain", "Fusiform", "MidTemp"]

FS7_ICV = "ST10CV"
# Regions with mostly missing values or that are not brain tissue.
FS7_EXCLUDE = {"ST8SV", "ST68SV", FS7_ICV}
FS7_MAX_MISSING = 0.2

# FreeSurfer 7 codes (left, right) used by the engineered markers.
HIPPOCAMPUS = ("ST29SV", "ST88SV")
INF_LAT_VENT = ("ST30SV", "ST89SV")
LAT_VENT = ("ST37SV", "ST96SV")
AMYGDALA = ("ST12SV", "ST71SV")
ENTORHINAL = ("ST24", "ST83")
PARAHIPPOCAMPAL = ("ST44", "ST103")
# Jack et al. 2015 (Alzheimer's & Dementia 11:757) AD-signature: surface-area weighted mean thickness.
AD_SIGNATURE = [("ST24", "ST83"), ("ST26", "ST85"), ("ST32", "ST91"), ("ST40", "ST99")]  # entorhinal,
#                                                         fusiform, inferior temporal, middle temporal
TOTAL_GM = "ST154SV"

ENGINEERED = {
    "hippocampus_icv": "total hippocampal volume / ICV",
    "hippocampus_asymmetry": "|L - R| / mean(L, R) of hippocampal volume",
    "hippocampal_occupancy": "hippocampus / (hippocampus + inferior lateral ventricle), Heister et al. 2011",
    "amygdala_icv": "total amygdala volume / ICV",
    "mtl_volume_icv": "hippocampus + amygdala + entorhinal + parahippocampal volume / ICV",
    "entorhinal_thickness": "mean entorhinal cortical thickness (mm)",
    "parahippocampal_thickness": "mean parahippocampal cortical thickness (mm)",
    "ad_signature_thickness": "area-weighted thickness of entorhinal, fusiform, inferior and middle temporal "
                              "cortex, Jack et al. 2015",
    "ventricles_icv": "lateral + inferior lateral ventricle volume / ICV",
    "total_gm_icv": "total grey matter volume / ICV",
}


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


def mri_engineered(cohort: pd.DataFrame, fs_matched: pd.DataFrame) -> pd.DataFrame:
    """The 10 literature-based markers of ENGINEERED (no fitting involved, so no leakage)."""
    fs = fs_matched.reindex(cohort.index)
    v = lambda c: pd.to_numeric(fs[c], errors="coerce") if c in fs else pd.Series(np.nan, index=fs.index)
    pair = lambda codes: v(codes[0]) + v(codes[1])
    icv = v(FS7_ICV)
    hip = pair(HIPPOCAMPUS)
    out = pd.DataFrame(index=cohort.index)
    out["hippocampus_icv"] = hip / icv
    out["hippocampus_asymmetry"] = (v(HIPPOCAMPUS[0]) - v(HIPPOCAMPUS[1])).abs() / (hip / 2)
    out["hippocampal_occupancy"] = hip / (hip + pair(INF_LAT_VENT))
    out["amygdala_icv"] = pair(AMYGDALA) / icv
    mtl = hip + pair(AMYGDALA) + pair(tuple(c + "CV" for c in ENTORHINAL)) \
        + pair(tuple(c + "CV" for c in PARAHIPPOCAMPAL))
    out["mtl_volume_icv"] = mtl / icv
    out["entorhinal_thickness"] = pair(tuple(c + "TA" for c in ENTORHINAL)) / 2
    out["parahippocampal_thickness"] = pair(tuple(c + "TA" for c in PARAHIPPOCAMPAL)) / 2
    num = sum(v(c + "TA") * v(c + "SA") for region in AD_SIGNATURE for c in region)
    den = sum(v(c + "SA") for region in AD_SIGNATURE for c in region)
    out["ad_signature_thickness"] = num / den
    out["ventricles_icv"] = (pair(LAT_VENT) + pair(INF_LAT_VENT)) / icv
    out["total_gm_icv"] = v(TOTAL_GM) / icv
    return out


def feature_sets(cohort: pd.DataFrame, fs_matched: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    d, c = demo(cohort), cog(cohort)
    sets = {"demo": d, "demo+cog": d.join(c), "demo+mri_merge": d.join(mri_merge(cohort))}
    if fs_matched is not None:
        e, f = mri_engineered(cohort, fs_matched), mri_fs7(cohort, fs_matched)
        sets["demo+mri_eng"] = d.join(e)
        sets["demo+mri_fs7"] = d.join(f)
        sets["demo+cog+mri_eng"] = d.join(c).join(e)
        sets["demo+cog+mri_fs7"] = d.join(c).join(f)
    return sets


def fs7_names(data_dir=None) -> dict[str, str]:
    """Readable names for FreeSurfer 7 codes from the ADNI data dictionary (DATADIC), if present."""
    try:
        path = cfg.find_file(cfg.DATADIC_GLOB, data_dir, exclude=None)
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
