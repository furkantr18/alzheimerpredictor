"""Unit tests for the ADNI tabular pipeline (stdlib unittest, synthetic rows only: no ADNI data).

Run from the repo root:  python -m unittest discover -s adni_predictor/tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ADNI = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ADNI))

import adni_config as cfg  # noqa: E402
import cohort as co  # noqa: E402
import evaluate as ev  # noqa: E402
import features as ft  # noqa: E402


def visits(rid, dx_bl, dx_by_month, origprot="ADNI1", examdate="2006-01-01"):
    """ADNIMERGE-like rows for one participant: {month: DX}."""
    rows = []
    for m, dx in dx_by_month.items():
        rows.append({"RID": rid, "VISCODE": "bl" if m == 0 else f"m{m:02d}", "DX_bl": dx_bl, "DX": dx, "M": m,
                     "ORIGPROT": origprot,
                     "EXAMDATE": pd.Timestamp(examdate) + pd.DateOffset(months=m)})
    return rows


class Labels(unittest.TestCase):
    def setUp(self):
        rows = []
        rows += visits(1, "LMCI", {0: "MCI", 12: "MCI", 24: "Dementia", 36: "Dementia"})   # pMCI
        rows += visits(2, "LMCI", {0: "MCI", 12: "MCI", 24: "MCI", 36: "MCI"})             # sMCI
        rows += visits(3, "EMCI", {0: "MCI", 24: "MCI", 48: "Dementia"})                   # late converter
        rows += visits(4, "LMCI", {0: "MCI", 12: "MCI", 24: "MCI"})                        # too short
        rows += visits(5, "LMCI", {0: "MCI", 12: "CN", 36: "MCI"})                         # reverter
        rows += visits(6, "EMCI", {0: "MCI", 6: None, 36: "MCI"})                          # missing DX ignored
        rows += visits(7, "CN", {0: "CN", 12: "CN"})
        rows += visits(8, "AD", {0: "Dementia"})
        rows += visits(9, "SMC", {0: "CN"})
        self.df = pd.DataFrame(rows)

    def test_mci_conversion_definition(self):
        y = co.mci_conversion_labels(self.df).to_dict()
        self.assertEqual(y, {1: 1, 2: 0, 6: 0})

    def test_cn_ad_excludes_smc_and_mci(self):
        c = co.build_cohort(self.df, "CN_AD")
        self.assertEqual(c["label"].to_dict(), {7: 0, 8: 1})

    def test_one_row_per_participant(self):
        for task in ("CN_AD", "sMCI_pMCI"):
            self.assertFalse(co.build_cohort(self.df, task).index.duplicated().any())


class ScanMatching(unittest.TestCase):
    def test_prefers_protocol_field_strength_then_closest_and_drops_failed_qc(self):
        cohort = pd.DataFrame({"EXAMDATE": pd.to_datetime(["2006-01-01", "2012-01-01", "2006-01-01"]),
                               "phase": ["ADNI1", "ADNI2", "ADNI1"]}, index=pd.Index([1, 2, 3], name="RID"))
        fs = pd.DataFrame({
            "RID":            [1, 1, 2, 2, 3, 3],
            "IMAGEUID":       [10, 11, 20, 21, 30, 31],
            "EXAMDATE":       pd.to_datetime(["2006-01-05", "2006-02-20", "2011-12-01", "2012-01-03",
                                              "2006-01-02", "2007-06-01"]),
            "FIELD_STRENGTH": ["3T", "1.5T", "3T", "3T", "1.5T", "1.5T"],
            "OVERALLQC":      [np.nan, "Pass", np.nan, "Fail", np.nan, np.nan],
        })
        m = co.match_baseline_scan(cohort, fs)
        self.assertEqual(m["IMAGEUID"].to_dict(), {1: 11, 2: 20, 3: 30})

    def test_scan_too_far_from_baseline_is_not_used(self):
        cohort = pd.DataFrame({"EXAMDATE": pd.to_datetime(["2006-01-01"]), "phase": ["ADNI1"]},
                              index=pd.Index([1], name="RID"))
        fs = pd.DataFrame({"RID": [1], "IMAGEUID": [1], "EXAMDATE": pd.to_datetime(["2007-01-01"]),
                           "FIELD_STRENGTH": ["1.5T"], "OVERALLQC": [np.nan]})
        self.assertTrue(co.match_baseline_scan(cohort, fs).empty)


class Features(unittest.TestCase):
    def test_fs7_volumes_are_icv_normalised_thickness_is_not(self):
        cohort = pd.DataFrame(index=pd.Index([1, 2], name="RID"))
        fs = pd.DataFrame({"ST29SV": [4000.0, 3000.0], "ST24TA": [3.0, 2.5], "ST24SA": [500.0, 400.0],
                           "ST8SV": [np.nan, np.nan], ft.FS7_ICV: [1.6e6, 1.5e6]}, index=[1, 2])
        x = ft.mri_fs7(cohort, fs)
        self.assertEqual(sorted(x.columns), ["ST24TA", "ST29SV"])
        self.assertAlmostEqual(x.loc[1, "ST29SV"], 4000 / 1.6e6)
        self.assertEqual(x.loc[2, "ST24TA"], 2.5)

    def test_participant_without_scan_gets_nan_row(self):
        cohort = pd.DataFrame(index=pd.Index([1, 2], name="RID"))
        fs = pd.DataFrame({"ST29SV": [4000.0], ft.FS7_ICV: [1.6e6]}, index=[1])
        self.assertTrue(ft.mri_fs7(cohort, fs).loc[2].isna().all())


class Evaluation(unittest.TestCase):
    def test_repeated_cv_on_separable_and_random_data(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
        y = pd.Series((X["a"] > 0).astype(int))
        good, _ = ev.repeated_cv(ev.models()["logreg"], X, y, n_repeats=2)
        self.assertEqual(len(good), 2)
        self.assertGreater(good["roc_auc"].mean(), 0.95)
        noise, _ = ev.repeated_cv(ev.models()["logreg"], X[["b", "c"]], y, n_repeats=2)
        self.assertLess(noise["roc_auc"].mean(), 0.7)


class Config(unittest.TestCase):
    def test_find_file_ignores_dictionary_and_handles_brackets(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "ADNIMERGE - Key ADNI tables merged into one table [ADNI1,GO,2,3]_23Oct2026.csv").touch()
            Path(d, "ADNIMERGE - Key ADNI tables merged into one table-Data Dictionary [X].csv").touch()
            self.assertIn("[ADNI1,GO,2,3]", cfg.find_file(cfg.ADNIMERGE_GLOB, Path(d)).name)

    def test_data_and_output_folders_are_git_ignored(self):
        local = (cfg.PROJECT_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("data/", local)
        self.assertIn("output/", local)
        self.assertIn("adni_data/", (cfg.REPO_ROOT / ".gitignore").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
