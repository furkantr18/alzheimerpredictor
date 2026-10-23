"""Unit tests for the ADNI tabular pipeline (stdlib unittest, synthetic rows only: no ADNI data).

Run from the repo root:  python -m unittest discover -s PredictorV2/tests -v
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

from common import cohort as co  # noqa: E402
from common import config as cfg  # noqa: E402
from common import features as ft  # noqa: E402
from data_analysis import statistics as st  # noqa: E402
from explainability import explain as xp  # noqa: E402
from training import evaluate as ev  # noqa: E402
from training.models import SEARCH_SPACES, models  # noqa: E402


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
        good, oof, _, _ = ev.repeated_cv(models()["logreg"], X, y, n_repeats=2)
        self.assertEqual(len(good), 2)
        self.assertFalse(np.isnan(oof).any())
        self.assertGreater(good["roc_auc"].mean(), 0.95)
        noise, _, _, _ = ev.repeated_cv(models()["logreg"], X[["b", "c"]], y, n_repeats=2)
        self.assertLess(noise["roc_auc"].mean(), 0.7)

    def test_nested_tuning_saves_params_per_outer_fold(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(90, 3)), columns=list("abc"))
        y = pd.Series((X["a"] + 0.3 * rng.normal(size=90) > 0).astype(int))
        _, _, params, _ = ev.repeated_cv(models()["logreg"], X, y, n_repeats=1, tune="logreg")
        self.assertEqual(len(params), cfg.N_SPLITS)
        self.assertIn("clf__C", params.columns)

    def test_anova_selection_is_fit_inside_each_fold(self):
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.normal(size=(150, 6)), columns=list("abcdef"))
        y = pd.Series((X["a"] + 0.5 * rng.normal(size=150) > 0).astype(int))
        _, _, _, freq = ev.repeated_cv(models(select=True)["logreg"], X, y, n_repeats=1)
        self.assertEqual(freq["a"], 1.0)                     # the informative feature is always kept
        self.assertLess(freq.drop("a").mean(), 0.5)          # noise features mostly dropped

    def test_selector_keeps_at_least_one_feature(self):
        from training.models import AnovaFdrSelector
        rng = np.random.default_rng(3)
        sel = AnovaFdrSelector().fit(rng.normal(size=(40, 4)), np.r_[np.zeros(20), np.ones(20)])
        self.assertGreaterEqual(sel.get_support().sum(), 1)

    def test_selection_in_pipeline_never_sees_test_rows(self):
        """Fitting on train rows only must give the same selection as fitting the selector on them alone."""
        from training.models import AnovaFdrSelector
        rng = np.random.default_rng(4)
        X = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
        y = pd.Series((X["b"] > 0).astype(int))
        m, _ = ev.fit(models(select=True)["logreg"], X.iloc[:70], y.iloc[:70])
        alone = AnovaFdrSelector().fit(X.iloc[:70].values, y.iloc[:70])
        np.testing.assert_array_equal(m.named_steps["select"].get_support(), alone.get_support())

    def test_metrics_are_consistent(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.6, 0.2, 0.9, 0.4, 0.8])
        m = ev.metrics(y, p)
        self.assertAlmostEqual(m["fpr"], 1 / 3)
        self.assertAlmostEqual(m["fnr"], 1 / 3)
        self.assertAlmostEqual(m["recall"], 1 - m["fnr"])
        self.assertAlmostEqual(m["specificity"], 1 - m["fpr"])
        self.assertTrue(set(ev.METRICS) <= set(m))

    def test_every_model_has_a_search_space(self):
        for name, model in models().items():
            for key in SEARCH_SPACES[name]:
                self.assertIn(key, model.get_params(), f"{name}: {key}")


class ParamTables(unittest.TestCase):
    def test_none_is_kept_and_foreign_params_dropped(self):
        from training import run_models as rm
        folds = pd.DataFrame({"features": "s", "model": ["rf", "rf", "logreg"], "repeat": 0, "fold": [0, 1, 0],
                              "clf__max_depth": [None, 5, None], "clf__C": [None, None, 0.5]})
        with tempfile.TemporaryDirectory() as d:
            rm.write_param_tables(folds, Path(d))
            long = pd.read_csv(Path(d) / "best_params_per_outer_fold.csv", keep_default_na=False)
            summary = pd.read_csv(Path(d) / "best_params_summary.csv")
        self.assertEqual(len(long), 3)                       # rf max_depth x2, logreg C x1
        self.assertEqual(sorted(long["value"].astype(str)), ["0.5", "5.0", "None"])
        self.assertEqual(int(summary.set_index(["model", "parameter"]).loc[("rf", "max_depth"), "n_folds"]), 2)


class Engineered(unittest.TestCase):
    def test_hippocampal_markers(self):
        cohort = pd.DataFrame(index=pd.Index([1], name="RID"))
        fs = pd.DataFrame({"ST29SV": [3000.0], "ST88SV": [3200.0], "ST30SV": [400.0], "ST89SV": [400.0],
                           ft.FS7_ICV: [1.5e6]}, index=[1])
        e = ft.mri_engineered(cohort, fs).loc[1]
        self.assertAlmostEqual(e["hippocampus_icv"], 6200 / 1.5e6)
        self.assertAlmostEqual(e["hippocampal_occupancy"], 6200 / 7000)
        self.assertAlmostEqual(e["hippocampus_asymmetry"], 200 / 3100)
        self.assertTrue(np.isnan(e["entorhinal_thickness"]))      # missing columns give NaN, no crash

    def test_ad_signature_is_area_weighted(self):
        cohort = pd.DataFrame(index=pd.Index([1], name="RID"))
        cols = {}
        for k, region in enumerate(ft.AD_SIGNATURE):
            for c in region:
                cols[c + "TA"], cols[c + "SA"] = [2.0 + k], [100.0 * (k + 1)]
        e = ft.mri_engineered(cohort, pd.DataFrame(cols, index=[1])).loc[1]
        expected = sum((2.0 + k) * 100 * (k + 1) for k in range(4)) / sum(100 * (k + 1) for k in range(4))
        self.assertAlmostEqual(e["ad_signature_thickness"], expected)


class Statistics(unittest.TestCase):
    def test_fdr_and_effect_size(self):
        q = st.fdr([0.01, 0.04, 0.03, np.nan])
        self.assertTrue(np.isnan(q[3]))
        self.assertTrue(np.all(q[:3] >= [0.01, 0.04, 0.03]))
        a = pd.Series(np.arange(10.0))
        self.assertAlmostEqual(st.hedges_g(a, a), 0.0)
        self.assertGreater(st.hedges_g(a, a + 5), 0)


class Explain(unittest.TestCase):
    def test_pick_cases(self):
        y = pd.Series([0, 0, 1, 1, 1])
        p = np.array([0.1, 0.95, 0.9, 0.6, 0.2])
        self.assertEqual(xp.pick_cases(y, p), {"correct_positive": 2, "correct_negative": 0,
                                               "most_confident_error": 1})

    def test_icv_ratio_detection(self):
        self.assertTrue(xp.is_icv_ratio("hippocampus_icv"))
        self.assertTrue(xp.is_icv_ratio("ST29SV"))
        self.assertFalse(xp.is_icv_ratio("ST24TA"))
        self.assertFalse(xp.is_icv_ratio("AGE"))

    def test_rebuild_applies_saved_params(self):
        m = xp.rebuild("rf", '{"n_estimators": 50, "max_depth": "None", "unknown": 1}')
        self.assertEqual(m.named_steps["clf"].n_estimators, 50)
        self.assertIsNone(m.named_steps["clf"].max_depth)


class Config(unittest.TestCase):
    def test_find_file_ignores_dictionary_and_handles_brackets(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "ADNIMERGE - Key ADNI tables merged into one table [ADNI1,GO,2,3]_23Oct2026.csv").touch()
            Path(d, "ADNIMERGE - Key ADNI tables merged into one table-Data Dictionary [X].csv").touch()
            self.assertIn("[ADNI1,GO,2,3]", cfg.find_file(cfg.ADNIMERGE_GLOB, Path(d)).name)

    def test_out_dir_uses_numbered_stage_folders(self):
        self.assertEqual(cfg.out_dir("models", "CN_AD").relative_to(cfg.OUTPUT_DIR).parts[0], "2_model_comparison")

    def test_data_and_output_folders_are_git_ignored(self):
        local = (cfg.PROJECT_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("data/", local)
        self.assertIn("output/", local)
        self.assertIn("adni_data/", (cfg.REPO_ROOT / ".gitignore").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
