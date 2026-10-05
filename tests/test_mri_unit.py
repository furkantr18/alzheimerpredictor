"""Unit tests for the MRI pipeline fixes (stdlib unittest, synthetic images only).

Run from the repo root:  .venv\\Scripts\\python.exe -m unittest tests.test_mri_unit -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

IMG = Path(__file__).resolve().parents[1] / "src" / "imgProcessing"
sys.path.insert(0, str(IMG))

import cv2  # noqa: E402


def ellipse(h=224, w=224, ry=90, rx=65, angle_deg=0.0, bg=0) -> np.ndarray:
    img = np.full((h, w), bg, np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (rx, ry), angle_deg, 0, 360, 200, -1)
    return img


def long_axis_tilt_from_vertical(img: np.ndarray) -> float:
    ys, xs = np.nonzero(img > 100)
    cov = np.cov(np.stack([xs, ys]).astype(float))
    w, v = np.linalg.eigh(cov)
    a = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
    return abs(((a - 90) + 90) % 180 - 90)


class RotationCorrection(unittest.TestCase):
    """B3 (numpy 2.x crash) and B4 (upright brains turned 90 degrees)."""

    def setUp(self):
        import preprocess_mri as pm
        self.pm = pm

    def test_no_crash_on_numpy2_and_upright_untouched(self):
        img = ellipse()
        out = self.pm.rotation_correction_from_mask(img, (img > 0).astype(np.uint8) * 255)
        self.assertTrue(np.array_equal(out, img), "an upright slice must not be rotated")

    def test_tilted_slice_is_straightened(self):
        img = ellipse(angle_deg=20)                       # long axis 20 degrees off vertical
        self.assertGreater(long_axis_tilt_from_vertical(img), 15)
        out = self.pm.rotation_correction_from_mask(img, (img > 0).astype(np.uint8) * 255)
        self.assertLess(long_axis_tilt_from_vertical(out), 2.0)

    def test_round_mask_is_left_alone(self):
        img = ellipse(ry=80, rx=79)
        out = self.pm.rotation_correction_from_mask(img, (img > 0).astype(np.uint8) * 255)
        self.assertTrue(np.array_equal(out, img))


class SharedPreprocessing(unittest.TestCase):
    def setUp(self):
        import mri_preprocess as mp
        self.mp = mp
        self.spec = mp.PreprocessSpec(size=224)

    def test_output_shape_dtype_and_determinism(self):
        img = ellipse(208, 176)
        a = self.mp.preprocess_gray(img, self.spec)
        b = self.mp.preprocess_gray(img.copy(), self.spec)
        self.assertEqual(a.shape, (224, 224)); self.assertEqual(a.dtype, np.uint8)
        self.assertTrue(np.array_equal(a, b))

    def test_grey_background_becomes_black(self):
        out = self.mp.preprocess_gray(ellipse(bg=60), self.spec)
        self.assertLess(int(np.median(out[:3])), 5)

    def test_rgb_and_bytes_input(self):
        rgb = cv2.cvtColor(ellipse(), cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".png", rgb)
        g = self.mp.decode_gray(buf.tobytes())
        self.assertEqual(g.ndim, 2)
        self.assertEqual(self.mp.preprocess_gray(rgb, self.spec).shape, (224, 224))

    def test_undecodable_raises_valueerror(self):
        with self.assertRaises(ValueError):
            self.mp.decode_gray(b"not an image")
        with self.assertRaises(ValueError):
            self.mp.decode_gray(b"")

    def test_tensor_is_3ch_imagenet_normalised(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        t = self.mp.to_tensor(np.zeros((224, 224), np.uint8))
        self.assertEqual(tuple(t.shape), (3, 224, 224))
        self.assertAlmostEqual(float(t[0, 0, 0]), -0.485 / 0.229, places=4)


class UnreadableImagesAreSkipped(unittest.TestCase):
    """B5: one bad file must be skipped and counted, not crash."""

    def test_cache_worker_returns_none_with_warning(self):
        import build_cache
        import mri_preprocess as mp
        build_cache._init(mp.PreprocessSpec(size=64).to_dict())
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.jpg"; bad.write_bytes(b"not really a jpeg")
            good = Path(d) / "good.png"; cv2.imwrite(str(good), ellipse())
            self.assertIsNone(build_cache._work(str(bad)))
            self.assertEqual(build_cache._work(str(good)).shape, (64, 64))

    def test_predict_skips_bad_file(self):
        import io
        from contextlib import redirect_stdout

        import predict_mri
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline
        import mri_config as C
        import mri_preprocess as mp
        pipe = Pipeline([("clf", DummyClassifier(strategy="prior"))]).fit(np.zeros((4, 64 * 64)), [0, 1, 2, 3])
        art = {"pipeline": pipe, "feature_size": 64, "class_names": C.CLASS_NAMES, "preprocess": mp.PreprocessSpec().to_dict()}
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.jpg"; bad.write_bytes(b"xx")
            good = Path(d) / "good.png"; cv2.imwrite(str(good), ellipse())
            buf = io.StringIO()
            with redirect_stdout(buf):
                out = predict_mri.predict_paths([bad, good], art)
        self.assertEqual(len(out), 1)
        self.assertIn("skipped unreadable", buf.getvalue())


class Labels(unittest.TestCase):
    def test_probabilities_follow_saved_class_order_even_if_a_class_is_missing(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from mri_model_trainer import predict_proba
        X = np.random.default_rng(0).normal(size=(60, 5))
        y = np.array([0, 2, 3] * 20)                      # class 1 (ModerateDemented) absent
        X[y == 3] += 3
        pipe = Pipeline([("clf", LogisticRegression())]).fit(X, y)
        p = predict_proba(pipe, X)
        self.assertEqual(p.shape[1], 4)
        self.assertTrue(np.all(p[:, 1] == 0))
        self.assertGreater(p[y == 3, 3].mean(), 0.5)

    def test_tail_sequences_are_relinked_by_continuity(self):
        """Regression test for the 2026-10-05 fix: tails whose numbering is shuffled must be matched back."""
        import subject_ids as S
        rng = np.random.default_rng(3)
        n, d = 12, 256
        start, step = rng.normal(size=(n, d)), rng.normal(size=(n, d))

        def unit(v):
            v = v - v.mean(1, keepdims=True)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        head_prev, head_last = unit(start + 0.95 * step), unit(start + 1.0 * step)       # slices k-1, k
        tail_first, tail_next = unit(start + 1.05 * step), unit(start + 1.1 * step)      # slices k+1, k+2
        perm = rng.permutation(n)                                                         # tail numbering shuffled
        a = S.match_tails(head_last, head_prev, tail_first[perm], tail_next[perm])
        self.assertTrue(np.array_equal(perm[a], np.arange(n)), "every head must get its own tail back")

    def test_subject_id_parsing(self):
        import subject_ids as S
        self.assertEqual(S.parse("MildDemented", "mildDem0"), (0, 0))
        self.assertEqual(S.parse("MildDemented", "mildDem29"), (1, 1))      # 28 subjects per slice row
        self.assertEqual(S.parse("NonDemented", "nonDem250"), (50, 2))
        self.assertEqual(S.parse("VeryMildDemented", "27 (43)"), (42, 27))
        self.assertEqual(S.parse("ModerateDemented", "28"), (0, 28))
        self.assertIsNone(S.parse("NonDemented", "0a1b2c3d-0000-0000-0000-000000000000"))


class Splits(unittest.TestCase):
    def test_units_never_shared_and_moderate_rule(self):
        import pandas as pd
        from make_splits import assign_units
        units = pd.DataFrame({"u": [f"M{i}" for i in range(2)] + [f"N{i}" for i in range(100)] + [f"V{i}" for i in range(70)],
                              "class_name": ["ModerateDemented"] * 2 + ["NonDemented"] * 100 + ["VeryMildDemented"] * 70})
        s = assign_units(units, "u", seed=1)
        self.assertEqual(len(s), len(units))
        self.assertEqual(sorted(s[f"M{i}"] for i in range(2)), ["test", "train"])
        n = pd.Series({k: v for k, v in s.items() if k.startswith("N")}).value_counts()
        self.assertEqual((n["train"], n["val"], n["test"]), (70, 15, 15))
        self.assertEqual(assign_units(units, "u", seed=1), s, "same seed -> same split")


class Augmentation(unittest.TestCase):
    def test_augment_batch_keeps_shape_and_range(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from mri_dl_trainer import augment_batch
        torch.manual_seed(0)
        x = torch.rand(8, 1, 64, 64)
        y = augment_batch(x.clone(), 1.0)
        self.assertEqual(tuple(y.shape), (8, 1, 64, 64))
        self.assertGreaterEqual(float(y.min()), 0.0); self.assertLessEqual(float(y.max()), 1.0)
        self.assertFalse(torch.allclose(y, x), "strength 1.0 must change the images")


if __name__ == "__main__":
    unittest.main(verbosity=2)
