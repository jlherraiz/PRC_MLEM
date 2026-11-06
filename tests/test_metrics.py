"""CPU-only checks for reference-based image metrics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from evaluate import Evaluator, format_table, load


def phantom():
    truth = np.zeros((35, 35, 35), dtype=np.float32)
    truth[4:31, 4:31, 4:31] = 1
    truth[13:22, 13:22, 13:22] = 4
    return truth


class MetricsTests(unittest.TestCase):
    def test_identity_and_strict_json(self):
        truth = phantom()
        metrics = Evaluator(truth)(truth)
        for key in ("nrmse", "nrmse_tissue", "mse", "mae_tissue", "edge_nrmse", "ring", "overshoot",
                    "undershoot", "bg_bias", "bg_cov", "halo", "count_bias"):
            self.assertEqual(metrics[key], 0, key)
        for key in ("ssim", "ssim_tissue", "ssim_edge", "rc_hot", "rc_min"):
            self.assertAlmostEqual(metrics[key], 1, places=12, msg=key)
        self.assertIsNone(metrics["psnr"])
        json.dumps(metrics, allow_nan=False)

    def test_preserves_historical_global_metrics(self):
        truth = phantom()
        candidate = ndi.gaussian_filter(truth.astype(np.float64), sigma=0.7)
        evaluator = Evaluator(truth)
        metric = evaluator(candidate)
        expected = np.linalg.norm(candidate - truth) / np.linalg.norm(truth.astype(np.float64))
        self.assertAlmostEqual(metric["nrmse"], expected, places=12)
        expected_ssim = structural_similarity(truth[evaluator.roi].astype(np.float64),
                                             candidate[evaluator.roi], data_range=4,
                                             gaussian_weights=True, sigma=1.5,
                                             use_sample_covariance=False)
        self.assertAlmostEqual(metric["ssim"], expected_ssim, places=12)

    def test_blur_and_local_range_violations_are_distinct(self):
        truth = phantom()
        evaluator = Evaluator(truth)
        blur = evaluator(ndi.uniform_filter(truth, size=3))
        self.assertGreater(blur["edge_nrmse"], 0.05)
        self.assertLess(blur["ring"], 1e-7)
        ring = truth.copy()
        ring[13:15, 13:22, 13:22] += 0.5
        ring[11:13, 13:22, 13:22] = -0.2
        metric = evaluator(ring)
        self.assertGreater(metric["overshoot"], 0)
        self.assertGreater(metric["undershoot"], 0)
        self.assertAlmostEqual(metric["ring"] ** 2,
                               metric["overshoot"] ** 2 + metric["undershoot"] ** 2)

    def test_tissue_ssim_matches_cropped_map(self):
        truth = phantom()
        evaluator = Evaluator(truth)
        candidate = truth + np.random.default_rng(4).normal(0, 0.2, truth.shape) * (truth > 0)
        _, smap = structural_similarity(truth[evaluator.roi].astype(np.float64), candidate[evaluator.roi],
                                       data_range=4, gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=False, full=True)
        valid = (slice(5, -5),) * 3
        expected = smap[valid][evaluator.body[evaluator.roi][valid]].mean()
        self.assertAlmostEqual(evaluator(candidate)["ssim_tissue"], expected, places=12)

    def test_scale_invariant_scores(self):
        truth = phantom()
        candidate = ndi.gaussian_filter(truth, 0.8)
        a = Evaluator(truth)(candidate)
        for scale in (1e-4, 20, 1e4):
            with self.subTest(scale=scale):
                b = Evaluator(truth.astype(np.float64) * scale)(candidate.astype(np.float64) * scale)
                for key in ("nrmse", "ssim", "ssim_tissue", "ssim_edge", "ring", "halo",
                            "edge_nrmse", "bg_bias", "bg_cov", "rc_hot", "rc_min"):
                    self.assertAlmostEqual(a[key], b[key], places=9, msg=key)

    def test_no_lesions_or_edges_has_explicit_missing_metrics(self):
        truth = np.ones((15, 15, 15), dtype=np.float32)
        metric = Evaluator(truth)(truth)
        for key in ("rc_hot", "rc_min", "ring", "halo", "edge_nrmse", "ssim_edge"):
            self.assertIsNone(metric[key], key)
        self.assertEqual(metric["ssim"], 1)
        self.assertIn("—", format_table([("constant", metric)]))
        json.dumps(metric, allow_nan=False)

    def test_small_volume_does_not_fake_3d_ssim(self):
        truth = np.ones((5, 15, 15), dtype=np.float32)
        metric = Evaluator(truth)(truth)
        self.assertIsNone(metric["ssim"])
        self.assertEqual(metric["nrmse"], 0)

    def test_mask_can_include_cold_regions(self):
        truth = phantom()
        body = truth > 0
        truth[8:12, 8:12, 8:12] = 0
        candidate = truth.copy()
        candidate[8:12, 8:12, 8:12] = 1
        self.assertEqual(Evaluator(truth)(candidate)["nrmse_tissue"], 0)
        self.assertGreater(Evaluator(truth, mask=body)(candidate)["nrmse_tissue"], 0)

    def test_explicit_hot_mask_excludes_bright_background(self):
        truth = phantom()
        lesions = truth == 4
        truth[5:9, 5:9, 5:9] = 2  # bright background, unrelated to the lesion
        candidate = truth.copy()
        candidate[5:9, 5:9, 5:9] = 1
        threshold = Evaluator(truth, background=1)
        explicit = Evaluator(truth, background=1, hot_mask=lesions)
        self.assertLess(threshold(candidate)["rc_hot"], 1)
        self.assertEqual(explicit(candidate)["rc_hot"], 1)
        self.assertEqual(explicit(candidate)["rc_min"], 1)
        self.assertEqual(explicit.metric_settings["hot_region_count"], 1)
        self.assertEqual(threshold.metric_settings["hot_region_count"], 2)
        self.assertIn("explicit hot mask", explicit.metric_settings["hot_regions"])
        bad = lesions.copy()
        bad[0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "inside tissue"):
            Evaluator(truth, hot_mask=bad)
        with self.assertRaisesRegex(ValueError, "truth shape"):
            Evaluator(truth, hot_mask=lesions[:, :, :-1])

    def test_evaluate_cli_preserves_flat_json_and_reports_mask_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            truth = phantom()
            truth.tofile(data / "truth.raw")
            (truth > 1.5).astype(np.float32).tofile(data / "hot.raw")
            proc = subprocess.run([sys.executable, str(ROOT / "tools" / "evaluate.py"),
                                   "--truth", str(data / "truth.raw"), "--shape", "35", "35", "35",
                                   "--hot-mask", str(data / "hot.raw"), "--background", "1",
                                   "--json", str(data / "metrics.json"), f"identity={data / 'truth.raw'}"],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads((data / "metrics.json").read_text())
            self.assertEqual(set(report), {"identity"})
            self.assertEqual(report["identity"]["rc_hot"], 1)
            self.assertIn("explicit hot mask", proc.stdout)

    def test_rejects_invalid_inputs(self):
        truth = phantom()
        evaluator = Evaluator(truth)
        bad = truth.copy()
        bad[0, 0, 0] = np.nan
        for value in (bad, -truth, np.zeros_like(truth), np.ones((3, 3))):
            with self.assertRaises(ValueError):
                Evaluator(value)
        for kwargs in ({"data_range": 0}, {"data_range": np.nan}, {"background": -1},
                       {"mask": np.zeros_like(truth)}, {"mask": np.ones((3, 3, 3))}):
            with self.assertRaises(ValueError):
                Evaluator(truth, **kwargs)
        for value in (bad, truth[:, :, :-1]):
            with self.assertRaises(ValueError):
                evaluator(value)

    def test_raw_size_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.raw"
            np.zeros(7, dtype=np.float32).tofile(path)
            with self.assertRaisesRegex(ValueError, "expected 32 bytes, got 28"):
                load(path, (2, 2, 2))
            path.write_bytes(np.ones(8, dtype=np.float32).tobytes() + b"x")
            with self.assertRaisesRegex(ValueError, "expected 32 bytes, got 33"):
                load(path, (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
