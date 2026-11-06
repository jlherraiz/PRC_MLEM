"""GPU end-to-end numerical regression tests.

Build first with ``bash build.sh``; run ``python -m unittest discover -s tests``.
Set PRC_EXE to test another executable. Tests skip when CUDA is unavailable.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_range_table import read_model  # noqa: E402
from water_kernel import kernel  # noqa: E402


EXE = Path(os.environ.get("PRC_EXE", Path(__file__).resolve().parents[1] / "prc")).resolve()


def gpu_available():
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        return False
    if not EXE.is_file() or not shutil.which("nvidia-smi"):
        return False
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, timeout=10)
        return result.returncode == 0 and bool(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(gpu_available(), "build prc and provide a CUDA GPU to run numerical tests")
class ProjectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="prc_test_")
        cls.directory = Path(cls.workspace.name)
        cls.shape = (12, 14, 16)
        z, y, x = np.indices(cls.shape)
        body = ((x - 7.5) / 6) ** 2 + ((y - 6.5) / 5) ** 2 + ((z - 5.5) / 5) ** 2 < 1
        cavity = (x - 7.5) ** 2 + (y - 6.5) ** 2 + (z - 5.5) ** 2 < 3
        rng = np.random.default_rng(72)
        cls.rho = np.where(body & ~cavity, rng.uniform(0.2, 1.8, cls.shape), 0.001).astype(np.float32)
        cls.support = cls.rho >= 0.05
        cls.rho.tofile(cls.directory / "density.raw")
        cls.x = (rng.uniform(0.1, 2, cls.shape) * cls.support).astype(np.float32)
        cls.y = (rng.uniform(0.1, 2, cls.shape) * cls.support).astype(np.float32)
        cls.counter = 0

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def run_prc(self, image, mode="correct", *options, isotope="68Ga"):
        type(self).counter += 1
        stem = self.directory / str(self.counter)
        source, output = str(stem) + "_in.raw", str(stem) + "_out.raw"
        image.astype(np.float32).tofile(source)
        # Anisotropic voxels and heterogeneous tissue exercise both interpolation
        # and ray tracing, including transport across the central air cavity.
        command = [str(EXE), isotope, "16", "14", "12", "1.1", "1.6", "2.3",
                   "--input", source, "--density", str(self.directory / "density.raw"),
                   "--output", output, "--mode", mode, "--rays", "32", *options]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        corrected = np.fromfile(output, dtype=np.float32)
        self.assertEqual(corrected.size, image.size)
        self.assertTrue(np.isfinite(corrected).all())
        return corrected.reshape(self.shape)

    def test_projector_adjoint_and_normalization(self):
        ax = self.run_prc(self.x, "forward")
        aty = self.run_prc(self.y, "adjoint")
        lhs = np.sum(ax.astype(np.float64) * self.y)
        rhs = np.sum(self.x.astype(np.float64) * aty)
        self.assertAlmostEqual(lhs / rhs, 1.0, delta=3e-6)
        self.assertAlmostEqual(ax.sum(dtype=np.float64) / self.x.sum(dtype=np.float64), 1.0, delta=3e-6)
        np.testing.assert_array_equal(ax[~self.support], 0)
        at_one = self.run_prc(np.ones(self.shape, np.float32), "adjoint")
        np.testing.assert_allclose(at_one[self.support], 1, rtol=3e-6, atol=3e-6)
        np.testing.assert_array_equal(at_one[~self.support], 0)

    def test_cached_and_recomputed_corrections_agree(self):
        cached = self.run_prc(self.x, "correct", "--iterations", "3", "--beta", "0.1")
        uncached = self.run_prc(self.x, "correct", "--iterations", "3", "--beta", "0.1",
                                "--no-cache-near")
        np.testing.assert_allclose(cached, uncached, rtol=3e-5, atol=3e-6)
        exact = self.run_prc(self.x, "correct", "--iterations", "3", "--beta", "0.1",
                             "--no-cache-near", "--exact-range")
        np.testing.assert_allclose(cached, exact, rtol=3e-5, atol=3e-6)

    def test_range_table_matches_exact_all_isotopes(self):
        for isotope in ["11C", "13N", "15O", "18F", "64Cu", "68Ga", "82Rb", "124I"]:
            with self.subTest(isotope=isotope):
                ax = self.run_prc(self.x, "forward", isotope=isotope)
                exact_ax = self.run_prc(self.x, "forward", "--exact-range", isotope=isotope)
                aty = self.run_prc(self.y, "adjoint", isotope=isotope)
                exact_aty = self.run_prc(self.y, "adjoint", "--exact-range", isotope=isotope)
                np.testing.assert_allclose(ax, exact_ax, rtol=1e-5, atol=2e-7)
                np.testing.assert_allclose(aty, exact_aty, rtol=1e-5, atol=2e-7)
                lhs = np.sum(ax.astype(np.float64) * self.y)
                rhs = np.sum(self.x.astype(np.float64) * aty)
                self.assertAlmostEqual(lhs / rhs, 1.0, delta=3e-6)
                self.assertAlmostEqual(ax.sum(dtype=np.float64) / self.x.sum(dtype=np.float64),
                                       1.0, delta=3e-6)

    def test_water_kernel_matches_exact_voxel_average(self):
        # tools/water_kernel.py integrates the range model for a source spread uniformly over its
        # voxel. The earlier centre-emission, 2x2x2-quadrature near field missed it by 17% of the blur
        # at 4 mm and 4% at 2 mm; ray tracing from the whole voxel stays below 1%.
        branches = read_model(Path(__file__).resolve().parents[1] / "prc.cu")[1]["68Ga"]
        for size in (4.0, 2.0):
            with self.subTest(voxel_mm=size):
                half = int(np.ceil(10 / size)) + 1
                n, c = 2 * half + 5, half + 2
                delta = np.zeros((n, n, n), np.float32)
                delta[c, c, c] = 1
                stem = self.directory / f"water_{size}"
                delta.tofile(str(stem) + "_in.raw")
                np.ones((n, n, n), np.float32).tofile(str(stem) + "_rho.raw")
                command = [str(EXE), "68Ga", str(n), str(n), str(n), str(size), str(size), str(size),
                           "--mode", "forward", "--input", str(stem) + "_in.raw", "--density",
                           str(stem) + "_rho.raw", "--output", str(stem) + "_out.raw"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                P = np.fromfile(str(stem) + "_out.raw", np.float32).reshape(n, n, n)
                P = P[c - half:c + half + 1, c - half:c + half + 1, c - half:c + half + 1].astype(np.float64)
                exact = kernel(branches, (size,) * 3, half)
                blur = exact.copy()
                blur[half, half, half] -= 1
                self.assertLess(np.linalg.norm(P - exact) / np.linalg.norm(blur), 0.01)
                offsets = np.indices(P.shape) - half  # mirrored rays: no sub-voxel shift
                for axis in offsets:
                    self.assertLess(abs((P * axis).sum()), 1e-6)

    def test_sparse_observations_do_not_lock_supported_zeros(self):
        sparse = self.x.copy()
        rng = np.random.default_rng(21)
        sparse[rng.random(self.shape) < 0.5] = 0
        corrected = self.run_prc(sparse, "correct", "--iterations", "5", "--beta", "0.1")
        observed_zeros = (sparse == 0) & self.support
        self.assertGreater(np.mean(corrected[observed_zeros] > 0), 0.95)
        self.assertGreaterEqual(corrected.min(), 0)
        np.testing.assert_array_equal(corrected[~self.support], 0)
        self.assertAlmostEqual(corrected.sum(dtype=np.float64) / sparse.sum(dtype=np.float64),
                               1.0, delta=3e-6)

    def test_zero_iterations_preserves_supported_input_exactly(self):
        source = self.x.copy()
        source[5, 4, 5] = 0
        result = self.run_prc(source, "correct", "--iterations", "0")
        np.testing.assert_array_equal(result, source)

    @unittest.skipUnless(Path("/dev/full").exists(), "/dev/full is unavailable")
    def test_buffered_output_write_failure_is_reported(self):
        # A small write fits in stdio's buffer; the failure occurs at fclose.
        source, density = self.directory / "full_input.raw", self.directory / "full_density.raw"
        np.ones(8, np.float32).tofile(source)
        np.ones(8, np.float32).tofile(density)
        command = [str(EXE), "68Ga", "2", "2", "2", "1", "1", "1", "--mode", "forward",
                   "--input", str(source), "--density", str(density), "--output", "/dev/full",
                   "--rays", "32"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot write /dev/full", result.stderr)

    def test_overflowing_update_fails_without_writing_output(self):
        source = self.directory / "extreme_input.raw"
        density = self.directory / "extreme_density.raw"
        output = self.directory / "extreme_output.raw"
        np.full(8, 1e20, np.float32).tofile(source)
        np.ones(8, np.float32).tofile(density)
        command = [str(EXE), "68Ga", "2", "2", "2", "1", "1", "1",
                   "--input", str(source), "--density", str(density), "--output", str(output),
                   "--iterations", "1", "--no-accel", "--rays", "32"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Non-finite output", result.stderr)
        self.assertFalse(output.exists())

    def test_activity_unit_scaling(self):
        original = self.run_prc(self.x, "correct", "--iterations", "4", "--beta", "0.1")
        scaled = self.run_prc(self.x * 1000, "correct", "--iterations", "4", "--beta", "0.1") / 1000
        np.testing.assert_allclose(scaled, original, rtol=5e-5, atol=5e-6)

    def test_acceleration_warmup_matches_first_two_plain_steps(self):
        # Two residuals are needed before the third step can extrapolate.
        for iterations in ("1", "2"):
            with self.subTest(iterations=iterations):
                options = ("--iterations", iterations, "--beta", "0.2", "--gamma", "20")
                accelerated = self.run_prc(self.x, "correct", *options)
                plain = self.run_prc(self.x, "correct", "--no-accel", *options)
                np.testing.assert_allclose(accelerated, plain, rtol=3e-5, atol=3e-6)

    def test_unregularized_rl_conserves_counts_with_and_without_acceleration(self):
        for acceleration in ((), ("--no-accel",)):
            with self.subTest(options=acceleration):
                corrected = self.run_prc(self.x, "correct", "--iterations", "25", "--beta", "0",
                                         *acceleration)
                self.assertGreaterEqual(corrected.min(), 0)
                np.testing.assert_array_equal(corrected[~self.support], 0)
                self.assertAlmostEqual(corrected.sum(dtype=np.float64) / self.x.sum(dtype=np.float64),
                                       1.0, delta=5e-6)

    def test_long_sparse_cold_reconstruction_stays_finite_and_preserves_counts(self):
        truth = self.support.astype(np.float32)
        cold = np.zeros(self.shape, dtype=bool)
        cold[4:8, 3:7, 3:7] = True
        cold &= self.support
        truth[cold] = 0
        blurred = self.run_prc(truth, "forward")
        sparse = (np.random.default_rng(81).poisson(2 * blurred) / 2).astype(np.float32)
        missed_tissue = (sparse == 0) & self.support & ~cold
        self.assertGreater(missed_tissue.sum(), 20)
        for acceleration in ((), ("--no-accel",)):
            with self.subTest(options=acceleration):
                corrected = self.run_prc(sparse, "correct", "--iterations", "30", "--beta", "0.2",
                                         "--gamma", "20", *acceleration)
                self.assertGreaterEqual(corrected.min(), 0)
                np.testing.assert_array_equal(corrected[~self.support], 0)
                self.assertGreater(np.mean(corrected[missed_tissue] > 0), 0.95)
                self.assertLess(corrected[cold].mean(), corrected[self.support & ~cold].mean())
                self.assertAlmostEqual(corrected.sum(dtype=np.float64) / sparse.sum(dtype=np.float64),
                                       1.0, delta=5e-6)

    def test_acceleration_approaches_a_long_plain_reference_sooner(self):
        blurred = self.run_prc(self.x, "forward")
        options = ("--beta", "0.2", "--gamma", "20")
        plain = self.run_prc(blurred, "correct", "--iterations", "400", "--no-accel", *options)
        accelerated = self.run_prc(blurred, "correct", "--iterations", "100", *options)
        plain_short = self.run_prc(blurred, "correct", "--iterations", "100", "--no-accel", *options)

        def distance(image):
            return np.linalg.norm(image - plain) / np.linalg.norm(plain)

        self.assertLess(distance(accelerated), 1e-3)
        self.assertLess(distance(accelerated), 0.5 * distance(plain_short))
        self.assertAlmostEqual(accelerated.sum(dtype=np.float64) / blurred.sum(dtype=np.float64), 1.0,
                               delta=3e-6)

    def test_checkpoint_and_reprojection_names_in_dotted_directory(self):
        dotted = self.directory / "dir.with.dots"
        dotted.mkdir()
        source = self.directory / "dotted_in.raw"
        self.x.tofile(source)
        command = [str(EXE), "68Ga", "16", "14", "12", "1.1", "1.6", "2.3", "--input", str(source),
                   "--density", str(self.directory / "density.raw"), "--output", str(dotted / "output"),
                   "--rays", "32", "--iterations", "3", "--save-every", "1", "--reproject"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ["output", "output_it001", "output_it002", "output_REPROJ"]:
            self.assertEqual((dotted / name).stat().st_size, self.x.nbytes)


if __name__ == "__main__":
    unittest.main()
