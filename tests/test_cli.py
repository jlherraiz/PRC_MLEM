"""Input validation and zero-data paths; these tests deliberately hide every GPU."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("PRC_EXE", os.environ.get("PRC_BINARY", ROOT / "prc"))).resolve()


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.density = self.directory / "density.raw"
        self.input = self.directory / "input.raw"
        self.output = self.directory / "output.raw"
        self.write_raw(self.density, [1.0] * 8)
        self.write_raw(self.input, [0.0] * 8)

    @staticmethod
    def write_raw(path, values):
        path.write_bytes(struct.pack(f"={len(values)}f", *values))

    def run_prc(self, *options, geometry=None):
        args = geometry or ["68Ga", "2", "2", "2", "1", "1", "1"]
        return subprocess.run(
            [str(BINARY), *args, "--density", str(self.density),
             "--input", str(self.input), "--output", str(self.output), *options],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            text=True, capture_output=True, timeout=10,
        )

    def assert_rejected(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, result.stderr)
        self.assertNotIn("CUDA error", result.stderr)
        self.assertFalse(self.output.exists())

    def test_invalid_numeric_options(self):
        cases = [
            ("--beta", "garbage"), ("--beta", "nan"), ("--gamma", "inf"),
            ("--rays", "128x"), ("--rays", "2147483648"),
            ("--iterations", "2.5"), ("--beta", "1e999"),
        ]
        for option, value in cases:
            with self.subTest(option=option, value=value):
                self.assert_rejected(self.run_prc(option, value), "Invalid")

    def test_invalid_bounds(self):
        cases = [
            ("--support-rho", "-1"), ("--air-cap-mm", "0"),
            ("--save-every", "-1"), ("--iterations", "2147483647"),
            ("--rays", "2147483647"),
            ("--rays", "1"), ("--rays", "3"), ("--rays", "129"),
        ]
        for option, value in cases:
            with self.subTest(option=option):
                self.assert_rejected(self.run_prc(option, value), "Invalid")

    def test_invalid_geometry_and_index_overflow(self):
        for geometry in [
            ["68Ga", "2x", "2", "2", "1", "1", "1"],
            ["68Ga", "2", "2", "2", "nan", "1", "1"],
            ["68Ga", "0", "2", "2", "1", "1", "1"],
        ]:
            with self.subTest(geometry=geometry):
                self.assert_rejected(self.run_prc(geometry=geometry), "Invalid")
        self.assert_rejected(
            self.run_prc(geometry=["68Ga", "65536", "65536", "1", "1", "1", "1"]),
            "INT_MAX",
        )

    def test_raw_size_must_match_exactly(self):
        for size in [7, 9]:
            with self.subTest(size=size):
                self.write_raw(self.input, [0.0] * size)
                self.assert_rejected(self.run_prc(), "Expected exactly")
        self.input.write_bytes(b"\x00" * 33)
        self.assert_rejected(self.run_prc(), "Expected exactly")

    def test_nonfinite_raw_values(self):
        for path in [self.input, self.density]:
            for value in [float("nan"), float("inf"), float("-inf")]:
                with self.subTest(path=path.name, value=value):
                    self.write_raw(path, [value] + [1.0] * 7)
                    self.assert_rejected(self.run_prc(), "Non-finite")
            self.write_raw(path, [1.0] * 8)

    def test_zero_activity_is_rejected_without_gpu(self):
        for mode in ["correct", "forward", "adjoint"]:
            with self.subTest(mode=mode):
                self.assert_rejected(self.run_prc("--mode", mode), "No activity")

    def test_empty_support_is_rejected_without_gpu(self):
        self.write_raw(self.density, [0.0] * 8)
        self.write_raw(self.input, [3.0] * 8)
        self.assert_rejected(self.run_prc(), "No activity")


if __name__ == "__main__":
    unittest.main()
