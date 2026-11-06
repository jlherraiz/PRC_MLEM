"""Dense float32 check of the survival table in prc.cu against the analytic survival function.

Reads the isotope coefficients and the table size from prc.cu and repeats the device lookup on the
CPU (linear interpolation of S on [0, cutoff]). Every annihilation probability is a difference
S(w_in) - S(w_out), so its error is at most twice the reported maximum absolute error of S. The
largest errors (~1e-5, 64Cu) sit in the first bin (w < 0.003 mm), where the curvature of S is
unbounded; elsewhere they are far smaller. This is a numerical regression check, not a proven bound; tests/test_projector.py separately
compares the CUDA operators with the analytic evaluation (--exact-range).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_model(path):
    source = path.read_text()
    table = source.split("static const Isotope kIsotopes[] = {", 1)[1].split("\n};", 1)[0]
    isotopes = {}
    for name, count, body in re.findall(r'\{"([^"]+)",\s*(\d+),\s*\{\{(.*?)\}\}\}', table, re.S):
        values = [float(x.removesuffix("f")) for x in re.findall(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)f?', body)]
        isotopes[name] = np.array(values, dtype=np.float32).reshape(int(count), 5)
    if not isotopes:
        raise ValueError("No isotope coefficients found in source")
    bins = int(re.search(r'kRangeBins = (\d+)', source).group(1))
    return source, isotopes, bins


def survival(branches, r, dtype):
    r = np.asarray(r, dtype=dtype)
    s = np.zeros_like(r)
    for w, a, b, c, f in np.asarray(branches, dtype=dtype):
        x, y = b * r, c * r
        s += w * np.exp(-((a * r) ** (dtype(1) + f) + x * x * np.sqrt(x) + y * y * y * np.sqrt(y)))
    return s


def check(isotopes, bins, samples):
    reports = []
    for name, br in isotopes.items():
        br = br.copy()  # as in main(): the smallest branch absorbs the rounding of the weights
        br[np.argmin(br[:, 0]), 0] += np.float32(1) - np.sum(br[:, 0], dtype=np.float32)
        lo, hi = np.float32(0), np.float32(1)  # range_cutoff(): 99.95% of the positrons
        while np.float32(1) - survival(br, hi, np.float32) < np.float32(0.9995):
            hi *= np.float32(2)
        for _ in range(64):
            mid = np.float32(0.5) * (lo + hi)
            if np.float32(1) - survival(br, mid, np.float32) < np.float32(0.9995):
                lo = mid
            else:
                hi = mid
        scale = np.float32(bins) / hi
        table = survival(br, np.arange(bins + 1, dtype=np.float32) / scale, np.float32)
        w = np.concatenate([np.linspace(0, hi, samples + 1, dtype=np.float32),
                            (np.arange(bins, dtype=np.float32) + np.float32(0.5)) / scale])
        u = w * scale
        i = np.minimum(u.astype(np.int32), bins - 1)
        alpha = np.minimum(u - i.astype(np.float32), np.float32(1))
        lookup = alpha * (table[i + 1] - table[i]) + table[i]
        error = np.abs(lookup.astype(np.float64) - survival(br, w, np.float64))
        row = dict(isotope=name, cutoff_mm=float(hi), max_abs_error=float(error.max()),
                   at_mm=float(w[np.argmax(error)]), samples=int(w.size))
        reports.append(row)
        print(json.dumps(row))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "prc.cu")
    parser.add_argument("--output", type=Path, default=ROOT / "results/nearfield/range_table_accuracy.json")
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--max-abs-error", type=float, default=2e-5,
                        help="limit for |S_table - S| (default 2e-5 for 4096 bins)")
    args = parser.parse_args()
    if args.samples < 10 or not np.isfinite(args.max_abs_error) or args.max_abs_error <= 0:
        parser.error("samples must be >= 10 and the error limit finite and positive")
    source, isotopes, bins = read_model(args.source)
    reports = check(isotopes, bins, args.samples)
    observed = max(row["max_abs_error"] for row in reports)
    passed = observed <= args.max_abs_error
    result = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "bins": bins,
              "description": "Empirical CPU float32 lookup error of S; not a proven global bound",
              "limit": args.max_abs_error, "observed_max_abs_error": observed, "passed": passed,
              "results": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not passed:
        raise SystemExit("Survival table error exceeds the requested tolerance")


if __name__ == "__main__":
    main()
