#!/usr/bin/env python3
"""
Test deblurring convergence of the fixed RL-TV iterative positron range correction.

Uses a Peneasy-style setup:
  - 68Ga isotope (main isotope in Peneasy comparison Case 2)
  - Heterogeneous density: water (1.0) / lung (0.26) slab interface
  - Two Gaussian blob sources spanning the interface
  - Measures RMSE and correlation vs truth at each iteration count
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FORWARD_BINARY = SCRIPT_DIR / "pr.x"
SOLVER_BINARY = SCRIPT_DIR / "pr_rl_hetero_cache.x"


def save_raw(path: Path, volume: np.ndarray) -> None:
    path.write_bytes(volume.astype(np.float32, copy=False).tobytes(order="C"))


def load_raw(path: Path, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = shape_xyz[0] * shape_xyz[1] * shape_xyz[2]
    if arr.size != expected:
        raise ValueError(f"{path} has {arr.size} floats, expected {expected}")
    return arr.reshape((shape_xyz[2], shape_xyz[1], shape_xyz[0]))


def compute_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float64).ravel()
    cand = candidate.astype(np.float64).ravel()
    diff = cand - ref
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ref_rms = float(np.sqrt(np.mean(ref * ref)))
    rel_l2 = rmse / ref_rms if ref_rms > 0 else 0.0
    corr = float(np.corrcoef(ref, cand)[0, 1]) if ref.std() > 0 and cand.std() > 0 else 0.0
    return {"rmse": rmse, "rel_l2": rel_l2, "corr": corr, "max_abs_diff": float(np.max(np.abs(diff)))}


def create_phantom(shape_xyz, interface_x, left_density, right_density):
    resx, resy, nzs = shape_xyz
    zz, yy, xx = np.indices((nzs, resy, resx), dtype=np.float32)

    # Two blobs: one in water, one straddling the interface into lung
    cx1, cy1, cz1 = resx * 0.3, resy * 0.4, nzs * 0.4
    cx2, cy2, cz2 = resx * 0.7, resy * 0.6, nzs * 0.5
    blob1 = 12.0 * np.exp(-((xx - cx1)**2 + (yy - cy1)**2 + (zz - cz1)**2) / (2.0 * 3.0**2))
    blob2 = 8.5 * np.exp(-((xx - cx2)**2 + (yy - cy2)**2 + (zz - cz2)**2) / (2.0 * 4.0**2))
    truth = (blob1 + blob2).astype(np.float32)
    truth[truth < 1e-3] = 0.0

    density = np.empty((nzs, resy, resx), dtype=np.float32)
    density[:, :, :interface_x] = left_density
    density[:, :, interface_x:] = right_density

    return truth, density


def run_forward(tmpdir, truth, density, shape_xyz, spacing_mm, isotope, nrays):
    save_raw(tmpdir / "ACTIVITY.raw", truth)
    save_raw(tmpdir / "DENSITY.raw", density)
    resx, resy, nzs = shape_xyz
    dx, dy, dz = spacing_mm
    cmd = [str(FORWARD_BINARY), isotope, str(resx), str(resy), str(nzs),
           str(dx), str(dy), str(dz), str(nrays)]
    proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, check=True)
    blurred = load_raw(tmpdir / "ACTIVITY_PRC.raw", shape_xyz)
    return blurred


def run_solver(tmpdir, blurred, density, shape_xyz, spacing_mm, isotope, nrays,
               iterations, tv_weight, tile_sources=256):
    save_raw(tmpdir / "ACTIVITY.raw", blurred)
    save_raw(tmpdir / "DENSITY.raw", density)
    resx, resy, nzs = shape_xyz
    dx, dy, dz = spacing_mm
    cmd = [str(SOLVER_BINARY), isotope, str(resx), str(resy), str(nzs),
           str(dx), str(dy), str(dz), str(nrays), str(iterations),
           str(tv_weight), str(tile_sources),
           "1e-3",   # tv_eps
           "1e-6",   # epsilon
           "0",      # verbose
           "0",      # support_eps
           "0"]      # entry_eps
    proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, check=True)
    corrected = load_raw(tmpdir / "ACTIVITY_PRC_RL_HETERO_CACHE.raw", shape_xyz)
    return corrected, proc.stdout


def main():
    # Peneasy-style parameters
    shape_xyz = (48, 48, 48)
    spacing_mm = (2.0, 2.0, 2.0)  # 2mm spacing (typical PET)
    isotope = "68Ga"
    nrays = 512
    tv_weight = 0.002
    interface_x = 24  # Half-way

    print("=" * 70)
    print("Deblurring Convergence Test — Peneasy-style setup")
    print("=" * 70)
    print(f"Isotope:    {isotope}")
    print(f"Grid:       {shape_xyz[0]}x{shape_xyz[1]}x{shape_xyz[2]}")
    print(f"Spacing:    {spacing_mm[0]}x{spacing_mm[1]}x{spacing_mm[2]} mm")
    print(f"Density:    water (1.0) / lung (0.26) slab at x={interface_x}")
    print(f"Rays:       {nrays}")
    print(f"TV weight:  {tv_weight}")
    print()

    truth, density = create_phantom(shape_xyz, interface_x, 1.0, 0.26)

    print(f"Truth: sum={truth.sum():.2f}, max={truth.max():.4f}, nonzero={np.count_nonzero(truth)}")
    print()

    # Step 1: Forward-project to get blurred image
    print("Step 1: Forward projection (blurring)...")
    with tempfile.TemporaryDirectory(prefix="deblur_fwd_") as tmpdir:
        blurred = run_forward(Path(tmpdir), truth, density, shape_xyz, spacing_mm, isotope, nrays)

    # Scale blurred to match truth sum (standard preprocessing)
    truth_sum = float(truth.sum(dtype=np.float64))
    blurred_sum = float(blurred.sum(dtype=np.float64))
    if blurred_sum > 0:
        blurred = blurred * np.float32(truth_sum / blurred_sum)
    blurred = np.maximum(blurred, 0.0).astype(np.float32)

    blurred_metrics = compute_metrics(truth, blurred)
    print(f"Blurred:  RMSE={blurred_metrics['rmse']:.6f}  "
          f"rel_L2={blurred_metrics['rel_l2']:.6f}  "
          f"corr={blurred_metrics['corr']:.6f}")
    print()

    # Step 2: Run RL-TV at multiple iteration counts
    iteration_counts = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20]
    print("Step 2: Iterative RL-TV correction")
    print("-" * 70)
    print(f"{'Iter':>5s}  {'RMSE':>10s}  {'rel_L2':>10s}  {'Corr':>10s}  {'max_diff':>10s}  {'Improved?':>10s}")
    print("-" * 70)

    # Print blurred baseline
    print(f"{'blur':>5s}  {blurred_metrics['rmse']:10.6f}  {blurred_metrics['rel_l2']:10.6f}  "
          f"{blurred_metrics['corr']:10.6f}  {blurred_metrics['max_abs_diff']:10.6f}  {'(baseline)':>10s}")

    results = []
    prev_rel_l2 = blurred_metrics["rel_l2"]

    for iters in iteration_counts:
        with tempfile.TemporaryDirectory(prefix=f"deblur_{iters}_") as tmpdir:
            corrected, stdout = run_solver(Path(tmpdir), blurred, density, shape_xyz,
                                           spacing_mm, isotope, nrays, iters, tv_weight)

        m = compute_metrics(truth, corrected)
        improved = "YES" if m["rel_l2"] < prev_rel_l2 else "no"
        print(f"{iters:5d}  {m['rmse']:10.6f}  {m['rel_l2']:10.6f}  "
              f"{m['corr']:10.6f}  {m['max_abs_diff']:10.6f}  {improved:>10s}")

        results.append({"iterations": iters, **m})
        prev_rel_l2 = m["rel_l2"]

    print("-" * 70)
    print()

    # Summary
    best = min(results, key=lambda r: r["rel_l2"])
    print("Summary:")
    print(f"  Blurred baseline:  rel_L2 = {blurred_metrics['rel_l2']:.6f}, corr = {blurred_metrics['corr']:.6f}")
    print(f"  Best iteration:    {best['iterations']} iters, rel_L2 = {best['rel_l2']:.6f}, corr = {best['corr']:.6f}")
    print(f"  Relative improvement: {(1 - best['rel_l2']/blurred_metrics['rel_l2'])*100:.1f}% reduction in rel_L2")
    print(f"  Correlation gain:     {best['corr'] - blurred_metrics['corr']:.6f}")
    print()

    if best["rel_l2"] < blurred_metrics["rel_l2"]:
        print("RESULT: The method successfully deblurs the image.")
        print(f"        rel_L2 decreased from {blurred_metrics['rel_l2']:.6f} to {best['rel_l2']:.6f}")
        print(f"        Correlation increased from {blurred_metrics['corr']:.6f} to {best['corr']:.6f}")
    else:
        print("RESULT: The method did NOT improve the image.")

    # Save results
    payload = {
        "setup": {
            "isotope": isotope, "shape_xyz": list(shape_xyz),
            "spacing_mm": list(spacing_mm), "nrays": nrays,
            "tv_weight": tv_weight, "density_left": 1.0, "density_right": 0.26,
        },
        "blurred_metrics": blurred_metrics,
        "iteration_results": results,
        "best": best,
    }
    output_path = SCRIPT_DIR / "deblurring_convergence_results.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nDetailed results saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
