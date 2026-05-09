#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from rl_positron_correction import compute_metrics


SUPPORTED_ISOTOPES = ["11C", "13N", "15O", "18F", "64Cu", "68Ga", "82Rb", "124I"]

TIME_RE = re.compile(r"^([A-Z_]+)\s*=\s*([-+0-9.eE]+)\s*$")
CACHE_RE = re.compile(
    r"stored_rows = (?P<stored_rows>\d+), total_nnz = (?P<total_nnz>\d+), cache_mb = (?P<cache_mb>[-+0-9.eE]+)"
)
ACTIVE_RE = re.compile(r"active_sources = (?P<active>\d+)/(?P<total>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse cached heterogeneous RL-TV per isotope on a shared synthetic case."
    )
    parser.add_argument("--output", default="benchmark_rl_hetero_cache_isotopes.json")
    parser.add_argument("--forward-binary", default=str(Path(__file__).resolve().parent / "pr.x"))
    parser.add_argument("--solver-binary", default=str(Path(__file__).resolve().parent / "pr_rl_hetero_cache.x"))
    parser.add_argument("--truth", help="Optional float32 RAW truth / input activity volume.")
    parser.add_argument("--observed", help="Optional float32 RAW blurred / measured volume.")
    parser.add_argument("--density-file", help="Optional float32 RAW density volume.")
    parser.add_argument(
        "--preserve-observed-sum",
        action="store_true",
        help="Do not rescale the external observed image to the truth sum before deconvolution.",
    )
    parser.add_argument("--resx", type=int, default=24)
    parser.add_argument("--resy", type=int, default=24)
    parser.add_argument("--nzs", type=int, default=24)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dy", type=float, default=1.0)
    parser.add_argument("--dz", type=float, default=1.0)
    parser.add_argument("--nrays", type=int, default=256)
    parser.add_argument("--iteration-grid", nargs="+", type=int, default=[4, 8, 12, 16, 20])
    parser.add_argument("--tv-weight", type=float, default=0.002)
    parser.add_argument("--tv-eps", type=float, default=1.0e-3)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--tile-sources", type=int, default=256)
    parser.add_argument("--support-eps", type=float, default=0.0)
    parser.add_argument("--entry-eps", type=float, default=0.0)
    parser.add_argument("--left-density", type=float, default=1.0)
    parser.add_argument("--right-density", type=float, default=0.35)
    parser.add_argument("--isotopes", nargs="+", default=SUPPORTED_ISOTOPES)
    return parser.parse_args()


def save_raw(path: Path, volume: np.ndarray) -> None:
    path.write_bytes(volume.astype(np.float32, copy=False).tobytes(order="C"))


def load_raw(path: Path, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = shape_xyz[0] * shape_xyz[1] * shape_xyz[2]
    if arr.size != expected:
        raise ValueError(f"{path} has {arr.size} floats, expected {expected}")
    return arr.reshape((shape_xyz[2], shape_xyz[1], shape_xyz[0]))


def scaled_blobs(shape_xyz: tuple[int, int, int]) -> np.ndarray:
    resx, resy, nzs = shape_xyz
    zz, yy, xx = np.indices((nzs, resy, resx), dtype=np.float32)

    def coord(frac: float, size: int) -> float:
        return frac * float(size - 1)

    blob1 = 12.0 * np.exp(
        -(
            (xx - coord(15.0 / 47.0, resx)) ** 2
            + (yy - coord(18.0 / 47.0, resy)) ** 2
            + (zz - coord(20.0 / 47.0, nzs)) ** 2
        )
        / (2.0 * (max(1.5, 3.0 * resx / 48.0) ** 2))
    )
    blob2 = 8.5 * np.exp(
        -(
            (xx - coord(33.0 / 47.0, resx)) ** 2
            + (yy - coord(28.0 / 47.0, resy)) ** 2
            + (zz - coord(24.0 / 47.0, nzs)) ** 2
        )
        / (2.0 * (max(2.0, 4.0 * resx / 48.0) ** 2))
    )
    truth = (blob1 + blob2).astype(np.float32)
    truth[truth < 1.0e-3] = 0.0
    return truth


def slab_density(shape_xyz: tuple[int, int, int], left_density: float, right_density: float) -> np.ndarray:
    resx, resy, nzs = shape_xyz
    plane_x = resx // 2
    density = np.empty((nzs, resy, resx), dtype=np.float32)
    density[:, :, :plane_x] = left_density
    density[:, :, plane_x:] = right_density
    return density


def match_sum_to_reference(candidate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    candidate_sum = float(candidate.sum(dtype=np.float64))
    reference_sum = float(reference.sum(dtype=np.float64))
    if candidate_sum <= 0.0 or reference_sum <= 0.0:
        return candidate.astype(np.float32, copy=False), {
            "applied": 0.0,
            "scale_factor": 1.0,
            "candidate_sum_before": candidate_sum,
            "candidate_sum_after": candidate_sum,
            "reference_sum": reference_sum,
        }

    scale_factor = reference_sum / candidate_sum
    scaled = candidate.astype(np.float32, copy=False) * np.float32(scale_factor)
    return scaled.astype(np.float32, copy=False), {
        "applied": 1.0,
        "scale_factor": scale_factor,
        "candidate_sum_before": candidate_sum,
        "candidate_sum_after": float(scaled.sum(dtype=np.float64)),
        "reference_sum": reference_sum,
    }


def parse_solver_stdout(stdout: str) -> dict[str, float | int]:
    parsed: dict[str, float | int] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        match = TIME_RE.match(stripped)
        if match:
            parsed[match.group(1)] = float(match.group(2))
            continue
        if stripped.startswith("Cached heterogeneous RL-TV:"):
            active_match = ACTIVE_RE.search(stripped)
            if active_match:
                parsed["active_sources"] = int(active_match.group("active"))
                parsed["total_sources"] = int(active_match.group("total"))
            continue
        if stripped.startswith("Cache build complete:"):
            cache_match = CACHE_RE.search(stripped)
            if cache_match:
                parsed["stored_rows"] = int(cache_match.group("stored_rows"))
                parsed["total_nnz"] = int(cache_match.group("total_nnz"))
                parsed["cache_mb"] = float(cache_match.group("cache_mb"))
    return parsed


def run_cmd(cmd: list[str], cwd: Path) -> tuple[str, float]:
    start_value = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    end_value = time.perf_counter()
    return proc.stdout, end_value - start_value


def benchmark_isotope(
    isotope: str,
    args: argparse.Namespace,
    truth: np.ndarray,
    density: np.ndarray,
    observed: np.ndarray | None,
    shape_xyz: tuple[int, int, int],
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"bench_{isotope}_") as tmp_name:
        tmpdir = Path(tmp_name)
        save_raw(tmpdir / "DENSITY.raw", density)
        if observed is None:
            save_raw(tmpdir / "ACTIVITY.raw", truth)
            forward_cmd = [
                str(Path(args.forward_binary).resolve()),
                isotope,
                str(args.resx),
                str(args.resy),
                str(args.nzs),
                str(args.dx),
                str(args.dy),
                str(args.dz),
                str(args.nrays),
            ]
            forward_stdout, blur_wall_s = run_cmd(forward_cmd, tmpdir)
            blurred = load_raw(tmpdir / "ACTIVITY_PRC.raw", shape_xyz)
        else:
            blurred = observed.astype(np.float32, copy=False)
            forward_stdout = "External observed volume provided."
            blur_wall_s = 0.0
        save_raw(tmpdir / "blurred.raw", blurred)

        iteration_results: list[dict[str, object]] = []
        best_result: dict[str, object] | None = None

        for iterations in args.iteration_grid:
            save_raw(tmpdir / "ACTIVITY.raw", blurred)
            solver_cmd = [
                str(Path(args.solver_binary).resolve()),
                isotope,
                str(args.resx),
                str(args.resy),
                str(args.nzs),
                str(args.dx),
                str(args.dy),
                str(args.dz),
                str(args.nrays),
                str(iterations),
                str(args.tv_weight),
                str(args.tile_sources),
                str(args.tv_eps),
                str(args.epsilon),
                "0",
                str(args.support_eps),
                str(args.entry_eps),
            ]
            solver_stdout, solver_wall_s = run_cmd(solver_cmd, tmpdir)
            corrected = load_raw(tmpdir / "ACTIVITY_PRC_RL_HETERO_CACHE.raw", shape_xyz)
            reprojection = load_raw(tmpdir / "ACTIVITY_PRC_RL_HETERO_CACHE_REPROJ.raw", shape_xyz)
            parsed = parse_solver_stdout(solver_stdout)
            truth_metrics = compute_metrics(truth, corrected)
            reproj_metrics = compute_metrics(blurred, reprojection)

            result = {
                "iterations": iterations,
                "solver_wall_s": solver_wall_s,
                "timing": parsed,
                "truth_metrics": truth_metrics,
                "reprojection_metrics": reproj_metrics,
            }
            iteration_results.append(result)

            if best_result is None or truth_metrics["rel_l2_rmse"] < best_result["truth_metrics"]["rel_l2_rmse"]:
                best_result = result

        assert best_result is not None
        return {
            "isotope": isotope,
            "blur_wall_s": blur_wall_s,
            "forward_stdout": forward_stdout.strip(),
            "best": best_result,
            "all_results": iteration_results,
        }


def main() -> int:
    args = parse_args()
    shape_xyz = (args.resx, args.resy, args.nzs)
    observed_preprocessing: dict[str, float] | None = None
    if args.truth or args.observed or args.density_file:
        if not (args.truth and args.observed and args.density_file):
            raise SystemExit("--truth, --observed, and --density-file must be provided together.")
        truth = load_raw(Path(args.truth).resolve(), shape_xyz)
        observed = load_raw(Path(args.observed).resolve(), shape_xyz)
        density = load_raw(Path(args.density_file).resolve(), shape_xyz)
        if args.preserve_observed_sum:
            observed_preprocessing = {
                "applied": 0.0,
                "scale_factor": 1.0,
                "candidate_sum_before": float(observed.sum(dtype=np.float64)),
                "candidate_sum_after": float(observed.sum(dtype=np.float64)),
                "reference_sum": float(truth.sum(dtype=np.float64)),
            }
            case_label = "external_real_case_raw_observed"
        else:
            observed, observed_preprocessing = match_sum_to_reference(observed, truth)
            case_label = "external_real_case_sum_matched"
    else:
        truth = scaled_blobs(shape_xyz)
        observed = None
        density = slab_density(shape_xyz, args.left_density, args.right_density)
        case_label = "synthetic_slab_blobs"

    results = []
    for isotope in args.isotopes:
        print(f"Benchmarking {isotope}...")
        results.append(benchmark_isotope(isotope, args, truth, density, observed, shape_xyz))

    payload = {
        "case_label": case_label,
        "shape_xyz": list(shape_xyz),
        "spacing_mm": [args.dx, args.dy, args.dz],
        "nrays": args.nrays,
        "iteration_grid": args.iteration_grid,
        "tv_weight": args.tv_weight,
        "tv_eps": args.tv_eps,
        "epsilon": args.epsilon,
        "tile_sources": args.tile_sources,
        "support_eps": args.support_eps,
        "entry_eps": args.entry_eps,
        "truth": str(Path(args.truth).resolve()) if args.truth else None,
        "observed": str(Path(args.observed).resolve()) if args.observed else None,
        "density_file": str(Path(args.density_file).resolve()) if args.density_file else None,
        "observed_preprocessing": observed_preprocessing,
        "left_density": args.left_density if not args.density_file else None,
        "right_density": args.right_density if not args.density_file else None,
        "results": results,
    }

    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Results written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
