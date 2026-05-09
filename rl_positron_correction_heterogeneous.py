#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl_positron_correction import (
    compute_metrics,
    ensure_positive_geometry,
    load_raw_volume,
    print_metric_block,
    save_raw_volume,
    tv_divergence,
)


@dataclass
class DeterministicProjectorRunner:
    binary: Path
    output_name: str
    isotope: str
    shape_xyz: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    nrays: int
    density: np.ndarray
    label: str

    def __post_init__(self) -> None:
        self._tmpdir_name: str | None = None
        self._tmpdir: Path | None = None

    def __enter__(self) -> "DeterministicProjectorRunner":
        if not self.binary.exists():
            raise FileNotFoundError(f"Projector not found: {self.binary}")
        self._tmpdir_name = tempfile.mkdtemp(prefix=f"{self.label}_")
        self._tmpdir = Path(self._tmpdir_name)
        save_raw_volume(self._tmpdir / "DENSITY.raw", self.density)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tmpdir_name is not None:
            shutil.rmtree(self._tmpdir_name, ignore_errors=True)
        self._tmpdir_name = None
        self._tmpdir = None

    @property
    def cmd(self) -> list[str]:
        resx, resy, nzs = self.shape_xyz
        dx, dy, dz = self.spacing_mm
        return [
            str(self.binary),
            self.isotope,
            str(resx),
            str(resy),
            str(nzs),
            str(dx),
            str(dy),
            str(dz),
            str(self.nrays),
        ]

    def run(self, volume: np.ndarray) -> tuple[np.ndarray, float, str]:
        if self._tmpdir is None:
            raise RuntimeError("ProjectorRunner must be used as a context manager.")
        save_raw_volume(self._tmpdir / "ACTIVITY.raw", volume)
        start = time.perf_counter()
        proc = subprocess.run(
            self.cmd,
            cwd=self._tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        elapsed_s = time.perf_counter() - start
        output_path = self._tmpdir / self.output_name
        output = load_raw_volume(output_path, self.shape_xyz)
        return output, elapsed_s, proc.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full heterogeneous RL-TV correction using pr.x for A and pr_adj.x for A^T."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    deconv = subparsers.add_parser("deconvolve", help="Run heterogeneous RL-TV deconvolution on a blurred RAW volume.")
    add_common_args(deconv)
    deconv.add_argument("--input", required=True, help="Blurred float32 RAW image.")
    deconv.add_argument("--density", required=True, help="Density float32 RAW image.")
    deconv.add_argument("--output", required=True, help="Corrected float32 RAW image.")
    deconv.add_argument("--reproject-output", help="Optional CUDA forward reprojection RAW output.")
    deconv.add_argument("--truth", help="Optional float32 RAW reference image for metrics.")
    deconv.add_argument(
        "--match-input-sum-to",
        help="Optional float32 RAW reference image used to rescale the blurred input sum before RL.",
    )
    deconv.add_argument("--metrics-output", help="Optional JSON metrics output.")
    deconv.add_argument("--verbose", action="store_true")

    demo = subparsers.add_parser("demo-rb82", help="Run a small heterogeneous 82Rb slab-interface demo.")
    demo.add_argument("--output-dir", default="rl_demo_rb82_heterogeneous")
    demo.add_argument("--resx", type=int, default=48)
    demo.add_argument("--resy", type=int, default=48)
    demo.add_argument("--nzs", type=int, default=48)
    demo.add_argument("--dx", type=float, default=1.0)
    demo.add_argument("--dy", type=float, default=1.0)
    demo.add_argument("--dz", type=float, default=1.0)
    demo.add_argument("--nrays", type=int, default=512)
    demo.add_argument("--iterations", type=int, default=12)
    demo.add_argument("--tv-weight", type=float, default=0.002)
    demo.add_argument("--tv-eps", type=float, default=1.0e-3)
    demo.add_argument("--epsilon", type=float, default=1.0e-6)
    demo.add_argument("--left-density", type=float, default=1.0)
    demo.add_argument("--right-density", type=float, default=0.35)
    demo.add_argument("--interface-x", type=int, default=24)
    demo.add_argument("--verbose", action="store_true")
    demo.add_argument("--forward-binary", default=str(Path(__file__).resolve().parent / "pr.x"))
    demo.add_argument("--adjoint-binary", default=str(Path(__file__).resolve().parent / "pr_adj.x"))

    return parser.parse_args()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--isotope", required=True)
    parser.add_argument("--resx", type=int, required=True)
    parser.add_argument("--resy", type=int, required=True)
    parser.add_argument("--nzs", type=int, required=True)
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dy", type=float, required=True)
    parser.add_argument("--dz", type=float, required=True)
    parser.add_argument("--nrays", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--tv-weight", type=float, default=0.002)
    parser.add_argument("--tv-eps", type=float, default=1.0e-3)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--forward-binary", default=str(Path(__file__).resolve().parent / "pr.x"))
    parser.add_argument("--adjoint-binary", default=str(Path(__file__).resolve().parent / "pr_adj.x"))


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


def slab_density(shape_xyz: tuple[int, int, int], plane_x: int, left: float, right: float) -> np.ndarray:
    resx, resy, nzs = shape_xyz
    density = np.empty((nzs, resy, resx), dtype=np.float32)
    density[:, :, :plane_x] = left
    density[:, :, plane_x:] = right
    return density


def heterogeneous_blobs(shape_xyz: tuple[int, int, int]) -> np.ndarray:
    resx, resy, nzs = shape_xyz
    zz, yy, xx = np.indices((nzs, resy, resx), dtype=np.float32)
    blob1 = 12.0 * np.exp(-((xx - 15.0) ** 2 + (yy - 18.0) ** 2 + (zz - 20.0) ** 2) / (2.0 * 3.0**2))
    blob2 = 8.5 * np.exp(-((xx - 33.0) ** 2 + (yy - 28.0) ** 2 + (zz - 24.0) ** 2) / (2.0 * 4.0**2))
    arr = (blob1 + blob2).astype(np.float32)
    arr[arr < 1.0e-3] = 0.0
    return arr


def richardson_lucy_tv_heterogeneous(
    observed: np.ndarray,
    forward_runner: DeterministicProjectorRunner,
    adjoint_runner: DeterministicProjectorRunner,
    iterations: int,
    tv_weight: float,
    tv_eps: float,
    epsilon: float,
    verbose: bool,
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, float]]:
    observed = np.maximum(observed.astype(np.float32, copy=False), 0.0)
    # RL starts from the measured blurred image, not from a uniform image.
    estimate = np.maximum(observed.copy(), epsilon)
    ones = np.ones_like(observed, dtype=np.float32)
    ones_backproj, ones_time_s, _ = adjoint_runner.run(ones)
    ones_backproj = np.maximum(ones_backproj, epsilon)
    observed_sum = float(observed.sum(dtype=np.float64))
    history: list[dict[str, float]] = []
    totals = {
        "ones_backproj_runtime_s": ones_time_s,
        "forward_runtime_s": 0.0,
        "adjoint_runtime_s": 0.0,
    }

    for iteration in range(1, iterations + 1):
        blurred, forward_time_s, _ = forward_runner.run(estimate)
        ratio = observed / np.maximum(blurred, epsilon)
        correction, adjoint_time_s, _ = adjoint_runner.run(ratio)
        correction = correction / ones_backproj
        estimate *= correction

        if tv_weight > 0.0:
            div = tv_divergence(estimate, tv_eps)
            tv_denom = np.maximum(1.0 - tv_weight * div, 0.25)
            estimate /= tv_denom

        estimate = np.maximum(estimate, 0.0)
        estimate_sum = float(estimate.sum(dtype=np.float64))
        if estimate_sum > 0.0 and observed_sum > 0.0:
            estimate *= np.float32(observed_sum / estimate_sum)

        totals["forward_runtime_s"] += forward_time_s
        totals["adjoint_runtime_s"] += adjoint_time_s

        entry = {
            "iteration": iteration,
            "estimate_sum": float(estimate.sum(dtype=np.float64)),
            "estimate_max": float(np.max(estimate)) if estimate.size > 0 else 0.0,
            "forward_runtime_s": forward_time_s,
            "adjoint_runtime_s": adjoint_time_s,
        }

        if verbose:
            reprojection, reproj_time_s, _ = forward_runner.run(estimate)
            totals["forward_runtime_s"] += reproj_time_s
            residual = reprojection - observed
            entry["reprojection_rel_l2"] = float(
                np.linalg.norm(residual.ravel().astype(np.float64)) /
                max(np.linalg.norm(observed.ravel().astype(np.float64)), epsilon)
            )
            print(
                f"iter={iteration:02d} "
                f"sum={entry['estimate_sum']:.6f} "
                f"max={entry['estimate_max']:.6f} "
                f"reproj_rel_l2={entry['reprojection_rel_l2']:.6e}"
            )

        history.append(entry)

    return estimate, history, totals


def handle_deconvolve(args: argparse.Namespace) -> int:
    shape_xyz = (args.resx, args.resy, args.nzs)
    spacing_mm = (args.dx, args.dy, args.dz)
    ensure_positive_geometry(shape_xyz, spacing_mm)

    observed = load_raw_volume(Path(args.input).resolve(), shape_xyz)
    density = load_raw_volume(Path(args.density).resolve(), shape_xyz)
    match_sum_reference_path = Path(args.match_input_sum_to).resolve() if args.match_input_sum_to else None
    truth_path = Path(args.truth).resolve() if args.truth else None
    truth_reference: np.ndarray | None = None
    preprocessing: dict[str, float] | None = None

    if match_sum_reference_path is not None:
        truth_reference = load_raw_volume(match_sum_reference_path, shape_xyz)
        observed, preprocessing = match_sum_to_reference(observed, truth_reference)
        print(
            "Matched blurred input sum to reference: "
            f"scale={preprocessing['scale_factor']:.6f}, "
            f"sum_before={preprocessing['candidate_sum_before']:.6f}, "
            f"sum_after={preprocessing['candidate_sum_after']:.6f}, "
            f"reference_sum={preprocessing['reference_sum']:.6f}"
        )

    with DeterministicProjectorRunner(
        binary=Path(args.forward_binary).resolve(),
        output_name="ACTIVITY_PRC.raw",
        isotope=args.isotope,
        shape_xyz=shape_xyz,
        spacing_mm=spacing_mm,
        nrays=args.nrays,
        density=density,
        label="hetero_forward",
    ) as forward_runner, DeterministicProjectorRunner(
        binary=Path(args.adjoint_binary).resolve(),
        output_name="ACTIVITY_PRC_ADJOINT.raw",
        isotope=args.isotope,
        shape_xyz=shape_xyz,
        spacing_mm=spacing_mm,
        nrays=args.nrays,
        density=density,
        label="hetero_adjoint",
    ) as adjoint_runner:
        corrected, history, totals = richardson_lucy_tv_heterogeneous(
            observed=observed,
            forward_runner=forward_runner,
            adjoint_runner=adjoint_runner,
            iterations=args.iterations,
            tv_weight=args.tv_weight,
            tv_eps=args.tv_eps,
            epsilon=args.epsilon,
            verbose=args.verbose,
        )
        reprojection = None
        reprojection_time_s = 0.0
        if args.reproject_output or args.metrics_output or args.truth:
            reprojection, reprojection_time_s, _ = forward_runner.run(corrected)

    corrected_path = Path(args.output).resolve()
    save_raw_volume(corrected_path, corrected)

    payload: dict[str, object] = {
        "command": "deconvolve",
        "input": str(Path(args.input).resolve()),
        "density": str(Path(args.density).resolve()),
        "output": str(corrected_path),
        "shape_xyz": list(shape_xyz),
        "spacing_mm": list(spacing_mm),
        "isotope": args.isotope,
        "nrays": args.nrays,
        "iterations": args.iterations,
        "tv_weight": args.tv_weight,
        "tv_eps": args.tv_eps,
        "epsilon": args.epsilon,
        "history": history,
        "timings": totals | {"final_reprojection_runtime_s": reprojection_time_s},
        "initial_estimate": "blurred_input",
        "input_preprocessing": preprocessing,
    }

    if args.reproject_output and reprojection is not None:
        save_raw_volume(Path(args.reproject_output).resolve(), reprojection)

    if reprojection is not None:
        reproj_metrics = compute_metrics(observed, reprojection)
        payload["reprojection_metrics"] = reproj_metrics
        print_metric_block("CUDA reprojection vs blurred input", reproj_metrics)

    if args.truth:
        if truth_reference is not None and truth_path == match_sum_reference_path:
            truth = truth_reference
        else:
            truth = load_raw_volume(truth_path, shape_xyz)
        truth_metrics = compute_metrics(truth, corrected)
        payload["truth_metrics"] = truth_metrics
        print_metric_block("Corrected image vs truth", truth_metrics)

    if args.metrics_output:
        metrics_path = Path(args.metrics_output).resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Corrected image written to {corrected_path}")
    return 0


def handle_demo_rb82(args: argparse.Namespace) -> int:
    shape_xyz = (args.resx, args.resy, args.nzs)
    spacing_mm = (args.dx, args.dy, args.dz)
    ensure_positive_geometry(shape_xyz, spacing_mm)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = heterogeneous_blobs(shape_xyz)
    density = slab_density(shape_xyz, args.interface_x, args.left_density, args.right_density)

    with DeterministicProjectorRunner(
        binary=Path(args.forward_binary).resolve(),
        output_name="ACTIVITY_PRC.raw",
        isotope="82Rb",
        shape_xyz=shape_xyz,
        spacing_mm=spacing_mm,
        nrays=args.nrays,
        density=density,
        label="hetero_demo_forward",
    ) as forward_runner, DeterministicProjectorRunner(
        binary=Path(args.adjoint_binary).resolve(),
        output_name="ACTIVITY_PRC_ADJOINT.raw",
        isotope="82Rb",
        shape_xyz=shape_xyz,
        spacing_mm=spacing_mm,
        nrays=args.nrays,
        density=density,
        label="hetero_demo_adjoint",
    ) as adjoint_runner:
        blurred, blur_runtime_s, blur_log = forward_runner.run(truth)
        corrected, history, totals = richardson_lucy_tv_heterogeneous(
            observed=blurred,
            forward_runner=forward_runner,
            adjoint_runner=adjoint_runner,
            iterations=args.iterations,
            tv_weight=args.tv_weight,
            tv_eps=args.tv_eps,
            epsilon=args.epsilon,
            verbose=args.verbose,
        )
        reprojection, reproj_runtime_s, reproj_log = forward_runner.run(corrected)

    save_raw_volume(output_dir / "truth.raw", truth)
    save_raw_volume(output_dir / "density.raw", density)
    save_raw_volume(output_dir / "blurred_cuda.raw", blurred)
    save_raw_volume(output_dir / "corrected_rl_tv.raw", corrected)
    save_raw_volume(output_dir / "reprojected_cuda.raw", reprojection)

    metrics = {
        "truth_vs_corrected": compute_metrics(truth, corrected),
        "blurred_vs_cuda_reprojection": compute_metrics(blurred, reprojection),
        "shape_xyz": list(shape_xyz),
        "spacing_mm": list(spacing_mm),
        "nrays": args.nrays,
        "iterations": args.iterations,
        "tv_weight": args.tv_weight,
        "tv_eps": args.tv_eps,
        "left_density": args.left_density,
        "right_density": args.right_density,
        "interface_x": args.interface_x,
        "blur_runtime_s": blur_runtime_s,
        "reprojection_runtime_s": reproj_runtime_s,
        "iterative_timings": totals,
        "blur_projector_log": blur_log.strip(),
        "reprojection_projector_log": reproj_log.strip(),
        "history": history,
        "initial_estimate": "blurred_input",
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print_metric_block("Corrected image vs truth", metrics["truth_vs_corrected"])
    print_metric_block("CUDA reprojection vs blurred input", metrics["blurred_vs_cuda_reprojection"])
    print(f"Heterogeneous demo outputs written to {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.command == "deconvolve":
            return handle_deconvolve(args)
        if args.command == "demo-rb82":
            return handle_demo_rb82(args)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout, file=sys.stderr)
        print(f"Projector failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
