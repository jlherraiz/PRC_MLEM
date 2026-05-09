#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import oaconvolve


CUTOFF_TARGET_CDF = 0.9995


@dataclass(frozen=True)
class BranchModel:
    weight: float
    a: float
    b: float
    c: float
    f: float


@dataclass(frozen=True)
class IsotopeModel:
    name: str
    branches: tuple[BranchModel, ...]


ISOTOPE_MODELS = {
    "11C": IsotopeModel("11C", (BranchModel(1.000, 0.777, 0.468, 0.217, 0.010),)),
    "13N": IsotopeModel("13N", (BranchModel(1.000, 0.537, 0.347, 0.193, 0.050),)),
    "15O": IsotopeModel("15O", (BranchModel(1.000, 0.295, 0.221, 0.133, 0.118),)),
    "18F": IsotopeModel("18F", (BranchModel(1.000, 1.612, 0.703, 0.526, 0.011),)),
    "64Cu": IsotopeModel("64Cu", (BranchModel(1.000, 1.494, 0.000, 0.631, 0.251),)),
    "68Ga": IsotopeModel(
        "68Ga",
        (
            BranchModel(0.014, 1.048, 0.000, 0.475, 0.271),
            BranchModel(0.986, 0.289, 0.137, 0.159, 0.322),
        ),
    ),
    "82Rb": IsotopeModel(
        "82Rb",
        (
            BranchModel(0.136, 0.191, 0.084, 0.114, 0.408),
            BranchModel(0.857, 0.135, 0.063, 0.087, 0.451),
        ),
    ),
    "124I": IsotopeModel(
        "124I",
        (
            BranchModel(0.013, 1.007, 0.000, 0.421, 0.443),
            BranchModel(0.515, 0.412, 0.000, 0.210, 0.458),
            BranchModel(0.471, 0.241, 0.000, 0.150, 0.369),
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kernel-based 3D Richardson-Lucy positron-range correction. "
            "The kernel is generated from the existing deterministic CUDA ray projector."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    kernel_parser = subparsers.add_parser("generate-kernel", help="Generate a deterministic positron-range kernel.")
    add_spacing_args(kernel_parser)
    add_common_kernel_args(kernel_parser)
    kernel_parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix. The script writes <prefix>.npy, <prefix>.raw, and <prefix>.json.",
    )

    deconv_parser = subparsers.add_parser("deconvolve", help="Run RL-TV deconvolution on a 3D RAW volume.")
    deconv_parser.add_argument("--input", required=True, help="Blurred float32 RAW image.")
    deconv_parser.add_argument("--output", required=True, help="Corrected float32 RAW image.")
    add_geometry_args(deconv_parser)
    add_common_kernel_args(deconv_parser)
    add_rl_args(deconv_parser)
    deconv_parser.add_argument("--kernel", help="Optional precomputed kernel .npy file.")
    deconv_parser.add_argument("--kernel-prefix", help="Optional prefix to save the generated kernel.")
    deconv_parser.add_argument("--reproject-output", help="Optional RAW file for convolution reprojection.")
    deconv_parser.add_argument(
        "--simulate-output",
        help="Optional RAW file for CUDA reprojection of the corrected image using pr.x.",
    )
    deconv_parser.add_argument("--truth", help="Optional float32 RAW reference image for metrics.")
    deconv_parser.add_argument("--metrics-output", help="Optional JSON metrics output.")
    deconv_parser.add_argument("--verbose", action="store_true", help="Print per-iteration residuals.")

    demo_parser = subparsers.add_parser("demo-rb82", help="Run a uniform-water 82Rb correction demo.")
    demo_parser.add_argument(
        "--output-dir",
        default="rl_demo_rb82",
        help="Directory for the generated RAW volumes, kernel files, and metrics.",
    )
    demo_parser.add_argument("--resx", type=int, default=48)
    demo_parser.add_argument("--resy", type=int, default=48)
    demo_parser.add_argument("--nzs", type=int, default=48)
    demo_parser.add_argument("--dx", type=float, default=1.0)
    demo_parser.add_argument("--dy", type=float, default=1.0)
    demo_parser.add_argument("--dz", type=float, default=1.0)
    demo_parser.add_argument("--nrays", type=int, default=1024)
    demo_parser.add_argument("--density", type=float, default=1.0)
    demo_parser.add_argument("--padding-vox", type=int, default=3)
    demo_parser.add_argument("--iterations", type=int, default=20)
    demo_parser.add_argument("--tv-weight", type=float, default=0.002)
    demo_parser.add_argument("--tv-eps", type=float, default=1.0e-3)
    demo_parser.add_argument("--epsilon", type=float, default=1.0e-6)
    demo_parser.add_argument("--verbose", action="store_true")
    demo_parser.add_argument(
        "--binary",
        default=str(Path(__file__).resolve().parent / "pr.x"),
        help="Deterministic CUDA forward projector used to generate the kernel and validation reprojection.",
    )

    return parser.parse_args()


def add_geometry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resx", type=int, required=True)
    parser.add_argument("--resy", type=int, required=True)
    parser.add_argument("--nzs", type=int, required=True)
    add_spacing_args(parser)


def add_spacing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dy", type=float, required=True)
    parser.add_argument("--dz", type=float, required=True)


def add_common_kernel_args(parser: argparse.ArgumentParser) -> None:
    if "--isotope" not in parser._option_string_actions:
        parser.add_argument("--isotope", required=True, choices=sorted(ISOTOPE_MODELS))
    parser.add_argument("--density", type=float, default=1.0, help="Uniform density used for the kernel.")
    parser.add_argument("--nrays", type=int, default=1024, help="Number of ray directions for pr.x.")
    parser.add_argument("--padding-vox", type=int, default=3, help="Extra padding around the cutoff radius.")
    parser.add_argument(
        "--binary",
        default=str(Path(__file__).resolve().parent / "pr.x"),
        help="Deterministic CUDA forward projector used to build the kernel.",
    )


def add_rl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tv-weight", type=float, default=0.002)
    parser.add_argument("--tv-eps", type=float, default=1.0e-3)
    parser.add_argument("--epsilon", type=float, default=1.0e-6, help="Numerical floor for divisions.")


def model_p(branch: BranchModel, r_mm: float) -> float:
    ar = branch.a * r_mm
    br = branch.b * r_mm
    cr = branch.c * r_mm

    p = ar ** (1.0 + branch.f)
    if branch.b > 0.0:
        p += br ** 2.5
    if branch.c > 0.0:
        p += cr ** 3.5
    return p


def isotope_cdf(isotope: IsotopeModel, r_mm: float) -> float:
    survival = 0.0
    for branch in isotope.branches:
        survival += branch.weight * np.exp(-model_p(branch, r_mm))
    return float(1.0 - survival)


def compute_cutoff_mm(isotope: IsotopeModel, target_cdf: float = CUTOFF_TARGET_CDF) -> float:
    lo = 0.0
    hi = 1.0
    while isotope_cdf(isotope, hi) < target_cdf and hi < 256.0:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if isotope_cdf(isotope, mid) < target_cdf:
            lo = mid
        else:
            hi = mid
    return hi


def xyz_to_zyx_shape(shape_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
    return (shape_xyz[2], shape_xyz[1], shape_xyz[0])


def ensure_positive_geometry(shape_xyz: tuple[int, int, int], spacing_mm: tuple[float, float, float]) -> None:
    if min(shape_xyz) <= 0:
        raise ValueError("All dimensions must be positive.")
    if min(spacing_mm) <= 0.0:
        raise ValueError("All voxel spacings must be positive.")


def ensure_positive_spacing(spacing_mm: tuple[float, float, float]) -> None:
    if min(spacing_mm) <= 0.0:
        raise ValueError("All voxel spacings must be positive.")


def gaussian_blobs(shape_xyz: tuple[int, int, int]) -> np.ndarray:
    resx, resy, nzs = shape_xyz
    zz, yy, xx = np.indices((nzs, resy, resx), dtype=np.float32)
    blob1 = 12.0 * np.exp(-((xx - 12.0) ** 2 + (yy - 20.0) ** 2 + (zz - 16.0) ** 2) / (2.0 * 3.0**2))
    blob2 = 7.5 * np.exp(-((xx - 23.0) ** 2 + (yy - 11.0) ** 2 + (zz - 18.0) ** 2) / (2.0 * 4.0**2))
    arr = (blob1 + blob2).astype(np.float32)
    arr[arr < 1.0e-3] = 0.0
    return arr


def uniform_density(shape_xyz: tuple[int, int, int], value: float) -> np.ndarray:
    return np.full(xyz_to_zyx_shape(shape_xyz), value, dtype=np.float32)


def centered_point_source(shape_xyz: tuple[int, int, int], value: float) -> np.ndarray:
    resx, resy, nzs = shape_xyz
    arr = np.zeros((nzs, resy, resx), dtype=np.float32)
    arr[nzs // 2, resy // 2, resx // 2] = value
    return arr


def raw_size_ok(path: Path, nvoxels: int) -> bool:
    return path.exists() and path.stat().st_size == nvoxels * np.dtype(np.float32).itemsize


def load_raw_volume(path: Path, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    nvoxels = shape_xyz[0] * shape_xyz[1] * shape_xyz[2]
    arr = np.fromfile(path, dtype=np.float32, count=nvoxels)
    if arr.size != nvoxels:
        raise ValueError(f"{path} does not contain {nvoxels} float32 voxels.")
    return arr.reshape(xyz_to_zyx_shape(shape_xyz))


def save_raw_volume(path: Path, volume: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(volume, dtype=np.float32).tofile(path)


def save_kernel_bundle(prefix: Path, kernel: np.ndarray, metadata: dict[str, object]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(prefix.with_suffix(".npy"), kernel.astype(np.float32))
    save_raw_volume(prefix.with_suffix(".raw"), kernel)
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def run_forward_projector(
    binary: Path,
    isotope: str,
    shape_xyz: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
    nrays: int,
    activity: np.ndarray,
    density: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    if not binary.exists():
        raise FileNotFoundError(f"Forward projector not found: {binary}")

    resx, resy, nzs = shape_xyz
    dx, dy, dz = spacing_mm
    expected_shape = xyz_to_zyx_shape(shape_xyz)
    if activity.shape != expected_shape or density.shape != expected_shape:
        raise ValueError("Activity and density volumes must match the requested geometry.")

    with tempfile.TemporaryDirectory(prefix="pr_rl_") as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        save_raw_volume(tmpdir / "ACTIVITY.raw", activity)
        save_raw_volume(tmpdir / "DENSITY.raw", density)

        args = [
            str(binary),
            isotope,
            str(resx),
            str(resy),
            str(nzs),
            str(dx),
            str(dy),
            str(dz),
            str(nrays),
        ]
        if binary.name in {"prs.x", "positron_range_ray", "prs_notiled.x", "positron_range_ray_notiled"}:
            args.append("-1")

        start = time.perf_counter()
        proc = subprocess.run(
            args,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        elapsed_s = time.perf_counter() - start

        output_path = None
        for candidate in (
            "ACTIVITY_PRC.raw",
            "ACTIVITY_PRC_DET_RAY.raw",
            "ACTIVITY_PRC_DET_RAY_NOTILED.raw",
            "annihilation_ray_deterministic.raw",
            "annihilation_ray_notiled_deterministic.raw",
        ):
            candidate_path = tmpdir / candidate
            if raw_size_ok(candidate_path, resx * resy * nzs):
                output_path = candidate_path
                break
        if output_path is None:
            raise RuntimeError(f"Expected deterministic output file was not written by {binary}.")

        output = load_raw_volume(output_path, shape_xyz)
        return output, elapsed_s, proc.stdout


def kernel_shape_from_cutoff(
    isotope: IsotopeModel,
    spacing_mm: tuple[float, float, float],
    padding_vox: int,
) -> tuple[int, int, int]:
    cutoff_mm = compute_cutoff_mm(isotope)
    dx, dy, dz = spacing_mm
    rx = int(np.ceil(cutoff_mm / dx)) + padding_vox
    ry = int(np.ceil(cutoff_mm / dy)) + padding_vox
    rz = int(np.ceil(cutoff_mm / dz)) + padding_vox
    return (2 * rx + 1, 2 * ry + 1, 2 * rz + 1)


def generate_kernel(
    binary: Path,
    isotope_name: str,
    spacing_mm: tuple[float, float, float],
    density_value: float,
    nrays: int,
    padding_vox: int,
) -> tuple[np.ndarray, dict[str, object]]:
    isotope = ISOTOPE_MODELS[isotope_name]
    shape_xyz = kernel_shape_from_cutoff(isotope, spacing_mm, padding_vox)
    activity = centered_point_source(shape_xyz, 1.0)
    density = uniform_density(shape_xyz, density_value)
    kernel, elapsed_s, projector_log = run_forward_projector(
        binary=binary,
        isotope=isotope_name,
        shape_xyz=shape_xyz,
        spacing_mm=spacing_mm,
        nrays=nrays,
        activity=activity,
        density=density,
    )
    kernel = np.maximum(kernel, 0.0)
    kernel_sum = float(kernel.sum(dtype=np.float64))
    if not kernel_sum > 0.0:
        raise RuntimeError("Generated kernel has zero total mass.")
    kernel /= np.float32(kernel_sum)

    metadata = {
        "isotope": isotope_name,
        "shape_xyz": list(shape_xyz),
        "shape_zyx": list(xyz_to_zyx_shape(shape_xyz)),
        "spacing_mm": list(spacing_mm),
        "density": density_value,
        "nrays": nrays,
        "padding_vox": padding_vox,
        "cutoff_mm": compute_cutoff_mm(isotope),
        "sum": float(kernel.sum(dtype=np.float64)),
        "binary": str(binary),
        "generation_runtime_s": elapsed_s,
        "projector_log": projector_log.strip(),
    }
    return kernel.astype(np.float32, copy=False), metadata


def convolve_same(volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.asarray(oaconvolve(volume, kernel, mode="same"), dtype=np.float32)


def forward_project(volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.maximum(convolve_same(volume, kernel), 0.0)


def backward_project(volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.maximum(convolve_same(volume, kernel[::-1, ::-1, ::-1]), 0.0)


def tv_divergence(volume: np.ndarray, eps: float) -> np.ndarray:
    gx = np.zeros_like(volume, dtype=np.float32)
    gy = np.zeros_like(volume, dtype=np.float32)
    gz = np.zeros_like(volume, dtype=np.float32)

    gx[:, :, :-1] = volume[:, :, 1:] - volume[:, :, :-1]
    gy[:, :-1, :] = volume[:, 1:, :] - volume[:, :-1, :]
    gz[:-1, :, :] = volume[1:, :, :] - volume[:-1, :, :]

    mag = np.sqrt(gx * gx + gy * gy + gz * gz + eps * eps).astype(np.float32, copy=False)
    px = gx / mag
    py = gy / mag
    pz = gz / mag

    div = np.zeros_like(volume, dtype=np.float32)
    div[:, :, :-1] += px[:, :, :-1]
    div[:, :, 1:] -= px[:, :, :-1]
    div[:, :-1, :] += py[:, :-1, :]
    div[:, 1:, :] -= py[:, :-1, :]
    div[:-1, :, :] += pz[:-1, :, :]
    div[1:, :, :] -= pz[:-1, :, :]
    return div


def compute_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float64, copy=False).ravel()
    cand = candidate.astype(np.float64, copy=False).ravel()
    diff = cand - ref
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    ref_norm = float(np.linalg.norm(ref))
    rel_l2 = rmse / float(np.sqrt(np.mean(ref * ref))) if ref.size > 0 else 0.0
    if ref.size == 0:
        corr = 1.0
    else:
        ref_std = float(ref.std())
        cand_std = float(cand.std())
        if ref_std <= 0.0 or cand_std <= 0.0:
            corr = 1.0 if np.allclose(ref, cand) else 0.0
        else:
            corr = float(np.corrcoef(ref, cand)[0, 1])

    return {
        "sum_reference": float(ref.sum()),
        "sum_candidate": float(cand.sum()),
        "mean_abs_diff": mean_abs,
        "rmse": rmse,
        "rel_l2_rmse": rel_l2,
        "corrcoef": corr,
        "l2_norm_reference": ref_norm,
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size > 0 else 0.0,
    }


def richardson_lucy_tv(
    observed: np.ndarray,
    kernel: np.ndarray,
    iterations: int,
    tv_weight: float,
    tv_eps: float,
    epsilon: float,
    verbose: bool,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    observed = np.maximum(observed.astype(np.float32, copy=False), 0.0)
    estimate = np.maximum(observed.copy(), epsilon)
    ones_backproj = np.maximum(backward_project(np.ones_like(observed, dtype=np.float32), kernel), epsilon)
    observed_sum = float(observed.sum(dtype=np.float64))
    history: list[dict[str, float]] = []

    for iteration in range(1, iterations + 1):
        blurred = np.maximum(forward_project(estimate, kernel), epsilon)
        ratio = observed / blurred
        correction = backward_project(ratio, kernel) / ones_backproj
        estimate *= correction

        if tv_weight > 0.0:
            div = tv_divergence(estimate, tv_eps)
            tv_denom = np.maximum(1.0 - tv_weight * div, 0.25)
            estimate /= tv_denom

        estimate = np.maximum(estimate, 0.0)
        estimate_sum = float(estimate.sum(dtype=np.float64))
        if estimate_sum > 0.0 and observed_sum > 0.0:
            estimate *= np.float32(observed_sum / estimate_sum)

        reprojection = forward_project(estimate, kernel)
        residual = reprojection - observed
        rel_l2 = float(np.linalg.norm(residual.ravel().astype(np.float64)) /
                       max(np.linalg.norm(observed.ravel().astype(np.float64)), epsilon))
        entry = {
            "iteration": iteration,
            "estimate_sum": float(estimate.sum(dtype=np.float64)),
            "reprojection_rel_l2": rel_l2,
            "estimate_max": float(np.max(estimate)) if estimate.size > 0 else 0.0,
        }
        history.append(entry)
        if verbose:
            print(
                f"iter={iteration:02d} "
                f"sum={entry['estimate_sum']:.6f} "
                f"max={entry['estimate_max']:.6f} "
                f"reproj_rel_l2={entry['reprojection_rel_l2']:.6e}"
            )

    return estimate, history


def print_metric_block(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for key in (
        "sum_reference",
        "sum_candidate",
        "mean_abs_diff",
        "rmse",
        "rel_l2_rmse",
        "corrcoef",
        "max_abs_diff",
    ):
        print(f"  {key}: {metrics[key]:.6f}")


def handle_generate_kernel(args: argparse.Namespace) -> int:
    spacing_mm = (args.dx, args.dy, args.dz)
    ensure_positive_spacing(spacing_mm)

    kernel, metadata = generate_kernel(
        binary=Path(args.binary).resolve(),
        isotope_name=args.isotope,
        spacing_mm=spacing_mm,
        density_value=args.density,
        nrays=args.nrays,
        padding_vox=args.padding_vox,
    )

    prefix = Path(args.output_prefix).resolve()
    save_kernel_bundle(prefix, kernel, metadata)
    print(f"Kernel written to {prefix.with_suffix('.npy')}")
    print(f"Kernel shape (zyx): {tuple(kernel.shape)}")
    print(f"Kernel sum: {kernel.sum(dtype=np.float64):.6f}")
    return 0


def maybe_write_metrics(path: str | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def handle_deconvolve(args: argparse.Namespace) -> int:
    shape_xyz = (args.resx, args.resy, args.nzs)
    spacing_mm = (args.dx, args.dy, args.dz)
    ensure_positive_geometry(shape_xyz, spacing_mm)

    input_path = Path(args.input).resolve()
    observed = load_raw_volume(input_path, shape_xyz)

    if args.kernel:
        kernel_path = Path(args.kernel).resolve()
        kernel = np.load(kernel_path).astype(np.float32)
        kernel_metadata: dict[str, object] = {"kernel_path": str(kernel_path)}
    else:
        kernel, kernel_metadata = generate_kernel(
            binary=Path(args.binary).resolve(),
            isotope_name=args.isotope,
            spacing_mm=spacing_mm,
            density_value=args.density,
            nrays=args.nrays,
            padding_vox=args.padding_vox,
        )
        if args.kernel_prefix:
            save_kernel_bundle(Path(args.kernel_prefix).resolve(), kernel, kernel_metadata)

    corrected, history = richardson_lucy_tv(
        observed=observed,
        kernel=kernel,
        iterations=args.iterations,
        tv_weight=args.tv_weight,
        tv_eps=args.tv_eps,
        epsilon=args.epsilon,
        verbose=args.verbose,
    )

    corrected_path = Path(args.output).resolve()
    save_raw_volume(corrected_path, corrected)

    payload: dict[str, object] = {
        "command": "deconvolve",
        "input": str(input_path),
        "output": str(corrected_path),
        "shape_xyz": list(shape_xyz),
        "spacing_mm": list(spacing_mm),
        "isotope": args.isotope,
        "nrays": args.nrays,
        "density": args.density,
        "iterations": args.iterations,
        "tv_weight": args.tv_weight,
        "tv_eps": args.tv_eps,
        "epsilon": args.epsilon,
        "kernel_metadata": kernel_metadata,
        "history": history,
    }

    if args.reproject_output:
        reprojection = forward_project(corrected, kernel)
        reproject_path = Path(args.reproject_output).resolve()
        save_raw_volume(reproject_path, reprojection)
        reproj_metrics = compute_metrics(observed, reprojection)
        payload["kernel_reprojection"] = reproj_metrics
        print_metric_block("Kernel reprojection vs blurred input", reproj_metrics)

    if args.simulate_output:
        density = uniform_density(shape_xyz, args.density)
        simulated, elapsed_s, projector_log = run_forward_projector(
            binary=Path(args.binary).resolve(),
            isotope=args.isotope,
            shape_xyz=shape_xyz,
            spacing_mm=spacing_mm,
            nrays=args.nrays,
            activity=corrected,
            density=density,
        )
        simulate_path = Path(args.simulate_output).resolve()
        save_raw_volume(simulate_path, simulated)
        sim_metrics = compute_metrics(observed, simulated)
        payload["cuda_reprojection"] = sim_metrics | {
            "runtime_s": elapsed_s,
            "projector_log": projector_log.strip(),
        }
        print_metric_block("CUDA reprojection vs blurred input", sim_metrics)

    if args.truth:
        truth = load_raw_volume(Path(args.truth).resolve(), shape_xyz)
        truth_metrics = compute_metrics(truth, corrected)
        payload["truth_metrics"] = truth_metrics
        print_metric_block("Corrected image vs truth", truth_metrics)

    maybe_write_metrics(args.metrics_output, payload)
    print(f"Corrected image written to {corrected_path}")
    return 0


def handle_demo_rb82(args: argparse.Namespace) -> int:
    shape_xyz = (args.resx, args.resy, args.nzs)
    spacing_mm = (args.dx, args.dy, args.dz)
    ensure_positive_geometry(shape_xyz, spacing_mm)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = Path(args.binary).resolve()

    truth = gaussian_blobs(shape_xyz)
    density = uniform_density(shape_xyz, args.density)

    kernel, kernel_metadata = generate_kernel(
        binary=binary,
        isotope_name="82Rb",
        spacing_mm=spacing_mm,
        density_value=args.density,
        nrays=args.nrays,
        padding_vox=args.padding_vox,
    )

    blurred, blur_runtime_s, blur_log = run_forward_projector(
        binary=binary,
        isotope="82Rb",
        shape_xyz=shape_xyz,
        spacing_mm=spacing_mm,
        nrays=args.nrays,
        activity=truth,
        density=density,
    )

    corrected, history = richardson_lucy_tv(
        observed=blurred,
        kernel=kernel,
        iterations=args.iterations,
        tv_weight=args.tv_weight,
        tv_eps=args.tv_eps,
        epsilon=args.epsilon,
        verbose=args.verbose,
    )

    kernel_reprojection = forward_project(corrected, kernel)
    simulated_reprojection, reproj_runtime_s, reproj_log = run_forward_projector(
        binary=binary,
        isotope="82Rb",
        shape_xyz=shape_xyz,
        spacing_mm=spacing_mm,
        nrays=args.nrays,
        activity=corrected,
        density=density,
    )

    save_raw_volume(output_dir / "truth.raw", truth)
    save_raw_volume(output_dir / "density.raw", density)
    save_raw_volume(output_dir / "blurred_cuda.raw", blurred)
    save_raw_volume(output_dir / "corrected_rl_tv.raw", corrected)
    save_raw_volume(output_dir / "reprojected_kernel.raw", kernel_reprojection)
    save_raw_volume(output_dir / "reprojected_cuda.raw", simulated_reprojection)
    save_kernel_bundle(output_dir / "kernel_82Rb", kernel, kernel_metadata)

    metrics = {
        "truth_vs_corrected": compute_metrics(truth, corrected),
        "blurred_vs_kernel_reprojection": compute_metrics(blurred, kernel_reprojection),
        "blurred_vs_cuda_reprojection": compute_metrics(blurred, simulated_reprojection),
        "shape_xyz": list(shape_xyz),
        "spacing_mm": list(spacing_mm),
        "density": args.density,
        "nrays": args.nrays,
        "iterations": args.iterations,
        "tv_weight": args.tv_weight,
        "tv_eps": args.tv_eps,
        "blur_runtime_s": blur_runtime_s,
        "cuda_reprojection_runtime_s": reproj_runtime_s,
        "kernel_metadata": kernel_metadata,
        "blur_projector_log": blur_log.strip(),
        "reprojection_projector_log": reproj_log.strip(),
        "history": history,
    }
    maybe_write_metrics(str(output_dir / "metrics.json"), metrics)

    print_metric_block("Corrected image vs truth", metrics["truth_vs_corrected"])
    print_metric_block("Kernel reprojection vs blurred input", metrics["blurred_vs_kernel_reprojection"])
    print_metric_block("CUDA reprojection vs blurred input", metrics["blurred_vs_cuda_reprojection"])
    print(f"Demo outputs written to {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.command == "generate-kernel":
            return handle_generate_kernel(args)
        if args.command == "deconvolve":
            return handle_deconvolve(args)
        if args.command == "demo-rb82":
            return handle_demo_rb82(args)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout, file=sys.stderr)
        print(f"Forward projector failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
