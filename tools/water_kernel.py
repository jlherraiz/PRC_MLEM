#!/usr/bin/env python3
"""Exact voxel-to-voxel positron-range kernel in a uniform medium, independent of prc.cu.

Positrons are emitted uniformly inside the source voxel and isotropically, travel in straight lines
and annihilate at a water-equivalent distance distributed as 1 - S(r). The probability of
annihilating in the voxel at offset D (in voxels) is

    P(D) = int (S(r_in) - S(r_out)) over radial shells x (1 / 4 pi) int Lambda(r u / h - D) du

where Lambda(v) = prod_i max(0, 1 - |v_i|) is the overlap of two voxels shifted by v. Uniform emission
from the voxel is therefore the same as emission from its centre followed by a trilinear splat of the
annihilation point. The radial integral uses exact survival differences; the directions come from a
Fibonacci lattice. 1000 directions and 600 shells agree with 20000 x 3000 to 2e-5 of the blur.

  water_kernel.py 68Ga 2 2 2        # prints the central weights of a 2 mm water kernel
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_range_table import read_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def survival(branches, r):
    s = np.zeros_like(r)
    for w, a, b, c, f in np.asarray(branches, np.float64):
        s += w * np.exp(-((a * r) ** (1 + f) + (b * r) ** 2.5 + (c * r) ** 3.5))
    return s


def kernel(branches, voxel_mm, half_width, density=1.0, cutoff_mm=30.0, n_dirs=1000, n_shells=600):
    """P on offsets -half_width..half_width, indexed [z, y, x]; voxel_mm = (dx, dy, dz)."""
    branches = np.array(branches, np.float64)
    branches[np.argmin(branches[:, 0]), 0] += 1 - branches[:, 0].sum()
    edges = cutoff_mm * np.linspace(0, 1, n_shells + 1) ** 2 / density  # finer near the source
    mass = -np.diff(survival(branches, density * edges))
    radius = 0.5 * (edges[1:] + edges[:-1])
    k = np.arange(n_dirs)
    z = 1 - (2 * k + 1) / n_dirs
    phi = 2 * np.pi * k / ((1 + 5 ** 0.5) / 2)
    dirs = np.stack([np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z], 1)
    n = 2 * half_width + 1
    out = np.zeros(n ** 3)
    for start in range(0, n_shells, 100):
        p = (radius[start:start + 100, None, None] * dirs / np.asarray(voxel_mm)).reshape(-1, 3)
        m = np.repeat(mass[start:start + 100], n_dirs) / n_dirs
        base = np.floor(p).astype(int)
        frac = p - base
        for corner in range(8):
            o = np.array([(corner >> axis) & 1 for axis in range(3)])
            idx = base + o + half_width
            ok = np.all((idx >= 0) & (idx < n), axis=1)
            weight = m * np.prod(np.where(o, frac, 1 - frac), axis=1)
            out += np.bincount(((idx[ok, 2] * n) + idx[ok, 1]) * n + idx[ok, 0], weight[ok], n ** 3)
    return out.reshape(n, n, n)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("isotope")
    parser.add_argument("voxel", type=float, nargs=3, metavar=("DX", "DY", "DZ"))
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--half-width", type=int, default=2)
    args = parser.parse_args()
    _, isotopes, _ = read_model(ROOT / "prc.cu")
    P = kernel(isotopes[args.isotope], args.voxel, args.half_width, args.density)
    c = args.half_width
    print(f"self {P[c, c, c]:.5f}  +x {P[c, c, c + 1]:.5f}  +y {P[c, c + 1, c]:.5f}  +z {P[c + 1, c, c]:.5f}"
          f"  within +/-{c}: {P.sum():.5f}")


if __name__ == "__main__":
    main()
