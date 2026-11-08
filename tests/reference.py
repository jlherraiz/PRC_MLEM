"""Independent CPU range-model quadrature and float32 lookup-table checks."""
import re

import numpy as np


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


def survival(branches, r, dtype=np.float64):
    r = np.asarray(r, dtype=dtype)
    s = np.zeros_like(r)
    for w, a, b, c, f in np.asarray(branches, dtype=dtype):
        x, y = b * r, c * r
        s += w * np.exp(-((a * r) ** (dtype(1) + f) + x * x * np.sqrt(x) + y * y * y * np.sqrt(y)))
    return s


def range_table_error(branches, bins, samples=1_000_000):
    """Sample |S_table-S_float64| densely and at every bin midpoint; not a proven bound."""
    br = np.array(branches, dtype=np.float32)
    br[np.argmin(br[:, 0]), 0] += np.float32(1) - np.sum(br[:, 0], dtype=np.float32)
    lo, hi = np.float32(0), np.float32(1)  # 99.95% range cutoff, matching float32 setup
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
    return float(np.max(np.abs(lookup.astype(np.float64) - survival(br, w))))


def kernel(branches, voxel_mm, half_width, density=1.0, cutoff_mm=30.0, n_dirs=1000, n_shells=600):
    """Voxel-averaged kernel [z,y,x]; voxel_mm=(dx,dy,dz), offsets +/-half_width.

    Uniform voxel emission equals centre emission with trilinear deposition in
    a homogeneous medium. Integrate survival differences over radial shells and
    Fibonacci directions. This quadrature is independent of CUDA ray traversal.
    """
    branches = np.array(branches, np.float64)
    branches[np.argmin(branches[:, 0]), 0] += 1 - branches[:, 0].sum()
    edges = cutoff_mm * np.linspace(0, 1, n_shells + 1) ** 2 / density
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
