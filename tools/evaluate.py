#!/usr/bin/env python3
"""Reference-based NRMSE, 3D SSIM, uptake and local-range metrics.

SSIM uses a fixed reference range and an 11-voxel Gaussian window. Tissue/edge
scores avoid background-dominated averages; unavailable scores are None.
The local-range metric includes noise and mismatch, not only ringing.
Background, recovery and halo metrics assume a piecewise-constant phantom;
use explicit tissue/hot masks and background where available. These scores
require a known reference and do not establish accuracy on acquired PET.

Usage: evaluate.py --truth TRUTH.raw --shape NZ NY NX name=image.raw [...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity


def load(path, shape):
    if len(shape) != 3 or any(s <= 0 for s in shape):
        raise ValueError("shape must contain three positive dimensions (NZ NY NX)")
    expected_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    actual_bytes = Path(path).stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} bytes, got {actual_bytes}")
    values = np.fromfile(path, dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{path}: expected {int(np.prod(shape))} float32 values, got {values.size}")
    return values.reshape(shape)


def mean_or_none(values):
    return float(values.mean()) if values.size else None


def relative_error(diff, truth):
    norm = float(np.sum(truth ** 2))
    return float(np.sqrt(np.sum(diff ** 2) / norm)) if norm > 0 else None


class Evaluator:
    def __init__(self, truth, mask=None, data_range=None, background=None, hot_mask=None):
        truth = np.asarray(truth)
        if truth.ndim != 3 or not np.all(np.isfinite(truth)) or np.any(truth < 0):
            raise ValueError("truth must be a finite, nonnegative 3D activity volume")
        if not np.any(truth > 0):
            raise ValueError("truth must contain positive activity")
        self.t = truth.astype(np.float64)
        if mask is None:
            body = truth > 0
        else:
            mask = np.asarray(mask)
            if mask.shape != truth.shape or not np.all(np.isfinite(mask)):
                raise ValueError("mask must be finite and have the truth shape")
            body = mask > 0
            if not np.any(body):
                raise ValueError("mask must contain tissue voxels")
        self.body = body
        self.data_range = float(truth.max() if data_range is None else data_range)
        if not np.isfinite(self.data_range) or self.data_range <= 0:
            raise ValueError("data_range must be finite and positive")
        if background is None:
            values, counts = np.unique(truth[body & (truth > 0)], return_counts=True)
            if not values.size:
                raise ValueError("mask must include positive truth activity, or supply background")
            background = values[np.argmax(counts)]
        self.bg = float(background)
        if not np.isfinite(self.bg) or self.bg <= 0:
            raise ValueError("background must be finite and positive")
        if hot_mask is None:
            self.hot = (truth > 1.5 * self.bg) & body
            hot_definition = "truth > 1.5 * background (high-uptake regions; not necessarily lesions)"
        else:
            hot_mask = np.asarray(hot_mask)
            if hot_mask.shape != truth.shape or not np.all(np.isfinite(hot_mask)):
                raise ValueError("hot_mask must be finite and have the truth shape")
            self.hot = hot_mask > 0
            if np.any(self.hot & (~body | (truth <= 0))):
                raise ValueError("hot_mask must lie inside tissue with positive truth activity")
            hot_definition = "explicit hot mask; connected positive components define lesions"
        self.hot_labels, self.n_hot = ndi.label(self.hot)
        self.metric_settings = {
            "data_range": self.data_range, "background": self.bg,
            "tissue_mask": "explicit mask" if mask is not None else "truth > 0",
            "hot_regions": hot_definition, "hot_region_count": int(self.n_hot),
            "hot_region_voxels": int(self.hot.sum()),
            "ssim_window": "3D Gaussian sigma 1.5, 11 voxels",
        }

        self.roi = ndi.find_objects(ndi.binary_dilation(body, iterations=4), max_label=1)[0]

        # Relative tolerance preserves region membership when activity units change.
        is_background = (np.abs(self.t - self.bg) < 1e-6 * self.bg) & body
        self.bg_interior = ndi.binary_erosion(is_background, structure=np.ones((3, 3, 3)), iterations=4)

        lo = ndi.minimum_filter(truth, size=5, mode="nearest")
        hi = ndi.maximum_filter(truth, size=5, mode="nearest")
        self.lo, self.hi = lo.astype(np.float64), hi.astype(np.float64)
        self.edge_band = hi > lo

        hot_shell = ndi.binary_dilation(self.hot, iterations=3) & ~ndi.binary_dilation(self.hot, iterations=1)
        skin_core = ndi.binary_erosion(body, iterations=4)
        self.halo_shell = hot_shell & skin_core & is_background

    def __call__(self, f):
        f = np.asarray(f, dtype=np.float64)
        if f.shape != self.t.shape or not np.all(np.isfinite(f)):
            raise ValueError("reconstruction must be finite and have the truth shape")
        t = self.t
        diff = f - t
        out = {}
        out["nrmse"] = relative_error(diff, t)
        out["nrmse_tissue"] = relative_error(diff[self.body], t[self.body])
        out["mse"] = float(np.mean(diff ** 2))
        out["mae_tissue"] = mean_or_none(np.abs(diff[self.body]))
        out["count_bias"] = float(f.sum() / t.sum() - 1.0)
        fr, tr = f[self.roi], t[self.roi]
        rmse_roi = np.sqrt(((fr - tr) ** 2).mean())
        out["psnr"] = float(20.0 * np.log10(self.data_range / rmse_roi)) if rmse_roi > 0 else None
        out.update(ssim=None, ssim_tissue=None, ssim_edge=None)
        if min(tr.shape) >= 11:
            score, ssim_map = structural_similarity(
                tr, fr, data_range=self.data_range, gaussian_weights=True,
                sigma=1.5, use_sample_covariance=False, full=True)
            out["ssim"] = float(score)
            valid = (slice(5, -5),) * 3
            smap = ssim_map[valid]
            out["ssim_tissue"] = mean_or_none(smap[self.body[self.roi][valid]])
            out["ssim_edge"] = mean_or_none(smap[self.edge_band[self.roi][valid]])
        out["rc_hot"] = out["rc_min"] = None
        if self.n_hot:
            out["rc_hot"] = float(f[self.hot].mean() / t[self.hot].mean())
            labels = range(1, self.n_hot + 1)
            per = ndi.mean(f, self.hot_labels, labels) / ndi.mean(t, self.hot_labels, labels)
            out["rc_min"] = float(np.min(per))
        bgv = f[self.bg_interior]
        out["bg_bias"] = float(bgv.mean() / self.bg - 1.0) if bgv.size else None
        out["bg_cov"] = float(bgv.std() / bgv.mean()) if bgv.size and bgv.mean() > 0 else None
        over = np.maximum(f - self.hi, 0.0)[self.edge_band]
        under = np.maximum(self.lo - f, 0.0)[self.edge_band]
        out["ring"] = float(np.sqrt(np.mean((over + under) ** 2)) / self.bg) if over.size else None
        out["overshoot"] = float(np.sqrt(np.mean(over ** 2)) / self.bg) if over.size else None
        out["undershoot"] = float(np.sqrt(np.mean(under ** 2)) / self.bg) if under.size else None
        out["edge_nrmse"] = relative_error(diff[self.edge_band], t[self.edge_band])
        halo = mean_or_none(f[self.halo_shell])
        out["halo"] = 1.0 - halo / self.bg if halo is not None else None
        return out


COLUMNS = ["nrmse", "ssim", "ssim_tissue", "ssim_edge", "edge_nrmse", "rc_hot", "rc_min", "bg_cov", "ring", "halo"]


def format_table(rows):
    head = "| image | " + " | ".join(COLUMNS) + " |"
    sep = "|---|" + "---:|" * len(COLUMNS)
    lines = [head, sep]
    for name, m in rows:
        cells = []
        for c in COLUMNS:
            v = m.get(c)
            cells.append("—" if v is None else f"{v:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--shape", type=int, nargs=3, required=True, metavar=("NZ", "NY", "NX"))
    ap.add_argument("--json", help="write metrics to this JSON file")
    ap.add_argument("--mask", help="float32 tissue mask (positive = tissue), including cold regions")
    ap.add_argument("--hot-mask", help="float32 true hot-lesion mask (positive = lesion); recommended for smooth uptake")
    ap.add_argument("--data-range", type=float, help="fixed reference intensity range for SSIM/PSNR")
    ap.add_argument("--background", type=float, help="known background level for phantom metrics")
    ap.add_argument("images", nargs="+", help="name=path.raw")
    args = ap.parse_args()

    ev = Evaluator(load(args.truth, args.shape),
                   mask=load(args.mask, args.shape) if args.mask else None,
                   data_range=args.data_range, background=args.background,
                   hot_mask=load(args.hot_mask, args.shape) if args.hot_mask else None)
    rows = []
    for item in args.images:
        name, path = item.rsplit("=", 1)
        rows.append((name, ev(load(path, args.shape))))
    print(format_table(rows))
    print("Hot-region definition: " + ev.metric_settings["hot_regions"])
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({name: m for name, m in rows}, fh, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
