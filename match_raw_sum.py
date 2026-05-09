#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescale a float32 RAW image so its total sum matches a reference RAW image."
    )
    parser.add_argument("--input", required=True, help="Input float32 RAW image to rescale.")
    parser.add_argument("--reference", required=True, help="Reference float32 RAW image providing the target sum.")
    parser.add_argument("--output", required=True, help="Output float32 RAW image.")
    return parser.parse_args()


def load_raw(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"{path} is empty.")
    return arr


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    reference_path = Path(args.reference).resolve()
    output_path = Path(args.output).resolve()

    image = load_raw(input_path)
    reference = load_raw(reference_path)
    if image.size != reference.size:
        raise ValueError(
            f"Input and reference must contain the same number of float32 voxels: "
            f"{image.size} vs {reference.size}"
        )

    input_sum = float(image.sum(dtype=np.float64))
    reference_sum = float(reference.sum(dtype=np.float64))
    if input_sum <= 0.0:
        raise ValueError(f"Input sum must be positive, got {input_sum}.")
    if reference_sum <= 0.0:
        raise ValueError(f"Reference sum must be positive, got {reference_sum}.")

    scale = reference_sum / input_sum
    scaled = image * np.float32(scale)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scaled.astype(np.float32, copy=False).tofile(output_path)

    print(f"INPUT_SUM = {input_sum:.6f}")
    print(f"REFERENCE_SUM = {reference_sum:.6f}")
    print(f"SCALE_FACTOR = {scale:.9f}")
    print(f"OUTPUT_SUM = {float(scaled.sum(dtype=np.float64)):.6f}")
    print(f"WROTE = {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
