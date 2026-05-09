# Datasets

The repository ships **only source code**. Three input formats are referenced
across the documentation and scripts.

## File format

All volumes are **`float32`, C-order** raw binary files (X varies fastest).
Geometry is provided on the command line as `RESX RESY NZS DX DY DZ`
(voxel counts and voxel sizes in mm). When a dataset is described in `(z,y,x)`
order in the literature, convert to `RESX RESY NZS` before passing to the
binaries.

For every CUDA executable the working directory must contain:

- `ACTIVITY.raw` — the (blurred or true) activity image
- `DENSITY.raw` — the heterogeneous density map (g/cm³, water = 1.0)

The corrected output and its forward reprojection are written to the same
directory as `ACTIVITY_PRC_RL_HETERO_OPT.raw` and
`ACTIVITY_PRC_RL_HETERO_OPT_REPROJ.raw` (or the `_CACHE_` equivalents for the
sparse-cache binary).

## Reference cases used in the validation

These datasets are referenced by the documentation but are NOT included.

### 1. Synthetic 48³ heterogeneous Ga-68 phantom

Geometry: `48 × 48 × 48`, `2.0 × 2.0 × 2.0` mm, `68Ga`.
Density: water (1.0) on the left of `x = 24`, lung-like (0.26) on the right.
Activity: two Gaussian blobs straddling the interface.

Anyone can regenerate this phantom directly from the bundled
`test_deblurring_convergence.py` script — it builds the truth, forward-projects
it with `pr.x`, sum-matches the blurred image, and runs the cached RL-TV solver
across iteration counts 1, 2, 3, 4, 6, 8, 10, 12, 15, 20.

```bash
bash compilar.sh
python3 test_deblurring_convergence.py
```

### 2. Anthropomorphic PenEasy phantom

Geometry: `400 × 400 × 109`, `2.0 × 2.0 × 2.0` mm.
Density: CT-derived heterogeneous map.
Reference: Monte-Carlo Ga-68 reference produced with PenEasy 2024.

This dataset is part of an internal evaluation pipeline at UCM and is not
redistributed here. Replace `ACTIVITY.raw` and `DENSITY.raw` in the working
directory with your own equivalent inputs to reproduce the timing numbers
quoted in `docs/iterative_rl_tv_report.md` (≈27 min for 5 iterations on an
RTX 3090).

### 3. Real Ga-68 INVEON mouse acquisition

Geometry: `128 × 128 × 159`, `0.776 × 0.776 × 0.796` mm, `68Ga`.
Provenance: small-animal INVEON PET acquisition; density map derived from
co-registered CT.

This dataset is also internal to UCM and not redistributed. To run the bundled
`run_ga68_mouse_optim_4iter.sh` launcher with your own data, place the files
under `example_data/ga68_mouse_159x128x128/` as:

```
example_data/ga68_mouse_159x128x128/Double_img_final.raw   # ACTIVITY.raw
example_data/ga68_mouse_159x128x128/DENSITY.raw
```

or override the paths:

```bash
ACTIVITY_SRC_OVERRIDE=/path/to/ACTIVITY.raw \
DENSITY_SRC_OVERRIDE=/path/to/DENSITY.raw \
./run_ga68_mouse_optim_4iter.sh
```

## Pre-processing for real data

For real measured or Monte-Carlo blurred images, rescale the input so that its
total counts match the corresponding no-PR reference before running RL. The
deterministic forward model conserves counts in the image domain, but external
references typically do not.

```bash
python3 match_raw_sum.py \
    --input  blurred.raw \
    --reference truth_no_pr.raw \
    --output blurred_sum_matched.raw
```

The Python heterogeneous driver can do this internally with
`--match-input-sum-to`:

```bash
python3 rl_positron_correction_heterogeneous.py deconvolve \
    --input blurred.raw \
    --density DENSITY.raw \
    --output corrected.raw \
    --match-input-sum-to truth_no_pr.raw \
    --truth truth_no_pr.raw \
    --isotope 68Ga \
    --resx 96 --resy 96 --nzs 96 \
    --dx 1.65 --dy 1.65 --dz 1.65 \
    --nrays 512 --iterations 4 --tv-weight 0.002 \
    --metrics-output correction_metrics.json
```
