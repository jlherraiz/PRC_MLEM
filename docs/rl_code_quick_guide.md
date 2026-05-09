---
title: "RL Positron Range Code: Quick Guide"
date: "2026-03-18"
geometry: margin=1in
fontsize: 11pt
---

# What This Code Does

This repository contains CUDA tools for positron range correction in PET, including
deterministic forward and adjoint projectors and Richardson-Lucy with total variation
(RL-TV) for heterogeneous density maps.

Supported isotopes:

- `11C`
- `13N`
- `15O`
- `18F`
- `64Cu`
- `68Ga`
- `82Rb`
- `124I`

# Which Variant To Use

| Tool | Purpose | When to use it |
|---|---|---|
| `pr.x` | Forward projector only | Simulate blur from an activity image |
| `pr_adj.x` | Adjoint projector only | Backproject or test the adjoint |
| `pr_rl_hetero.x` | Standard heterogeneous RL-TV | Baseline RL implementation |
| `pr_rl_hetero_optim.x` | Optimized heterogeneous RL-TV | Best choice for dense volumes and the tested Ga-68 mouse case |
| `pr_rl_hetero_cache.x` | Sparse cached RL-TV | Best candidate for large sparse volumes when cache build cost can be amortized |

For the tested `68Ga` mouse volume described as `(z,y,x) = 159 x 128 x 128`,
the best tested end-to-end RL executable was `pr_rl_hetero_optim.x`.
The code itself expects dimensions as `RESX RESY NZS`, so this case is run as:

`128 128 159`

# Input Files

The CUDA executables always read these filenames from the current working directory:

- `ACTIVITY.raw`
- `DENSITY.raw`

Format:

- `float32`
- C-order volume layout
- geometry provided on the command line as `RESX RESY NZS`
- voxel sizes `DX DY DZ` in mm

If your dataset is described in `(z,y,x)` order, convert it mentally before running.
Example: `(159, 128, 128)` in `(z,y,x)` becomes `128 128 159` for the code.

# How To Compile

Standard build:

```bash
bash compilar.sh
```

This builds:

- `pr.x`
- `pr_adj.x`
- `pr_rl_hetero.x`
- `pr_rl_hetero_cache.x`

Optimized build:

```bash
bash compilar_optim.sh
```

This builds:

- `pr_optim.x`
- `pr_adj_optim.x`
- `pr_rl_hetero_optim.x`

`compilar_optim.sh` auto-detects the local GPU architecture. If you want to force it:

```bash
CUDA_ARCH=sm_86 bash compilar_optim.sh
```

# Main RL Command

Optimized RL executable:

```text
./pr_rl_hetero_optim.x ISOTOPE RESX RESY NZS DX DY DZ NRAYS ITERATIONS TV_WEIGHT [TV_EPS] [EPSILON] [VERBOSE]
```

Parameters:

- `ISOTOPE`: one of the supported isotope names, for example `68Ga`
- `RESX RESY NZS`: image size in voxels
- `DX DY DZ`: voxel size in mm
- `NRAYS`: number of sampled directions
- `ITERATIONS`: RL iterations
- `TV_WEIGHT`: TV regularization weight
- `TV_EPS`: optional TV stabilizer, default `1e-3`
- `EPSILON`: optional numerical floor, default `1e-6`
- `VERBOSE`: optional `0` or `1`

Practical starting points:

- `NRAYS = 256` often works for shorter-range isotopes such as `18F`
- `NRAYS = 512` or `1024` is more appropriate for `68Ga` and `82Rb`
- `TV_WEIGHT = 0.002` is a good first trial
- `ITERATIONS = 1` to `4` is a reasonable range for real data

# Examples

## 1. One-command run for the tested Ga-68 mouse case

This repository includes a helper script for the exact dataset used in benchmarking:

```bash
./run_ga68_mouse_optim_4iter.sh
```

That script:

- links the fixed input filenames into a dedicated run directory
- builds `pr_rl_hetero_optim.x` if needed
- runs the optimized RL code with `4` iterations

If the bundle includes the example data, the launcher uses:

`example_data/ga68_mouse_159x128x128/`

Otherwise it falls back to:

`/RAID/mpet/INVEON/nuccuda2/ANALYZED_MSKCC/MOUSE_68Ga_FH/output_mPET_last/`

You can override the input files without editing the script:

```bash
ACTIVITY_SRC_OVERRIDE=/path/to/activity.raw \
DENSITY_SRC_OVERRIDE=/path/to/DENSITY.raw \
./run_ga68_mouse_optim_4iter.sh
```

Output files:

- `ACTIVITY_PRC_RL_HETERO_OPT.raw`
- `ACTIVITY_PRC_RL_HETERO_OPT_REPROJ.raw`

## 2. Direct command for the same Ga-68 case

```bash
mkdir -p run_ga68_mouse
cd run_ga68_mouse
ln -sfn /RAID/mpet/INVEON/nuccuda2/ANALYZED_MSKCC/MOUSE_68Ga_FH/output_mPET_last/Double_img_final.raw ACTIVITY.raw
ln -sfn /RAID/mpet/INVEON/nuccuda2/ANALYZED_MSKCC/MOUSE_68Ga_FH/output_mPET_last/DENSITY.raw DENSITY.raw
../pr_rl_hetero_optim.x 68Ga 128 128 159 0.776 0.776 0.796 512 4 0.002 1e-3 1e-6 0
```

## 3. Forward projector only

```bash
./pr_optim.x 68Ga 128 128 159 0.776 0.776 0.796 512
```

Output:

- `ACTIVITY_PRC.raw`

## 4. Adjoint projector only

```bash
./pr_adj_optim.x 68Ga 128 128 159 0.776 0.776 0.796 512
```

Output:

- `ACTIVITY_PRC_ADJOINT.raw`

## 5. Sparse cached RL-TV

Use this when the volume is large and sparse enough that building a cached operator is worth it.

```text
./pr_rl_hetero_cache.x ISOTOPE RESX RESY NZS DX DY DZ NRAYS ITERATIONS TV_WEIGHT TILE_SOURCES [TV_EPS] [EPSILON] [VERBOSE] [SUPPORT_EPS] [ENTRY_EPS]
```

Extra parameters:

- `TILE_SOURCES`: source voxel batch size used while building the cache
- `SUPPORT_EPS`: optional threshold to drop weak source voxels
- `ENTRY_EPS`: optional threshold to drop weak matrix coefficients per cached row

# Outputs

Typical RL outputs:

- corrected image
- reprojection of the corrected image
- console timing information such as total runtime and time per iteration

File names depend on the executable:

- `pr_rl_hetero.x` and `pr_rl_hetero_optim.x`
  - `ACTIVITY_PRC_RL_HETERO_OPT.raw`
  - `ACTIVITY_PRC_RL_HETERO_OPT_REPROJ.raw`
- `pr_rl_hetero_cache.x`
  - `ACTIVITY_PRC_RL_HETERO_CACHE.raw`
  - `ACTIVITY_PRC_RL_HETERO_CACHE_REPROJ.raw`

# Recommended Starting Point

If you want a simple answer:

1. build with `bash compilar_optim.sh`
2. run `./run_ga68_mouse_optim_4iter.sh`

That is the most user-friendly path for the tested `68Ga` mouse example in this repository.
