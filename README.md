# PRC_MLEM — Heterogeneous positron range correction in CUDA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GPU-accelerated **heterogeneous positron range correction (PRC)** for PET
images. The repository ships:

- a deterministic **forward** projector `pr.x` and matched **adjoint**
  `pr_adj.x` based on the Paneque-Yunta 2025 multi-branch positron-range model
  (8 isotopes, density-weighted distance through a 3D CUDA texture),
- a single-binary GPU **MLEM / Richardson-Lucy with Total Variation**
  (`pr_rl_hetero.x`) using a joint EM-TV multiplicative update,
- a CPU-cached sparse SpMV variant for small/sparse volumes
  (`pr_rl_hetero_cache.x`),
- Python drivers and convergence tests around the same kernels.

The iterative path conserves total counts exactly at every iteration and
converges monotonically; on a 48³ heterogeneous Ga-68 phantom with a water/lung
interface, the relative L₂ error vs ground truth drops from **0.285 (blurred)
to 0.010 (k=20)** while the Pearson correlation rises from **0.971 to 0.99995**.
A clinical-scale 400 × 400 × 109 phantom converges in **≈27 min for 5 iterations
on an RTX 3090**.

## Repository layout

```
.
├── README.md                                 # this file
├── LICENSE                                   # MIT
├── DATA.md                                   # how to obtain / arrange input data
├── requirements.txt                          # Python deps for the helpers
│
├── compilar.sh                               # default build (sm_52)
├── compilar_optim.sh                         # auto-detects local arch (sm_XX)
├── run_ga68_mouse_optim_4iter.sh             # reference run launcher
│
├── kernel.h, kernel.cu                       # forward kernel (standard)
├── kernel_optim.cu                           # forward kernel (refactored)
├── kernel_adjoint.h, kernel_adjoint.cu       # adjoint kernel (standard)
├── kernel_adjoint_optim.cu                   # adjoint kernel (refactored)
├── positron_range_host_common.hpp            # shared isotope models & utils
│
├── positron_range.cu                         # standalone forward driver -> pr.x
├── positron_range_adjoint.cu                 # standalone adjoint driver -> pr_adj.x
├── positron_range_optim.cu                   # optim variants -> pr_optim.x ...
├── positron_range_adjoint_optim.cu
├── positron_range_rl_heterogeneous_optimized.cu  # GPU RL-TV   -> pr_rl_hetero(_optim).x
├── positron_range_rl_heterogeneous_cached.cu     # CPU-cached SpMV RL-TV -> pr_rl_hetero_cache.x
│
├── rl_positron_correction.py                 # spatially-invariant kernel RL-TV (Python)
├── rl_positron_correction_heterogeneous.py   # heterogeneous RL-TV via pr.x / pr_adj.x
├── benchmark_rl_hetero_cache_isotopes.py     # multi-isotope benchmark for the cached solver
├── match_raw_sum.py                          # rescales counts to a reference
├── test_deblurring_convergence.py            # end-to-end convergence test (48^3)
│
├── deblurring_convergence_results.json       # frozen reference numbers
└── docs/
    ├── iterative_rl_tv_report.md             # full method + validation + timing
    ├── rl_code_quick_guide.md                # short user-facing instructions
    └── fixes_done.md                         # bug-fix narrative (P0..P3)
```

## Requirements

- Linux with an NVIDIA GPU and a working CUDA toolkit (`nvcc`).
- Python ≥ 3.10 for the helpers (`numpy`, `scipy`).

```bash
python3 -m pip install -r requirements.txt
```

## Build

```bash
bash compilar.sh         # sm_52 generic build (pr.x, pr_adj.x, pr_rl_hetero.x, pr_rl_hetero_cache.x)
bash compilar_optim.sh   # auto-detects local arch via nvidia-smi (recommended)
```

`compilar_optim.sh` honours two environment variables:

```bash
CUDA_ARCH=sm_86 bash compilar_optim.sh   # force the target arch
CUDA_FAST_MATH=1 bash compilar_optim.sh  # add --use_fast_math
```

## Quick start

The repository does **not** ship input data — see [DATA.md](DATA.md). Once you
have `ACTIVITY.raw` (blurred) and `DENSITY.raw` of matching geometry:

```bash
# 1. Heterogeneous RL-TV in a single CUDA executable (recommended)
./pr_rl_hetero.x 68Ga 400 400 109 2.0 2.0 2.0 512 5 0.002

# 2. Forward projector only (simulate the blur from a no-PR image)
./pr.x 68Ga 400 400 109 2.0 2.0 2.0 512

# 3. Adjoint projector only
./pr_adj.x 68Ga 400 400 109 2.0 2.0 2.0 512

# 4. Sparse cached SpMV RL-TV (best for small/sparse volumes)
./pr_rl_hetero_cache.x 68Ga 48 48 48 2.0 2.0 2.0 512 12 0.002 256
```

## Reference run

```bash
# Build the optimised binaries for the local GPU and run 4 iterations
# of heterogeneous RL-TV on a Ga-68 INVEON mouse case.
bash compilar_optim.sh
./run_ga68_mouse_optim_4iter.sh
```

The launcher prints clear instructions if the example dataset is missing; see
[DATA.md](DATA.md) for the expected file layout and the `_OVERRIDE` env vars.

## Supported isotopes

`11C`, `13N`, `15O`, `18F`, `64Cu`, `68Ga`, `82Rb`, `124I` (single-,
two- and three-branch parametrisations from Paneque-Yunta et al., 2025).

## Algorithm in one paragraph

The iterative correction solves `g = A f` with `A` the heterogeneous
positron-range transfer matrix, using the Richardson-Lucy / MLEM update with a
**joint EM-TV** multiplicative correction:

```
f^(k+1) = f^(k) · ( A^T ( g / (A f^(k)) ) ) / ( (A^T 1) · D_TV(f^(k)) )
D_TV(f) = max(1 − λ · div(∇f / |∇f|), 0.5)
```

The TV gradient is computed in a separate read-only pass (no GPU race) and
folded into the same multiplicative update as the RL ratio (no split-step
instability). After each iteration the running estimate is rescaled to preserve
`Σ g`. See [docs/iterative_rl_tv_report.md](docs/iterative_rl_tv_report.md) for
the full derivation, validation tables, and benchmarks.

## Parameter starting points

- `NRAYS`: 256 for short-range isotopes (¹⁸F, ⁶⁴Cu); 512 or 1024 for ⁶⁸Ga
  and ⁸²Rb. 512 is already within 0.1 % of an 8192-ray reference on the tested
  cases.
- `TV_WEIGHT`: `0.002` is a robust default.
- `ITERATIONS`: 1–4 for noisy real data; 5–10 for cleaner Monte Carlo references.

## Citing

If you use this code in academic work, please cite the underlying
range model:

> Paneque-Yunta J. *et al.* "Analytical parametrization of the 3D positron
> range distribution for multi-isotope and heterogeneous media."
> *Phys. Med. Biol.* **70**, 165013 (2025).

## License

[MIT](LICENSE).
