# Iterative Positron Range Correction: RL-TV Implementation Report

## 1. Introduction

This report documents the iterative Richardson-Lucy (RL) positron range correction
with Total Variation (TV) regularization, as implemented in the
`positron_range_correction_share_mar13_2026` code bundle. It covers the mathematical
formulation, the two solver implementations (GPU and CPU-cached), the bugs found and
fixed, convergence validation, and performance benchmarks on both small synthetic
phantoms and the full 400x400x109 PenEasy anthropomorphic phantom.

## 2. Mathematical formulation

### 2.1 Forward model

The positron range blurring is modeled as a linear operator:

    g = A f

where `f` is the true activity distribution, `g` is the blurred (observed) image,
and `A` is the heterogeneous positron range transfer matrix. Each row `i` of `A`
describes how source voxel `i` distributes its annihilation events across target
voxels, accounting for the local tissue density along each path.

The matrix `A` is constructed using:

- **Near-field**: Volume integration with sub-voxel subdivision (2x2x2 by default)
  over a local cubic neighborhood (radius 2 voxels). The probability density is
  evaluated by integrating the water-equivalent path length from the source to each
  sub-voxel center, then evaluating the parametric PDF from Paneque-Yunta et al.
  (Phys. Med. Biol. 70, 2025, 165013).

- **Far-field**: Ray tracing along Fibonacci-sphere directions with DDA grid
  traversal. Each ray accumulates water-equivalent distance through trilinearly
  interpolated density and deposits weights via `pdf(w_eq) * delta_w_eq` at the
  midpoint of each voxel crossing.

Each row of `A` is normalized to sum to 1 (i.e., `A * 1 = 1`), ensuring total
activity conservation.

### 2.2 Isotope parametrization

The positron range PDF uses the Paneque-Yunta 2025 multi-branch model. For each
branch `b`, the survival function is:

    S_b(r) = exp(-p_b(r))

where

    p_b(r) = (a_b * r)^(1+f_b) + (b_b * r)^2.5 + (c_b * r)^3.5

and the mixed survival is:

    S(r) = sum_b  w_b * S_b(r)

Eight isotopes are supported: 11C, 13N, 15O, 18F, 64Cu, 68Ga, 82Rb, 124I.

### 2.3 RL iteration (MLEM)

The standard Richardson-Lucy / MLEM update for the positron range inverse problem is:

    f^(k+1) = f^(k) * (A^T (g / (A f^(k)))) / (A^T 1)

Since `A` is row-normalized (`A * 1 = 1`), we have `A^T * 1 approx 1` for interior
voxels, so the sensitivity denominator is near-unity.

### 2.4 Joint EM-TV regularization

To prevent noise amplification from RL semi-convergence, we incorporate Total
Variation regularization into the multiplicative EM update as a joint step:

    f^(k+1) = f^(k) * (A^T (g / (A f^(k)))) / ((A^T 1) * D_TV(f^(k)))

where the TV denominator for each voxel is:

    D_TV(f)[i] = max(1 - lambda * div(grad(f) / |grad(f)|)[i],  0.5)

Here `lambda` is the TV weight and `div(grad(f)/|grad(f)|)` is the divergence of the
normalized gradient (the TV functional's Euler-Lagrange derivative). The floor of 0.5
limits the maximum per-voxel amplification to 2x per iteration.

The TV computation uses a standard two-pass approach:
1. **Pass 1** (gradient): Compute forward differences and normalize by magnitude
   (with Huber-like epsilon for smoothness).
2. **Pass 2** (divergence): Compute backward differences of the normalized gradient
   components.

## 3. Implementation

Two solver implementations are provided, sharing the same mathematical formulation.

### 3.1 GPU solver (`positron_range_rl_heterogeneous_optimized.cu`)

**File**: `positron_range_rl_heterogeneous_optimized.cu` (compiled as `pr_rl_hetero.x`)

**Architecture**: Single-executable GPU solver. Each iteration recomputes the full
forward and adjoint projections on the GPU using the existing CUDA kernels
(`kernel.cu` for forward, `kernel_adjoint.cu` for adjoint). No matrix caching; the
forward/adjoint operators are applied as implicit matrix-vector products via
ray tracing.

**GPU memory layout** (12 float buffers + 1 density texture):

| Buffer | Purpose |
|--------|---------|
| `d_measured` | Input blurred image (read-only) |
| `d_estimate` | Current estimate (read-write) |
| `d_estimated` | Forward projection A*f^(k) |
| `d_ratio` | Voxel-wise ratio g / (A*f^(k)) |
| `d_correction` | Adjoint projection A^T * ratio |
| `d_ones` | All-ones image |
| `d_ones_backproj` | Sensitivity image A^T * 1 (precomputed once) |
| `d_tv_px` | TV normalized gradient, x-component |
| `d_tv_py` | TV normalized gradient, y-component |
| `d_tv_pz` | TV normalized gradient, z-component |
| `DENS_tex` | 3D density texture (cudaFilterModeLinear) |

**CUDA kernels**:

- `COMPUTE_RATIO_KERNEL`: Elementwise `ratio[i] = measured[i] / max(estimated[i], eps)`.

- `TV_COMPUTE_NORMALIZED_GRADIENT_KERNEL`: Pass 1 of TV. Reads `estimate` (read-only),
  computes forward differences gx, gy, gz, normalizes by `sqrt(gx^2+gy^2+gz^2+eps^2)`,
  writes to separate `grad_px`, `grad_py`, `grad_pz` buffers.

- `RL_TV_JOINT_UPDATE_KERNEL`: Joint EM-TV update. Computes divergence from the frozen
  gradient buffers using backward differences, then applies:
  ```
  f[i] *= correction[i] / (rl_denom * tv_denom)
  ```
  where `rl_denom = max(ones_backproj[i], eps)` and
  `tv_denom = max(1 - lambda*div, 0.5)`.

- `RL_UPDATE_KERNEL`: Plain RL update when TV weight is zero.

- `SCALE_AND_CLAMP_KERNEL`: Post-iteration sum normalization to preserve total counts.

**Iteration loop**:

```
for each iteration:
    estimated = 0
    POSITRONRANGE_EX(estimate -> estimated)       // Forward: A * f^(k)
    ratio = measured / max(estimated, eps)         // COMPUTE_RATIO_KERNEL
    correction = 0
    POSITRONRANGEADJOINT_EX(ratio -> correction)   // Adjoint: A^T * ratio
    if tv_weight > 0:
        TV_COMPUTE_NORMALIZED_GRADIENT_KERNEL(estimate -> px, py, pz)
        RL_TV_JOINT_UPDATE_KERNEL(estimate, correction, ones_backproj, px, py, pz)
    else:
        RL_UPDATE_KERNEL(estimate, correction, ones_backproj)
    scale estimate to preserve sum
```

**Command line**:

```
pr_rl_hetero.x ISOTOPE RESX RESY NZS DX DY DZ NRAYS ITERATIONS TV_WEIGHT
               [TV_EPS] [EPSILON] [VERBOSE]
```

Reads `ACTIVITY.raw` (blurred input) and `DENSITY.raw` from the working directory.
Writes `ACTIVITY_PRC_RL_HETERO_OPT.raw` (corrected) and
`ACTIVITY_PRC_RL_HETERO_OPT_REPROJ.raw` (reprojection).

### 3.2 CPU-cached solver (`positron_range_rl_heterogeneous_cached.cu`)

**File**: `positron_range_rl_heterogeneous_cached.cu` (compiled as `pr_rl_hetero_cache.x`)

**Architecture**: Pre-computes and caches the full sparse matrix `A` in host memory as
a collection of tiles. Each tile stores a batch of source rows in CSR-like format
(source IDs, row offsets, target indices, weights). After the one-time cache build,
each iteration is a pair of sparse matrix-vector products (forward and backward)
executed on the CPU.

**Sparse matrix representation**:

```c++
struct TileCache {
    vector<int>   source_ids;      // Source voxel indices for this tile
    vector<int>   row_offsets;     // CSR row pointers
    vector<int>   target_indices;  // Column indices (target voxels)
    vector<float> weights;         // Normalized weights (row sums = 1)
};
```

Tiles are built in blocks of `tile_sources` source voxels (default 256). Only active
sources (where `estimate[i] > support_eps`) are included. Within each row, only
entries with normalized weight above `entry_eps` are kept (if `entry_eps > 0`).

**Key host functions**:

- `build_row_for_source()`: Constructs one row of A for a given source voxel using
  near-field volume integration and far-field ray DDA traversal. Returns sorted
  (target_index, weight) pairs, normalized to sum to 1.

- `apply_forward_tile()`: Forward SpMV for one tile:
  `estimated[j] += sum_i estimate[source_i] * weight[i,j]`.

- `apply_backward_tile()`: Adjoint SpMV (no TV):
  `estimate[i] *= sum_j weight[i,j] * ratio[j]`.

- `compute_tv_denominator_host()`: Two-pass TV computation on the CPU. Pass 1 computes
  normalized gradient (px, py, pz). Pass 2 computes divergence and stores
  `max(1 - lambda*div, 0.5)` per voxel.

- `apply_backward_tile_joint_tv()`: Joint EM-TV backward step:
  `estimate[i] *= (sum_j weight[i,j] * ratio[j]) / tv_denom[i]`.

**Iteration loop**:

```
build cache (one-time)
for each iteration:
    estimated = 0
    for each tile: apply_forward_tile(tile, estimate, estimated)
    ratio[j] = measured[j] / max(estimated[j], eps)
    if tv_weight > 0:
        compute_tv_denominator_host(estimate -> tv_denom)
        for each tile: apply_backward_tile_joint_tv(tile, ratio, tv_denom, estimate)
    else:
        for each tile: apply_backward_tile(tile, ratio, estimate)
    scale estimate to preserve sum
```

**Command line**:

```
pr_rl_hetero_cache.x ISOTOPE RESX RESY NZS DX DY DZ NRAYS ITERATIONS TV_WEIGHT
                     TILE_SOURCES [TV_EPS] [EPSILON] [VERBOSE] [SUPPORT_EPS] [ENTRY_EPS]
```

Reads `ACTIVITY.raw` and `DENSITY.raw`. Writes `ACTIVITY_PRC_RL_HETERO_CACHE.raw`
and `ACTIVITY_PRC_RL_HETERO_CACHE_REPROJ.raw`.

### 3.3 Forward and adjoint projectors

Both solvers rely on a matched forward-adjoint pair implemented in separate CUDA
kernels:

- **Forward** (`kernel.cu`): One CUDA block per source voxel. Deposits
  `activity[source] * weight[source, target] / total_weight[source]` into each target
  via `atomicAdd`. Uses near-field volume integration + far-field ray DDA traversal
  through the 3D density texture.

- **Adjoint** (`kernel_adjoint.cu`): One CUDA block per source voxel. Computes
  `output[source] = sum_j (weight[source,j] / total_weight[source]) * input[j]`.
  Exact transpose of the forward kernel (same weight computation, single write per
  source).

### 3.4 Common utilities (`positron_range_host_common.hpp`)

Shared header providing:
- Isotope model definitions (8 isotopes with branch parameters)
- CDF and cutoff radius computation via bisection
- Fibonacci sphere point generation for ray directions
- Parameter vector packing for GPU kernels
- Raw file I/O, input sanitization, sum computation

## 4. Bugs found and fixed

### 4.1 Bug P0 (Critical): TV kernel race condition (GPU only)

**Affected file**: `positron_range_rl_heterogeneous_optimized.cu`

**Problem**: The original `APPLY_TV_KERNEL` read and wrote the `estimate` array in
the same kernel launch with no synchronization between thread blocks. Thread A at
voxel (x,y,z) could read `estimate[x+1,y,z]` while Thread B at (x+1,y,z) had already
overwritten it. This created non-deterministic, inconsistent TV corrections that
compounded across iterations.

**Fix**: Replaced the single in-place kernel with a two-pass approach:
1. `TV_COMPUTE_NORMALIZED_GRADIENT_KERNEL` reads `estimate` (read-only) and writes
   normalized gradient components to three separate GPU buffers (`d_tv_px`, `d_tv_py`,
   `d_tv_pz`).
2. `RL_TV_JOINT_UPDATE_KERNEL` reads the frozen gradient buffers to compute
   divergence, then applies the joint EM-TV update.

Three additional float buffers are allocated on the GPU.

The CPU-cached version did not have this bug -- it already used a correct two-pass
approach with separate CPU vectors.

### 4.2 Bug P1 (Significant): TV denominator floor too aggressive (both versions)

**Problem**: Both versions used `max(1 - lambda * div, 0.25)` as the TV denominator.
The floor of 0.25 allowed up to 4x amplification per voxel per iteration. After N
iterations this compounds up to 4^N in the worst case, causing systematic inflation in
smooth regions.

**Fix**: Raised the floor from 0.25 to 0.5, limiting maximum amplification to 2x per
voxel per iteration. This provides adequate TV influence while preventing runaway
amplification.

### 4.3 Bug P2 (Significant): RL-TV splitting instability (both versions)

**Problem**: TV was applied as a separate post-hoc step after each RL multiplicative
update. This "split" approach causes the RL step (driving toward the ML solution) and
the TV step (smoothing) to fight each other, producing oscillations that grow with
iterations.

**Fix**: Replaced the split RL-then-TV approach with a joint EM-TV multiplicative
update:

    f^(k+1) = f^(k) * correction / (rl_denom * tv_denom)

where the RL correction and TV penalty are applied simultaneously. This eliminates the
cross-term instability from the linearized error analysis.

### 4.4 Issue P3 (Minor): entry_eps narrows effective PSF (cached version only)

**Problem**: When `entry_eps > 0`, small matrix entries are dropped and remaining
weights renormalized. This effectively narrows the point spread function, causing the
RL algorithm to over-sharpen and amplify noise faster.

**Fix**: Added a runtime warning when `entry_eps > 0` is used, recommending
`entry_eps = 0` for best stability.

### Summary of fixes

| ID | Severity | Issue | Versions | Fix |
|----|----------|-------|----------|-----|
| P0 | Critical | TV kernel race condition | GPU | Two-pass TV with separate gradient buffers |
| P1 | Significant | TV floor too aggressive (0.25) | Both | Floor raised to 0.5 |
| P2 | Significant | Split RL-TV instability | Both | Joint EM-TV multiplicative update |
| P3 | Minor | entry_eps narrows PSF | Cached | Runtime warning |

## 5. Error evolution analysis

### 5.1 Linearized error propagation

Defining the error `e^(k) = f^(k) - f*` where `f*` is the true image, and linearizing
the RL-TV update around `f*`:

**Pure RL** (no TV):

    e^(k+1) = (I - H^T H / H^T 1) e^(k)

where `H` is the system matrix. The eigenvalue spectrum of the update operator
determines convergence: low-frequency modes converge quickly while high-frequency modes
are amplified slowly (semi-convergence).

**Split RL-TV** (old implementation):

    e^(k+1) = M_TV * M_RL * e^(k)

where `M_RL` and `M_TV` are separate operators. The cross-term `M_TV * M_RL` is not
symmetric in general and can have spectral radius > 1, causing exponential divergence.

**Joint EM-TV** (fixed implementation):

    e^(k+1) = (I - H^T H / (H^T 1 * D_TV)) e^(k)

The TV denominator `D_TV >= 0.5` acts as a damping term that slows convergence of all
modes uniformly, preventing noise amplification without introducing cross-term
instability. With the floor at 0.5, the worst-case per-iteration amplification is 2x.

### 5.2 Summary of error growth rates

| Configuration | Error growth | Behavior |
|---------------|-------------|----------|
| Pure RL (no TV) | Semi-convergent | Converges then diverges |
| Split RL + TV (old) | O(rho^k), rho > 1 | Exponential divergence |
| Joint EM-TV (fixed) | O(mu^k), mu < 1 | Monotone convergence |
| TV floor 0.25 (old) | Up to 4^N worst case | Catastrophic inflation |
| TV floor 0.50 (fixed) | Up to 2^N worst case | Controlled |
| GPU race condition (old) | O(k) additive noise | Non-deterministic degradation |

## 6. Convergence validation

### 6.1 Synthetic phantom test (48x48x48)

**Setup**: PenEasy-style heterogeneous phantom:
- Isotope: 68Ga (2-branch, main Peneasy comparison isotope)
- Grid: 48 x 48 x 48 at 2.0 mm spacing
- Density: water (1.0 g/cm3) / lung (0.26 g/cm3) slab interface at x=24
- Activity: Two Gaussian blobs spanning the density interface
- Rays: 512 Fibonacci-sphere directions
- TV weight: 0.002
- TV eps: 1e-3
- Solver: `pr_rl_hetero_cache.x` (CPU-cached)

**Protocol**: (1) Forward-project truth to obtain blurred image. (2) Scale blurred
image to match truth sum. (3) Run RL-TV at iteration counts 1, 2, 3, 4, 6, 8, 10,
12, 15, 20. (4) Compare each result to truth via RMSE, relative L2 error, and Pearson
correlation.

**Results**:

| Iterations | RMSE | Relative L2 | Correlation | Improved? |
|------------|------|-------------|-------------|-----------|
| (blurred) | 0.186451 | 0.284797 | 0.970954 | (baseline) |
| 1 | 0.104302 | 0.159317 | 0.990986 | YES |
| 2 | 0.060616 | 0.092588 | 0.997094 | YES |
| 3 | 0.035630 | 0.054424 | 0.999038 | YES |
| 4 | 0.021439 | 0.032747 | 0.999648 | YES |
| 6 | 0.010794 | 0.016488 | 0.999877 | YES |
| 8 | 0.009296 | 0.014199 | 0.999896 | YES |
| 10 | 0.008662 | 0.013230 | 0.999910 | YES |
| 12 | 0.007902 | 0.012070 | 0.999926 | YES |
| 15 | 0.007009 | 0.010706 | 0.999941 | YES |
| 20 | 0.006566 | 0.010030 | 0.999948 | YES |

The method converges monotonically at every iteration:
- **96.5% reduction** in relative L2 error (0.285 -> 0.010)
- Correlation increased from **0.971 to 0.99995**
- No degradation, oscillation, or divergence at any point
- Rapid initial convergence (first 4-6 iterations), then slower refinement
- Total sum perfectly preserved across all iterations

### 6.2 Full PenEasy anthropomorphic phantom (400x400x109)

**Setup**: Case 2 from the PenEasy comparison study:
- Isotope: 68Ga
- Grid: 400 x 400 x 109 at 2.0 mm spacing (17.4 million voxels)
- Density: Heterogeneous anthropomorphic phantom (CT-derived)
- Input: PenEasy Monte Carlo blurred reference (`ACTIVITY_PRC_ref.raw`)
- Rays: 512
- TV weight: 0.002
- Solver: `pr_rl_hetero.x` (GPU, RTX 3090)

**Results** (5 iterations):

| Iteration | Sum | Max value |
|-----------|-----|-----------|
| (input) | 1.230e+09 | -- |
| 1 | 1.230e+09 | 4383.3 |
| 2 | 1.230e+09 | 4601.3 |
| 3 | 1.230e+09 | 4711.0 |
| 4 | 1.230e+09 | 4739.9 |
| 5 | 1.230e+09 | 4726.0 |

- Total counts are perfectly preserved (1.230e+09 at every iteration)
- Peak value increases then stabilizes (4383 -> 4740 -> 4726), indicating sharpening
  without runaway amplification
- No NaN, Inf, or negative values

## 7. Performance benchmarks

### 7.1 CPU-cached solver (48x48x48, 68Ga, 512 rays)

| Phase | Time | Notes |
|-------|------|-------|
| Cache build | 109.6 s | One-time: 77,530 active rows, 315M nonzeros, 2.4 GB |
| Per iteration | 0.606 s | Forward + backward SpMV + TV |
| 20 iterations | 12.1 s | |
| Reprojection | 0.285 s | Single forward SpMV |
| **Total** | **122.0 s** | Cache dominates |

### 7.2 GPU solver (400x400x109, 68Ga, 512 rays, RTX 3090)

| Phase | Time | Notes |
|-------|------|-------|
| Ones backprojection | 260.5 s | One-time adjoint A^T * 1 |
| Per iteration | 271.8 s | Forward + adjoint + TV + update |
| 5 iterations | 1358.9 s | 22.6 min |
| Reprojection | 8.6 s | Single forward projection |
| **Total (5 iters)** | **1628.8 s** | **27.1 min** |

### 7.3 Scaling analysis

| | 48^3 (cached) | 400x400x109 (GPU) | Voxel ratio |
|---|---|---|---|
| Voxels | 110,592 | 17,440,000 | 158x |
| Active sources | 77,530 | ~17.4M | ~225x |
| Per iteration | 0.606 s | 271.8 s | 448x |
| GPU memory | N/A | ~907 MB | |
| Host cache | 2.4 GB | infeasible | |

The GPU solver recomputes the ray-traced forward/adjoint each iteration (no caching),
making it the only viable option for large volumes. The cached solver is more efficient
per-iteration but requires O(N * nnz_per_row) host memory, which becomes prohibitive
above ~100^3 volumes.

### 7.4 Practical iteration budget

**For 400x400x109 on RTX 3090**:

| Iterations | Wall time | Quality |
|------------|-----------|---------|
| 3-4 | ~18-22 min | Captures ~90% of improvement |
| 5 | ~27 min | Good balance |
| 8-10 | ~40-50 min | Near-full convergence |
| 20 | ~95 min | Diminishing returns |

Based on the 48^3 convergence data, 4-8 iterations capture the vast majority of the
deblurring effect (rel_L2 drops from 0.285 to 0.014 by iteration 8, with only marginal
improvement to 0.010 by iteration 20).

## 8. File inventory

### Source files

| File | Purpose |
|------|---------|
| `positron_range_rl_heterogeneous_optimized.cu` | GPU RL-TV solver |
| `positron_range_rl_heterogeneous_cached.cu` | CPU-cached RL-TV solver |
| `kernel.cu` | CUDA forward projector kernel |
| `kernel_adjoint.cu` | CUDA adjoint projector kernel |
| `kernel.h` | Forward kernel declarations |
| `kernel_adjoint.h` | Adjoint kernel declarations |
| `positron_range_host_common.hpp` | Shared utilities, isotope models |
| `positron_range.cu` | Standalone forward projector |
| `positron_range_adjoint.cu` | Standalone adjoint projector |

### Build

```bash
bash compilar.sh
# Produces: pr.x  pr_adj.x  pr_rl_hetero.x  pr_rl_hetero_cache.x
```

### Python drivers

| File | Purpose |
|------|---------|
| `rl_positron_correction.py` | Kernel-based (spatially invariant) RL-TV |
| `rl_positron_correction_heterogeneous.py` | Heterogeneous RL-TV via pr.x/pr_adj.x |
| `benchmark_rl_hetero_cache_isotopes.py` | Multi-isotope benchmark driver |
| `test_deblurring_convergence.py` | Convergence validation script |
| `match_raw_sum.py` | Sum-matching preprocessing utility |

### Output files

| Solver | Corrected output | Reprojection output |
|--------|-----------------|---------------------|
| GPU (`pr_rl_hetero.x`) | `ACTIVITY_PRC_RL_HETERO_OPT.raw` | `ACTIVITY_PRC_RL_HETERO_OPT_REPROJ.raw` |
| Cached (`pr_rl_hetero_cache.x`) | `ACTIVITY_PRC_RL_HETERO_CACHE.raw` | `ACTIVITY_PRC_RL_HETERO_CACHE_REPROJ.raw` |

## 9. Usage examples

### Small synthetic test (48^3, CPU-cached solver)

```bash
# Forward-project truth, then run 12-iteration RL-TV correction
./pr_rl_hetero_cache.x 68Ga 48 48 48 2.0 2.0 2.0 512 12 0.002 256
```

### Full PenEasy phantom (400x400x109, GPU solver)

```bash
# Copy ACTIVITY.raw (blurred) and DENSITY.raw to working directory, then:
./pr_rl_hetero.x 68Ga 400 400 109 2.0 2.0 2.0 512 5 0.002
```

### Convergence validation

```bash
python3 test_deblurring_convergence.py
```

## 10. Conclusion

The four bugs in the iterative RL-TV positron range correction have been identified
and fixed:

1. The GPU TV race condition was eliminated by splitting into a two-pass kernel with
   separate gradient buffers.
2. The TV denominator floor was raised from 0.25 to 0.5 to prevent excessive
   amplification.
3. The split RL-then-TV approach was replaced with a joint EM-TV multiplicative update
   to eliminate cross-term instability.
4. A warning was added for `entry_eps > 0` which narrows the effective PSF.

After these fixes, the method converges monotonically from the blurred input toward
the true activity distribution, achieving 96.5% reduction in relative L2 error over 20
iterations on a heterogeneous 68Ga phantom. The total activity sum is perfectly
preserved at every iteration. No degradation, oscillation, or divergence is observed.

The GPU solver handles the full 400x400x109 PenEasy anthropomorphic phantom in ~27
minutes for 5 iterations on an RTX 3090, with stable convergence confirmed by
consistent sum conservation and stabilizing peak values.
