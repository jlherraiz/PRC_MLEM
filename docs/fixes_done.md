# Positron Range Correction — Iterative RL-TV Fixes

## Problem

The iterative positron range correction was degrading the image rapidly with iterations instead of converging to a stable, deblurred result.

## Root causes identified

### Bug 1 (Critical — GPU version): TV kernel race condition

In `positron_range_rl_heterogeneous_optimized.cu`, the original `APPLY_TV_KERNEL` read and wrote the `estimate` array in the same kernel launch with no synchronization between thread blocks. Thread A computing voxel (x,y,z) could read `estimate[x+1,y,z]` while Thread B at (x+1,y,z) had already overwritten it with a new value. This created non-deterministic, inconsistent TV corrections that compounded across iterations.

The CPU cached version (`positron_range_rl_heterogeneous_cached.cu`) did not have this bug — it already used a correct two-pass approach with separate buffers.

### Bug 2 (Significant — both versions): TV denominator floor too aggressive

Both versions used `max(1 - tv_weight * div, 0.25)` as the TV denominator. The floor of 0.25 allowed up to 4x amplification per voxel per iteration. After N iterations this compounds up to 4^N, causing systematic inflation in smooth regions that erodes contrast and amplifies noise patterns.

### Bug 3 (Significant — both versions): RL-TV splitting instability

TV was applied as a separate post-hoc step after each RL multiplicative update. This "split" approach causes the RL step (driving toward the ML solution, which overfits noise) and the TV step (trying to smooth) to fight each other, creating oscillations that grow with iterations.

### Issue 4 (Minor — cached version): entry_eps narrows the effective PSF

When `entry_eps > 0`, small matrix entries are dropped and remaining weights renormalized. This effectively narrows the point spread function — the model thinks the blur is smaller than reality. The RL algorithm then over-sharpens, amplifying high-frequency content (noise) faster than expected.

## Fixes applied

### `positron_range_rl_heterogeneous_optimized.cu` (GPU version)

1. **TV race condition eliminated.** Replaced the single in-place `APPLY_TV_KERNEL` with a two-pass approach:
   - `TV_COMPUTE_NORMALIZED_GRADIENT_KERNEL`: reads `estimate` (read-only), writes normalized gradient components to separate `d_tv_px`, `d_tv_py`, `d_tv_pz` buffers.
   - The divergence is then computed in the joint update kernel from these frozen gradient buffers.
   - Three additional GPU float buffers are allocated for the gradient components.

2. **Joint EM-TV update.** Added `RL_TV_JOINT_UPDATE_KERNEL` that applies the RL correction and TV penalty in a single atomic step:
   ```
   estimate *= correction / (rl_denom * tv_denom)
   ```
   instead of the old split approach (RL step, then separate TV step). When `tv_weight = 0`, the original `RL_UPDATE_KERNEL` is used as before.

3. **TV denominator floor raised.** Changed from `0.25` (4x max amplification) to `0.5` (2x max amplification).

### `positron_range_rl_heterogeneous_cached.cu` (CPU cached version)

1. **Joint EM-TV update.** Added two new functions:
   - `compute_tv_denominator_host()`: computes the per-voxel TV denominator from the current estimate using the existing correct two-pass gradient/divergence approach.
   - `apply_backward_tile_joint_tv()`: applies the RL multiplicative correction divided by the TV denominator in one step.

   The iteration loop now computes the TV denominator first, then passes it to the joint backward tile function. When `tv_weight = 0`, the original `apply_backward_tile` is used as before.

2. **TV denominator floor raised.** Changed from `0.25` to `0.5`.

3. **entry_eps warning.** Prints a warning to stderr when `entry_eps > 0` is used, explaining that it narrows the effective PSF and accelerates noise amplification. Recommendation: use `entry_eps = 0` for best stability.

## Summary table

| Fix | Issue | Version affected | Change |
|-----|-------|-----------------|--------|
| P0 | TV kernel race condition | GPU | Two-pass TV with separate gradient buffers |
| P1 | TV denominator floor too aggressive | Both | Floor raised from 0.25 to 0.5 |
| P2 | RL-TV splitting instability | Both | Joint EM-TV multiplicative update |
| P3 | entry_eps narrows PSF | Cached (CPU) | Runtime warning when entry_eps > 0 |
