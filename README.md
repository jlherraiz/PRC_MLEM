# PRC_MLEM

Heterogeneous positron-range correction for 3D PET images, in one CUDA executable.
The solver uses full-data MAP-EM (Richardson–Lucy with a relative-difference prior)
and Biggs–Andrews acceleration. It does **not** use ordered subsets.

## Build and run

Requires an NVIDIA GPU and the CUDA toolkit (`nvcc`; tested with CUDA 11.6).
Python 3.9+ is needed only for tests and evaluation.

```bash
bash build.sh                         # detects the local GPU
# CUDA_ARCH=sm_86 bash build.sh        # explicit architecture if needed

./prc 68Ga 128 128 159 0.776 0.776 0.795 \
  --input ACTIVITY.raw --density DENSITY.raw --output CORRECTED.raw \
  --beta 0.03 --gamma 20 --iterations 6
```

Arguments are `ISOTOPE NX NY NZ DX DY DZ`, with voxel spacing in millimetres.
Inputs are headerless little-endian float32 arrays, X varying fastest
(NumPy shape `NZ, NY, NX`). PET and density must share the same grid.
Density is mass density in g/cm³, not CT Hounsfield units.

Supported isotopes: `11C`, `13N`, `15O`, `18F`, `64Cu`, `68Ga`, `82Rb`, `124I`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--iterations` | `20` | Full-data iterations |
| `--beta` | `0.2` | Prior strength; `0` disables the prior |
| `--gamma` | `20` | Relative-difference prior edge parameter |
| `--rays` | `128` | Far-field rays; must be a positive even number |
| `--support-rho` | `0.05` | Minimum density retaining activity |
| `--no-accel` | off | Disable extrapolation |
| `--mode` | `correct` | Also supports `forward` and `adjoint` |

Run `./prc` to list all options. Use `--beta 0 --no-accel` for plain
Richardson–Lucy. `--save-every K` saves intermediate estimates;
`--reproject` also writes the corrected image passed through the blur model.

The example uses a weak prior and six iterations; it is not a universal optimum.
More iterations can increase noise and overshoot. Counts are conserved within
the density support; input activity outside it is discarded. The accelerated,
count-normalized update has no general monotonic-convergence guarantee.

## Checks

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

Tests cover the forward/adjoint identity, count conservation, the homogeneous
water kernel, acceleration, input validation and image metrics. GPU tests require
a built executable and available GPU.

With a known reference, `tools/evaluate.py` reports NRMSE, 3D SSIM and local-range
violations (an overshoot/undershoot proxy, not a ringing-specific measure):

```bash
python3 tools/evaluate.py --truth REFERENCE.raw --shape 159 128 128 \
  before=ACTIVITY.raw after=CORRECTED.raw
```

This version replaces the earlier multiple binaries and TV prior; the previous
implementation remains in Git history. No image data or compiled binaries are included.
Range model: Paneque-Yunta et al., *Physics in Medicine & Biology* **70** (2025), 165013.
[MIT license](LICENSE).
