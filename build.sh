#!/usr/bin/env bash
# Builds ./prc for the local GPU (override with CUDA_ARCH=sm_XX).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
arch="${CUDA_ARCH:-}"
if [[ -z "$arch" ]] && command -v nvidia-smi > /dev/null; then
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -n 1 | tr -d '.')" || cap=""
    [[ "$cap" =~ ^[0-9]+$ ]] && arch="sm_${cap}"
fi
arch="${arch:-sm_52}"
echo "nvcc -O3 -arch=${arch} prc.cu -o prc"
nvcc -O3 -arch="${arch}" prc.cu -o prc
