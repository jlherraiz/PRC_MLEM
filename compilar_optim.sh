#!/usr/bin/env bash
set -euo pipefail

detect_cuda_arch() {
    if [[ -n "${CUDA_ARCH:-}" ]]; then
        printf '%s\n' "${CUDA_ARCH}"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local compute_cap
        compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')"
        if [[ "${compute_cap}" =~ ^[0-9]+$ ]]; then
            printf 'sm_%s\n' "${compute_cap}"
            return
        fi
    fi

    printf 'sm_52\n'
}

CUDA_TARGET_ARCH="$(detect_cuda_arch)"
NVCC_FLAGS=(-O3 "-arch=${CUDA_TARGET_ARCH}")

if [[ "${CUDA_FAST_MATH:-0}" == "1" ]]; then
    NVCC_FLAGS+=(--use_fast_math)
fi

echo "Building for ${CUDA_TARGET_ARCH} with flags: ${NVCC_FLAGS[*]}"

nvcc "${NVCC_FLAGS[@]}" positron_range_optim.cu -o pr_optim.x
nvcc "${NVCC_FLAGS[@]}" positron_range_adjoint_optim.cu -o pr_adj_optim.x
nvcc "${NVCC_FLAGS[@]}" positron_range_rl_heterogeneous_optimized.cu kernel_optim.cu kernel_adjoint_optim.cu -o pr_rl_hetero_optim.x

echo "Built: pr_optim.x pr_adj_optim.x pr_rl_hetero_optim.x"
