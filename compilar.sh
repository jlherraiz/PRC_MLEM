#!/usr/bin/env bash
set -euo pipefail

nvcc -O3 positron_range.cu -o pr.x
nvcc -O3 positron_range_adjoint.cu -o pr_adj.x
nvcc -O3 positron_range_rl_heterogeneous_optimized.cu kernel.cu kernel_adjoint.cu -o pr_rl_hetero.x
nvcc -O3 positron_range_rl_heterogeneous_cached.cu -o pr_rl_hetero_cache.x

echo "Built: pr.x pr_adj.x pr_rl_hetero.x pr_rl_hetero_cache.x"
