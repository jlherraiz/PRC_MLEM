#!/usr/bin/env bash
# Reference run: 4-iteration heterogeneous RL-TV correction on the bundled
# Ga-68 mouse INVEON example (128x128x159, 0.776 mm).
#
# This script does NOT ship a dataset. Place your own ACTIVITY.raw and
# DENSITY.raw under example_data/ga68_mouse_159x128x128/, or override the paths
# via ACTIVITY_SRC_OVERRIDE / DENSITY_SRC_OVERRIDE. See DATA.md for details.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
EXAMPLE_DIR="${REPO_DIR}/example_data/ga68_mouse_159x128x128"

DEFAULT_ACTIVITY_SRC="${EXAMPLE_DIR}/Double_img_final.raw"
DEFAULT_DENSITY_SRC="${EXAMPLE_DIR}/DENSITY.raw"

ACTIVITY_SRC="${ACTIVITY_SRC_OVERRIDE:-${DEFAULT_ACTIVITY_SRC}}"
DENSITY_SRC="${DENSITY_SRC_OVERRIDE:-${DEFAULT_DENSITY_SRC}}"
RUN_DIR="${1:-${REPO_DIR}/run_ga68_mouse_optim_4iter}"

if [[ ! -f "${ACTIVITY_SRC}" || ! -f "${DENSITY_SRC}" ]]; then
    cat >&2 <<EOF
Cannot find example dataset.

Expected:
  ACTIVITY: ${ACTIVITY_SRC}
  DENSITY:  ${DENSITY_SRC}

This repository does not ship the data. Place your float32 raw files under:
  ${EXAMPLE_DIR}/
or override the inputs:
  ACTIVITY_SRC_OVERRIDE=/path/to/ACTIVITY.raw \\
  DENSITY_SRC_OVERRIDE=/path/to/DENSITY.raw \\
  ${BASH_SOURCE[0]}

See DATA.md for the expected geometry and a description of the reference case.
EOF
    exit 1
fi

if [[ ! -x "${REPO_DIR}/pr_rl_hetero_optim.x" ]]; then
    echo "Building optimized binary..."
    (cd "${REPO_DIR}" && bash compilar_optim.sh)
fi

mkdir -p "${RUN_DIR}"
ln -sfn "${ACTIVITY_SRC}" "${RUN_DIR}/ACTIVITY.raw"
ln -sfn "${DENSITY_SRC}"  "${RUN_DIR}/DENSITY.raw"

echo "Running optimized heterogeneous RL-TV in ${RUN_DIR}"
(
    cd "${RUN_DIR}"
    "${REPO_DIR}/pr_rl_hetero_optim.x" 68Ga 128 128 159 0.776 0.776 0.796 512 4 0.002 1e-3 1e-6 0
)

echo "Outputs:"
echo "  ${RUN_DIR}/ACTIVITY_PRC_RL_HETERO_OPT.raw"
echo "  ${RUN_DIR}/ACTIVITY_PRC_RL_HETERO_OPT_REPROJ.raw"
