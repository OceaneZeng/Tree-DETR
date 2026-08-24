#!/usr/bin/env bash
# Explicit entry point for the CCF-A CL-DETR-style global replay baseline.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/exps/iod/40+20x2/order0/stage1_cl_detr_global_replay}"
exec bash "${SCRIPT_DIR}/run_stage1_uniform_replay.sh" "$@"
