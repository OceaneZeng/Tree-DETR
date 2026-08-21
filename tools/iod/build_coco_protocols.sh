#!/usr/bin/env bash
# Generate the three fixed category orders used by the IOD protocol.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
COCO_ROOT="${COCO_ROOT:-${PROJECT_ROOT}/data/coco}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data/coco-iod}"
PROTOCOL="${PROTOCOL:-40+20x2}"
SEED="${SEED:-42}"
MEMORY_FRACTION="${MEMORY_FRACTION:-0.10}"

usage() {
    echo "Usage: tools/iod/build_coco_protocols.sh [--protocol 40+20x2|40+10x4|70+10]"
    echo "Environment: COCO_ROOT OUTPUT_ROOT PYTHON_BIN SEED MEMORY_FRACTION"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --protocol) PROTOCOL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "${COCO_ROOT}/annotations/instances_train2017.json" ||
      ! -f "${COCO_ROOT}/annotations/instances_val2017.json" ||
      ! -d "${COCO_ROOT}/train2017" || ! -d "${COCO_ROOT}/val2017" ]]; then
    echo "ERROR: complete COCO 2017 layout not found under ${COCO_ROOT}" >&2
    echo "Required: train2017/, val2017/, annotations/instances_train2017.json," >&2
    echo "         annotations/instances_val2017.json" >&2
    echo "Download it with: bash tools/download_data.sh --root ${COCO_ROOT} --full --skip-ckpt" >&2
    exit 1
fi

for order_index in 0 1 2; do
    order_seed=$((SEED + order_index))
    output="${OUTPUT_ROOT}/${PROTOCOL}/order${order_index}"
    echo "Building ${PROTOCOL} order${order_index} (seed=${order_seed})"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/iod/coco_incremental.py" \
        --coco-root "${COCO_ROOT}" \
        --output-root "${output}" \
        --protocol "${PROTOCOL}" \
        --order random \
        --seed "${order_seed}" \
        --memory-fraction "${MEMORY_FRACTION}"
done

echo "COCO IOD protocol ready: ${OUTPUT_ROOT}/${PROTOCOL}"
