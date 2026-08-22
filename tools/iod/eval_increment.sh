#!/usr/bin/env bash
# Single-GPU evaluation for an incremental-stage checkpoint.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
STAGE_INDEX="${STAGE_INDEX:-1}"
COCO_ROOT="${COCO_ROOT:-${PROJECT_ROOT}/data/coco}"
SPLIT_ROOT="${SPLIT_ROOT:-${PROJECT_ROOT}/data/coco-iod/40+20x2/order0}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/exps/iod/40+20x2/order0/stage1_unprotected/checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/exps/iod/40+20x2/order0/stage1_unprotected_eval}"

TRAIN_ANN="${SPLIT_ROOT}/stage_${STAGE_INDEX}/instances_train2017.json"
VAL_ANN="${SPLIT_ROOT}/stage_${STAGE_INDEX}/instances_val2017.json"
for path in "${CHECKPOINT}" "${TRAIN_ANN}" "${VAL_ANN}"; do
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: missing ${path}" >&2
        exit 1
    fi
done
mkdir -p "${OUTPUT_DIR}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PROJECT_ROOT}/models/ops${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" main.py \
    --coco_path "${COCO_ROOT}" \
    --train-ann "${TRAIN_ANN}" \
    --val-ann "${VAL_ANN}" \
    --output_dir "${OUTPUT_DIR}" \
    --pretrained "${CHECKPOINT}" \
    --num_classes 91 \
    --eval \
    --batch_size "${BATCH_SIZE:-2}" \
    --num_workers "${NUM_WORKERS:-2}" \
    --device cuda
