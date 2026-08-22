#!/usr/bin/env bash
# Fine-tune on the new increment only, without replay or consolidation.
# This is the empirical forgetting probe used to validate M1.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
COCO_ROOT="${COCO_ROOT:-${PROJECT_ROOT}/data/coco}"
SPLIT_ROOT="${SPLIT_ROOT:-${PROJECT_ROOT}/data/coco-iod/40+20x2/order0}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PROJECT_ROOT}/exps/iod/40+20x2/order0/stage0_base/checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/exps/iod/40+20x2/order0/stage1_unprotected}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MASTER_PORT="${MASTER_PORT:-29512}"
SEED="${SEED:-42}"

TRAIN_ANN="${SPLIT_ROOT}/stage_1/instances_increment_only_train2017.json"
VAL_ANN="${SPLIT_ROOT}/stage_1/instances_val2017.json"
for path in "${BASE_CHECKPOINT}" "${TRAIN_ANN}" "${VAL_ANN}"; do
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: missing ${path}" >&2
        exit 1
    fi
done
if [[ ! -d "${COCO_ROOT}/train2017" || ! -d "${COCO_ROOT}/val2017" ]]; then
    echo "ERROR: COCO image directories are missing under ${COCO_ROOT}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}/training_complete.json" ]]; then
    echo "ERROR: completed output already exists: ${OUTPUT_DIR}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONPATH="${PROJECT_ROOT}/models/ops${PYTHONPATH:+:${PYTHONPATH}}"

echo "Starting unprotected stage-1 forgetting probe"
echo "  GPUs: ${GPU_LIST}  train annotation: ${TRAIN_ANN}"
echo "  base checkpoint: ${BASE_CHECKPOINT}"
echo "  output: ${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
    main.py \
    --coco_path "${COCO_ROOT}" \
    --train-ann "${TRAIN_ANN}" \
    --val-ann "${VAL_ANN}" \
    --output_dir "${OUTPUT_DIR}" \
    --pretrained "${BASE_CHECKPOINT}" \
    --num_classes 91 \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --eval_interval 10 \
    --skip-eval \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --backbone resnet50 \
    --num_queries 300 \
    --enc_layers 6 \
    --dec_layers 6 \
    --seed "${SEED}" \
    --device cuda
