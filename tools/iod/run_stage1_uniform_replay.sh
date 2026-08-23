#!/usr/bin/env bash
# Matched-budget uniform replay baseline for the first COCO increment.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
COCO_ROOT="${COCO_ROOT:-${PROJECT_ROOT}/data/coco}"
SPLIT_ROOT="${SPLIT_ROOT:-${PROJECT_ROOT}/data/coco-iod/40+20x2/order0}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PROJECT_ROOT}/exps/iod/40+20x2/order0/stage0_base/checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/exps/iod/40+20x2/order0/stage1_uniform_replay}"
REPLAY_BUDGET="${REPLAY_BUDGET:-}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MASTER_PORT="${MASTER_PORT:-29513}"
SEED="${SEED:-42}"
LR="${LR:-2e-4}"
LR_BACKBONE="${LR_BACKBONE:-2e-5}"
LR_DROP="${LR_DROP:-40}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLASS_EMBED_LR_MULT="${CLASS_EMBED_LR_MULT:-1.0}"
CLIP_MAX_NORM="${CLIP_MAX_NORM:-0.1}"

NEW_ANN="${SPLIT_ROOT}/stage_1/instances_increment_only_train2017.json"
BASE_ANN="${SPLIT_ROOT}/stage_0/instances_train2017.json"
VAL_ANN="${SPLIT_ROOT}/stage_1/instances_val2017.json"
TRAIN_ANN="${OUTPUT_DIR}/annotations/uniform_replay_train2017.json"
MEMORY_IDS="${SPLIT_ROOT}/stage_1/memory_image_ids.json"

for path in "${BASE_CHECKPOINT}" "${NEW_ANN}" "${BASE_ANN}" "${VAL_ANN}" "${MEMORY_IDS}"; do
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: missing ${path}" >&2
        exit 1
    fi
done
if [[ -z "${REPLAY_BUDGET}" ]]; then
    REPLAY_BUDGET="$(${PYTHON_BIN} -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["image_ids"]))' "${MEMORY_IDS}")"
fi
if [[ -e "${OUTPUT_DIR}/training_complete.json" ]]; then
    echo "ERROR: completed output already exists: ${OUTPUT_DIR}" >&2
    exit 1
fi
if [[ ! -d "${COCO_ROOT}/train2017" || ! -d "${COCO_ROOT}/val2017" ]]; then
    echo "ERROR: COCO image directories are missing under ${COCO_ROOT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/annotations"
export PYTHONPATH="${PROJECT_ROOT}/models/ops${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/iod/build_replay_annotation.py" \
    --new-ann "${NEW_ANN}" \
    --base-ann "${BASE_ANN}" \
    --output "${TRAIN_ANN}" \
    --replay-budget "${REPLAY_BUDGET}" \
    --seed "${SEED}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
echo "Starting matched-budget uniform replay"
echo "  GPUs: ${GPU_LIST}  replay budget: ${REPLAY_BUDGET}"
echo "  output: ${OUTPUT_DIR}"
echo "  epochs: ${EPOCHS}  lr: ${LR}  backbone lr: ${LR_BACKBONE}"
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
    --lr "${LR}" \
    --lr_backbone "${LR_BACKBONE}" \
    --lr_drop "${LR_DROP}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --class_embed_lr_mult "${CLASS_EMBED_LR_MULT}" \
    --clip_max_norm "${CLIP_MAX_NORM}" \
    --backbone resnet50 \
    --num_queries 300 \
    --enc_layers 6 \
    --dec_layers 6 \
    --seed "${SEED}" \
    --device cuda
