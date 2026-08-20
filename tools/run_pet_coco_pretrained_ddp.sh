#!/usr/bin/env bash
# Run the COCO-pretrained Oxford-Pet baseline on one Linux node with DDP.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/oxford-pet-tree-detr}"
DATASET_NAME="${DATASET_NAME:-coco_pet_pretrained_small6}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/pretrained/r50_deformable_detr.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/exps/pet_baseline_diagnostics/coco_pretrained_small6_seed42_ddp}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"       # per GPU; total batch is 2 with two GPUs
NUM_WORKERS="${NUM_WORKERS:-2}"     # per process
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
    echo "Run tools/download_experiment_assets.sh first." >&2
    exit 1
fi
COCO_PATH="${DATA_ROOT}/${DATASET_NAME}"
if [[ ! -f "${COCO_PATH}/annotations/instances_train2017.json" || ! -f "${COCO_PATH}/annotations/instances_val2017.json" ]]; then
    echo "ERROR: prepared dataset not found: ${COCO_PATH}" >&2
    echo "Run tools/download_experiment_assets.sh first." >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}/training_complete.json" ]]; then
    echo "ERROR: output already contains a completed run: ${OUTPUT_DIR}" >&2
    echo "Choose another OUTPUT_DIR to avoid mixing logs." >&2
    exit 1
fi
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
    echo "ERROR: output directory is not empty: ${OUTPUT_DIR}" >&2
    echo "Choose another OUTPUT_DIR, or set ALLOW_EXISTING_OUTPUT=1 deliberately." >&2
    exit 1
fi
if ! command -v torchrun >/dev/null 2>&1; then
    echo "ERROR: torchrun was not found. Activate the tree-detr environment." >&2
    exit 1
fi

NUM_CLASSES="$("${PYTHON_BIN}" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["num_known"])' "${DATA_ROOT}/${DATASET_NAME}/split_metadata.json")"
mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONPATH="${PROJECT_ROOT}/models/ops${PYTHONPATH:+:${PYTHONPATH}}"

echo "Starting DDP baseline"
echo "  GPUs: ${GPU_LIST} (processes=${NPROC_PER_NODE}, batch_per_gpu=${BATCH_SIZE})"
echo "  data: ${COCO_PATH}"
echo "  output: ${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
    main.py \
    --coco_path "${COCO_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --pretrained "${CHECKPOINT}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --eval_interval 5 \
    --lr 2e-5 \
    --lr_backbone 2e-6 \
    --class_embed_lr_mult 10 \
    --backbone resnet50 \
    --num_classes "${NUM_CLASSES}" \
    --num_queries 300 \
    --enc_layers 6 \
    --dec_layers 6 \
    --dim_feedforward 1024 \
    --seed "${SEED}" \
    --device cuda
