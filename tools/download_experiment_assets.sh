#!/usr/bin/env bash
# Download the checkpoint and prepare the Oxford-IIIT Pet experiment split.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/oxford-pet-tree-detr}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/pretrained}"
DATASET_NAME="${DATASET_NAME:-coco_pet_pretrained_small6}"
SEED="${SEED:-42}"
KNOWN_PER_SPECIES="${KNOWN_PER_SPECIES:-3}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-100}"
VAL_PER_CLASS="${VAL_PER_CLASS:-30}"
UNKNOWN_PER_SPECIES="${UNKNOWN_PER_SPECIES:-4}"
INCREMENT_PER_CLASS="${INCREMENT_PER_CLASS:-100}"
CHECKPOINT="${CHECKPOINT:-main}"
OXFORD_MIRROR_BASE="${OXFORD_MIRROR_BASE:-https://thor.robots.ox.ac.uk/pets}"
OXFORD_IMAGES_URL="${OXFORD_IMAGES_URL:-${OXFORD_MIRROR_BASE%/}/images.tar.gz}"
OXFORD_ANNOTATIONS_URL="${OXFORD_ANNOTATIONS_URL:-${OXFORD_MIRROR_BASE%/}/annotations.tar.gz}"
CKPT_MIRROR_URL="${CKPT_MIRROR_URL:-}"
CKPT_MIRROR_BASE="${CKPT_MIRROR_BASE:-}"

usage() {
    echo "Usage: bash tools/download_experiment_assets.sh [options]"
    echo "  --data-root PATH          Oxford-Pet output root"
    echo "  --checkpoint-dir PATH     checkpoint directory"
    echo "  --dataset-name NAME       generated COCO split name"
    echo "  --checkpoint NAME         main, single_scale, single_scale_dc5, refine, two_stage"
    echo "  --seed N                  split seed"
    echo "Mirror variables: OXFORD_MIRROR_BASE, OXFORD_IMAGES_URL, OXFORD_ANNOTATIONS_URL"
    echo "                 CKPT_MIRROR_URL or CKPT_MIRROR_BASE"
    echo "All defaults can also be overridden with environment variables."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --dataset-name) DATASET_NAME="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --known-per-species) KNOWN_PER_SPECIES="$2"; shift 2 ;;
        --train-per-class) TRAIN_PER_CLASS="$2"; shift 2 ;;
        --val-per-class) VAL_PER_CLASS="$2"; shift 2 ;;
        --unknown-per-species) UNKNOWN_PER_SPECIES="$2"; shift 2 ;;
        --increment-per-class) INCREMENT_PER_CLASS="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if ! command -v gdown >/dev/null 2>&1; then
    echo "ERROR: gdown is not installed. Run tools/setup_linux.sh first." >&2
    exit 1
fi

case "${CHECKPOINT}" in
    main) CKPT_ID="1nDWZWHuRwtwGden77NLM9JoWe-YisJnA" ;;
    single_scale) CKPT_ID="1WEjQ9_FgfI5sw5OZZ4ix-OKk-IJ_-SDU" ;;
    single_scale_dc5) CKPT_ID="1m_TgMjzH7D44fbA-c_jiBZ-xf-odxGdk" ;;
    refine) CKPT_ID="1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI" ;;
    two_stage) CKPT_ID="15I03A7hNTpwuLNdfuEmW9_taZMNVssEp" ;;
    *) echo "ERROR: unsupported checkpoint: ${CHECKPOINT}" >&2; exit 2 ;;
esac

mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/r50_deformable_detr.pth"
if [[ -s "${CHECKPOINT_PATH}" ]]; then
    echo "Using existing checkpoint: ${CHECKPOINT_PATH}"
else
    PARTIAL="${CHECKPOINT_PATH}.part"
    rm -f "${PARTIAL}"
    if [[ -n "${CKPT_MIRROR_URL}" ]]; then
        echo "Downloading checkpoint from mirror: ${CKPT_MIRROR_URL}"
        if command -v aria2c >/dev/null 2>&1; then
            aria2c --allow-overwrite=true --continue=true --out="$(basename "${PARTIAL}")" --dir="$(dirname "${PARTIAL}")" "${CKPT_MIRROR_URL}"
        elif command -v curl >/dev/null 2>&1; then
            curl -L --fail --retry 3 --retry-delay 2 -C - -o "${PARTIAL}" "${CKPT_MIRROR_URL}"
        elif command -v wget >/dev/null 2>&1; then
            wget -c -O "${PARTIAL}" "${CKPT_MIRROR_URL}"
        else
            echo "ERROR: mirror download needs aria2c, curl, or wget." >&2
            exit 1
        fi
    elif [[ -n "${CKPT_MIRROR_BASE}" ]]; then
        CKPT_URL="${CKPT_MIRROR_BASE%/}/r50_deformable_detr_${CHECKPOINT}.pth"
        echo "Downloading checkpoint from mirror: ${CKPT_URL}"
        if command -v aria2c >/dev/null 2>&1; then
            aria2c --allow-overwrite=true --continue=true --out="$(basename "${PARTIAL}")" --dir="$(dirname "${PARTIAL}")" "${CKPT_URL}"
        elif command -v curl >/dev/null 2>&1; then
            curl -L --fail --retry 3 --retry-delay 2 -C - -o "${PARTIAL}" "${CKPT_URL}"
        elif command -v wget >/dev/null 2>&1; then
            wget -c -O "${PARTIAL}" "${CKPT_URL}"
        else
            echo "ERROR: mirror download needs aria2c, curl, or wget." >&2
            exit 1
        fi
    else
        echo "No checkpoint mirror configured; using Google Drive"
        gdown "https://drive.google.com/uc?id=${CKPT_ID}" -O "${PARTIAL}"
    fi
    if [[ ! -s "${PARTIAL}" ]]; then
        echo "ERROR: checkpoint download produced an empty file." >&2
        exit 1
    fi
    mv -f "${PARTIAL}" "${CHECKPOINT_PATH}"
fi

echo "Preparing Oxford-IIIT Pet split under ${DATA_ROOT}"
echo "  images source:      ${OXFORD_IMAGES_URL}"
echo "  annotations source: ${OXFORD_ANNOTATIONS_URL}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/data/prepare_oxford_pet.py" \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --unknown-per-species "${UNKNOWN_PER_SPECIES}" \
    --known-per-species "${KNOWN_PER_SPECIES}" \
    --train-per-class "${TRAIN_PER_CLASS}" \
    --val-per-class "${VAL_PER_CLASS}" \
    --unknown-val-per-class "${VAL_PER_CLASS}" \
    --increment-per-class "${INCREMENT_PER_CLASS}" \
    --increment-unknown-index 0 \
    --images-url "${OXFORD_IMAGES_URL}" \
    --annotations-url "${OXFORD_ANNOTATIONS_URL}" \
    --dataset-name "${DATASET_NAME}"

echo
echo "Assets ready"
echo "  checkpoint: ${CHECKPOINT_PATH}"
echo "  dataset:    ${DATA_ROOT}/${DATASET_NAME}"
